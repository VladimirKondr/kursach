from __future__ import annotations
import asyncio
import math

from reinvent.runmodes.RL.reinvent import ReinventLearning

import logging
import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from reinvent.models.model_factory.sample_batch import SmilesState
from reinvent.models.reinvent.models.model import collate_fn
from reinvent.runmodes.RL.reward import dap_strategy
from reinvent.runmodes.impala_rl.nodes.learner_node import LearnerNode
from reinvent.runmodes.impala_rl.trajectory import Trajectory
from reinvent.scoring.results import ScoreResults
import torch
from reinvent.models.model_factory.sample_batch import SampleBatch

if TYPE_CHECKING:
    from reinvent.runmodes.RL import terminator_callable

# Telemetry imports
from reinvent.runmodes.impala_rl.telemetry import get_tracer, get_logger
from reinvent.runmodes.impala_rl.telemetry.traces import create_span
from reinvent.runmodes.impala_rl.telemetry.metrics import (
    record_duration_metric,
    record_value_metric,
)

logger = logging.getLogger(__name__)
telem_logger = get_logger(__name__)
tracer = get_tracer(__name__)

class ImpalaLearner(ReinventLearning):
    # Exponential moving average baseline for score centering.
    # Tracks the running mean of trajectory scores across all batches,
    # so the gradient signal reflects *absolute* quality improvement
    # rather than just relative ranking within a single batch.
    # alpha=0.05: slow-moving baseline (≈ 20-batch window).
    _score_ema: float = 0.5
    _score_ema_alpha: float = 0.05  # slower baseline tracking → sigma*(s-ema) stays non-zero longer

    def subscribe(self, learner_node: LearnerNode):
        self.learner_node = learner_node

    def score(self):
        """Override parent's score() to use pre-computed rewards from trajectories.
        
        Returns:
            ScoreResults with smilies, total_scores from trajectories
        """
        smilies = [traj.smiles for traj in self.trajectories]
        total_scores = np.array([float(traj.reward) for traj in self.trajectories], dtype=np.float32)
        
        results = ScoreResults(
            smilies=smilies,
            total_scores=total_scores,
            completed_components=[]
        )
        
        return results

    def update(self, results, orig_smilies):
        """Override parent's update() to apply clipped IS off-policy correction.
        
        Computes importance weights to correct for trajectories generated
        by old policy (behavior policy) when updating current policy (target policy).
        
        Args:
            results: ScoreResults with pre-computed scores
            orig_smilies: Original SMILES strings
            
        Returns:
            Tuple of (agent_lls, prior_lls, augmented_nll, loss)
        """
        with create_span("learner.update", tracer=tracer) as span:
            update_start = time.perf_counter()
            
            # 1. Compute NLLs for current agent and prior policies.
            # Use original token-ID sequences (stored in each Trajectory) rather
            # than re-tokenizing canonical SMILES strings.  RDKit canonicalization
            # can produce tokens (e.g. [SH]) that are absent from the LSTM
            # vocabulary even though the molecule is chemically valid.  The
            # original sequences are guaranteed to contain only vocabulary tokens
            # because the model itself generated them.
            span.add_event("computing_likelihoods")
            device = self._state.agent.model.device
            seqs = [traj.sequence.to(device) for traj in self.trajectories]
            padded_seqs = collate_fn(seqs)
            agent_nlls = self._state.agent.likelihood(padded_seqs)
            prior_nlls = self.prior.likelihood(padded_seqs)
            
            # 2. Compute target log probs for each trajectory (using current agent model)
            span.add_event("computing_target_log_probs")
            self._compute_target_log_probs()
            
            # 3. Compute trajectory-level clipped IS weights for off-policy correction
            span.add_event("computing_importance_weights")
            importance_weights = self._compute_importance_weights()
            
            # 5. Call reward strategy with weighted scores
            valid_mask = np.argwhere(self.sampled.states == SmilesState.VALID).flatten()

            orig_smilies = orig_smilies
            scores = results.total_scores
            agent_nlls = agent_nlls
            prior_nlls = prior_nlls
            mask_idx = valid_mask
            agent = self._state.agent

            self._strategy: Callable = dap_strategy
            # Use the sigma configured via reward_nlls (e.g. sigma=80), NOT a
            # hardcoded 120 that overrides the caller's configuration every step.
            self._sigma = self.reward_nlls._sigma

            ### FROM RLReward.__call__()
            scores = torch.from_numpy(scores).to(prior_nlls)

            # Build a single boolean mask: valid SMILES AND non-NaN score.
            # - valid_mask: indices of VALID SMILES (np array of int indices)
            # - nan_idx: boolean tensor of NaN scores
            # Both must be False to include a trajectory in the loss.
            valid_bool = torch.zeros(len(scores), dtype=torch.bool)
            if len(valid_mask) > 0:
                valid_bool[torch.from_numpy(valid_mask.astype(int))] = True
            keep = valid_bool & ~torch.isnan(scores)

            scores_keep = scores[keep]
            if scores_keep.numel() == 0:
                # Nothing to train on this batch — skip update
                telem_logger.warning("update: no valid non-NaN trajectories in batch, skipping")
                dummy = torch.tensor(0.0, requires_grad=False)
                return agent_nlls, prior_nlls, agent_nlls, dummy

            # Track the EMA of scores for monitoring/logging only.
            # DO NOT subtract it from scores before passing to dap_strategy.
            # DAP uses a squared-error objective: aug_lls = prior_lls + σ·score.
            # Subtracting a baseline shifts the target itself (not just the
            # gradient variance), which causes aug_target → prior as EMA catches
            # up to mean scores, making loss increase as agent diverges.
            # Standard REINVENT DAP always uses raw scores here.
            batch_mean = float(scores_keep.mean().item())
            self._score_ema = (
                (1 - self._score_ema_alpha) * self._score_ema
                + self._score_ema_alpha * batch_mean
            )
            record_value_metric("learner.score_ema_baseline", self._score_ema)

            agent_lls = -agent_nlls[keep]
            prior_lls = -prior_nlls[keep]
            importance_weights = importance_weights[keep]

            loss, augmented_lls = self._strategy(
                agent_lls,
                scores_keep,  # raw scores — no EMA centering for DAP target
                prior_lls,
                self._sigma,
            )

            loss = loss * importance_weights.detach()

            if self.inception is not None:
                # Pass uncentered raw scores so inception can do its own
                # normalization; pass orig_smilies/prior_lls filtered by the
                # same combined keep mask (valid AND non-NaN).
                keep_np = keep.cpu().numpy()
                inception_result = self.inception(
                    np.array(orig_smilies)[keep_np],
                    scores_keep.cpu().numpy(),
                    prior_lls,
                )
                
                # Check if inception returned valid data (may be None if memory is empty)
                if inception_result is not None:
                    _orig_smilies, _scores, _prior_lls = inception_result

                    # Inception stores SMILES strings without original sequences.
                    # RDKit canonicalization may have introduced tokens (e.g. [SH])
                    # absent from the vocabulary.  Filter those out before calling
                    # likelihood_smiles so they don't crash the training loop.
                    _vocab = agent.model.vocabulary
                    _tokenizer = agent.model.tokenizer
                    _ok = [
                        all(t in _vocab for t in _tokenizer.tokenize(smi))
                        for smi in _orig_smilies
                    ]
                    if not any(_ok):
                        inception_result = None  # nothing usable this step
                    else:
                        _ok_arr = np.array(_ok)
                        _orig_smilies = [s for s, ok in zip(_orig_smilies, _ok) if ok]
                        _scores = np.array(_scores)[_ok_arr]
                        if isinstance(_prior_lls, torch.Tensor):
                            _prior_lls = _prior_lls[_ok_arr]
                        else:
                            _prior_lls = np.array(_prior_lls)[_ok_arr]

                if inception_result is not None:
                    _orig_smilies, _scores, _prior_lls = _orig_smilies, _scores, _prior_lls

                    # compute the agent NLLs for the _current_ state of the agent
                    lls = agent.likelihood_smiles(_orig_smilies)

                    if isinstance(lls, torch.Tensor):  # Reinvent
                        _agent_lls = -lls
                    else:  # all other models
                        _agent_lls = -lls.likelihood

                    inception_loss, _ = self._strategy(
                        _agent_lls,
                        torch.tensor(_scores).to(_agent_lls),
                        torch.tensor(_prior_lls).to(_agent_lls),
                        self._sigma,
                    )

                    loss = torch.cat((loss, inception_loss), 0)

            loss = loss.mean()
            
            span.add_event("backward_pass")
            self.reward_nlls._optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent explosion from off-policy trajectories
            torch.nn.utils.clip_grad_norm_(
                self._state.agent.model.network.parameters(), max_norm=1.0
            )
            self.reward_nlls._optimizer.step()

            ### FROM RLReward.__call__()
            
            # Record metrics
            update_duration = time.perf_counter() - update_start
            record_duration_metric("learner.update.duration", update_duration)
            record_value_metric("learner.loss", float(loss.item()))
            record_value_metric("learner.agent_nll.mean", float(agent_nlls.mean().item()))
            record_value_metric("learner.prior_nll.mean", float(prior_nlls.mean().item()))

            # Gap between DAP target and current agent (nats).
            # gap > 0: agent hasn't reached the target yet (normal early training).
            # gap ≈ 0: agent has converged to augmented distribution.
            # gap < 0: agent overshot the target.
            gap = augmented_lls - agent_lls  # shape: (batch,)
            record_value_metric("learner.target_gap.mean", float(gap.mean().item()))
            record_value_metric("learner.target_gap.abs_mean", float(gap.abs().mean().item()))

            # Staleness: how many learner updates ago were these trajectories
            # generated?  Large staleness means IS weights do heavy lifting.
            if hasattr(self, 'learner_node') and self.learner_node is not None:
                current_version = self.learner_node.model_version
                staleness = [
                    current_version - getattr(traj, 'model_version', current_version)
                    for traj in self.trajectories
                ]
                if staleness:
                    record_value_metric("learner.staleness.mean", float(np.mean(staleness)))
                    record_value_metric("learner.staleness.max", float(max(staleness)))
            
            # Record per-actor importance weights for monitoring individual actors
            for traj, weight in zip(self.trajectories, importance_weights):
                record_value_metric(
                    "learner.importance_weight",
                    float(weight.item()),
                    attributes={"actor_id": traj.actor_id}
                )
            
            # Record aggregate importance weight statistics
            record_value_metric("learner.importance_weights.mean", float(importance_weights.mean().item()))
            record_value_metric("learner.importance_weights.min", float(importance_weights.min().item()))
            record_value_metric("learner.importance_weights.max", float(importance_weights.max().item()))
            
            span.set_attribute("loss", float(loss.item()))
            span.set_attribute("batch_size", len(agent_nlls))

            return agent_lls, prior_lls, augmented_lls, loss
    
    def _compute_target_log_probs(self):
        """Compute target policy log probabilities for each trajectory.
        
        Uses current agent model to compute log probs for each token in sequences.
        Stores results in trajectory.target_log_prob field.
        """
        with create_span("learner.compute_target_log_probs", tracer=tracer) as span:
            start_time = time.perf_counter()
            
            for traj in self.trajectories:
                # Use agent model to compute log probs for this sequence
                # Delegate to model's method for computing token-level log probs
                target_log_probs = self._state.agent.model._compute_token_log_probs(traj.sequence)
                traj.target_log_prob = target_log_probs
                traj.target_nll = -target_log_probs.sum()
            
            duration = time.perf_counter() - start_time
            record_duration_metric("learner.compute_target_log_probs.duration", duration)
            span.set_attribute("num_trajectories", len(self.trajectories))
    
    def _compute_importance_weights(self, clip_rho=2.0, normalize=True):
        """Compute trajectory-level importance weights for off-policy correction.

        Because molecular generation is a bandit (reward is a scalar on the
        complete molecule, not per-step), we need only trajectory-level IS —
        NOT per-step V-trace credit assignment that IMPALA uses for MDPs.

        Each trajectory's weight is:
            log_w_i = log p_target(τ_i) - log p_behavior(τ_i)
                    = Σ_t log π_target(a_t) - Σ_t log π_behavior(a_t)

        Symmetric clipping in log-space:
            log_w_i = clamp(log_w_i, -log(clip_rho), +log(clip_rho))

        Upper clip  — prevents over-weighting very on-policy (fresh) trajectories.
        Lower clip  — prevents near-zero weights for stale trajectories; without it,
                      effective batch size collapses from 32 to ~5-8 when actors are
                      a few versions behind the learner.

        Self-normalized IS (normalize=True):
            w̃_i = w_i / (Σ_j w_j) * N    (reduces variance; slightly biased but
                                            consistent estimator)

        Args:
            clip_rho: symmetric clipping threshold in ratio space (default: 2.0)
            normalize: apply self-normalized IS (default: True)

        Returns:
            Tensor of importance weights [batch_size]
        """
        with create_span("learner.compute_importance_weights", tracer=tracer) as span:
            start_time = time.perf_counter()
            
            log_ratios = []
            
            for i, traj in enumerate(self.trajectories):
                if traj.target_log_prob is None:
                    raise ValueError("Trajectory missing target_log_prob. Call _compute_target_log_probs first.")
                
                # Compute log importance ratio for entire trajectory
                target_log_prob_sum = traj.target_log_prob.sum()
                behavior_log_prob_sum = traj.behavior_log_prob.sum()
                log_ratio = target_log_prob_sum - behavior_log_prob_sum
                log_ratios.append(log_ratio)
            
            # Convert log ratios to tensor
            log_ratios = torch.stack(log_ratios)
            
            # Step 1: Clip in log-space BEFORE any numerical tricks.
            # Symmetric clipping: clamp(log_ratio, [-log_rho, +log_rho])
            # Upper clip: prevents over-weighting fresh (near on-policy) trajectories.
            # Lower clip: CRITICAL for same-machine distributed setup.
            #   Without it, very stale actors (3-5 versions behind) produce
            #   log_ratio << 0 → IS weight → 0 → effective batch shrinks from 32
            #   to ~5-8 trajectories → loss variance explodes.
            #   Symmetric clipping keeps every trajectory's weight in [1/rho, rho],
            #   so all 32 trajectories always contribute meaningfully.
            log_clip_rho = math.log(clip_rho)
            clipped_log_ratios = torch.clamp(log_ratios, min=-log_clip_rho, max=log_clip_rho)

            # Numerical stability: subtract max before exp.
            # The shift cancels out during self-normalization.
            center = clipped_log_ratios.max()
            importance_weights = torch.exp(clipped_log_ratios - center)

            # Self-normalized IS: w̃_i = w_i / Σw_j * N
            if normalize:
                weight_sum = importance_weights.sum()
                if weight_sum > 0:
                    weights_final = importance_weights / weight_sum * len(importance_weights)
                else:
                    weights_final = torch.ones_like(importance_weights)
            else:
                weights_final = importance_weights
            
            duration = time.perf_counter() - start_time
            record_duration_metric("learner.compute_importance_weights.duration", duration)
            span.set_attribute("num_weights", len(weights_final))
            span.set_attribute("clip_rho", clip_rho)
            span.set_attribute("normalize", normalize)
            span.set_attribute("log_ratio_min", float(log_ratios.min().item()))
            span.set_attribute("log_ratio_max", float(log_ratios.max().item()))
            
            return weights_final.to(self._state.agent.device)

    def optimize(self, converged: terminator_callable) -> bool:
        step = -1
        scaffolds = None
        self.start_time = time.time()

        for step in range(self.max_steps):
            # Get trajectories from learner node
            self.trajectories: list[Trajectory] = asyncio.run(
                self.learner_node.GetTrajectories(batch_size=self.sampling_model.batch_size)
            )
            # Move trajectories to device
            self.trajectories = [traj.to(self._state.agent.device) for traj in self.trajectories]

            # Update SMILES memory
            self.smiles_memory.update([trajectory.smiles for trajectory in self.trajectories])

            # Convert trajectories to SampleBatch for compatibility with parent methods
            self.sampled = self._trajectories_to_sample_batch(self.trajectories)

            # Create masks based on SMILES states from trajectories
            states = np.array([traj.state for traj in self.trajectories])
            self.invalid_mask = np.where(states == SmilesState.INVALID, False, True)
            self.duplicate_mask = np.where(states == SmilesState.DUPLICATE, False, True)

            # Score using parent's score() method (which uses self.sampled)
            results = self.score()

            if self._state.diversity_filter:
                df_mask = np.where(self.invalid_mask, True, False)

                scaffolds = self._state.diversity_filter.update_score(
                    results.total_scores, results.smilies, df_mask
                )
            elif self.intrinsic_penalty:
                df_mask = np.where(self.invalid_mask, True, False)

                scaffolds = self.intrinsic_penalty.update_score(
                    results.total_scores, results.smilies, df_mask, self.sampled
                )

            # Prepare data for update
            orig = self.sampled.output # smiles

            agent_lls, prior_lls, augmented_nll, loss = self.update(results, orig)

            state_dict = self._state.as_dict()
            self._state_info.update(state_dict)

            nan_idx = np.isnan(results.total_scores)
            scores = results.total_scores[~nan_idx]
            mean_scores = scores.mean()

            self.report(
                step,
                mean_scores,
                scaffolds,
                score_results=results,
                agent_lls=agent_lls,
                prior_lls=prior_lls,
                augmented_nll=augmented_nll,
                loss=float(loss),
            )

            if converged(mean_scores, step):
                logger.info(f"Terminating early in {step = }")
                break

        if self.tb_reporter:  # FIXME: context manager?
            self.tb_reporter.flush()
            self.tb_reporter.close()

        if step >= self.max_steps - 1:
            return True

        return False
    
    def _trajectories_to_sample_batch(self, trajectories: list[Trajectory]):
        """Convert list of Trajectory to SampleBatch for compatibility with parent methods
        
        For Reinvent: items1=None (no input scaffold), items2=SMILES output
        
        Args:
            trajectories: List of Trajectory objects
            
        Returns:
            SampleBatch with data from trajectories
        """
        sequences = [traj.sequence for traj in trajectories]
        smiles = [traj.smiles for traj in trajectories]
        behavior_nlls = [traj.behavior_nll for traj in trajectories]
        nlls = torch.stack(behavior_nlls)
        states = np.array([traj.state for traj in trajectories])
        
        # Create SampleBatch for Reinvent (items1=None for generative model)
        sample_batch = SampleBatch(sequences, smiles, nlls)
        sample_batch.smilies = smiles
        sample_batch.states = states
        
        return sample_batch