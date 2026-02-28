from __future__ import annotations
import asyncio

from reinvent.runmodes.RL.reinvent import ReinventLearning

import logging
import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from reinvent.models.model_factory.sample_batch import SmilesState
from reinvent.runmodes.RL.reward import dap_strategy
from reinvent.runmodes.impala_rl.nodes.learner_node import LearnerNode
from reinvent.runmodes.impala_rl.trajectory import Trajectory
from reinvent.scoring.results import ScoreResults
import torch
from reinvent.models.model_factory.sample_batch import SampleBatch

if TYPE_CHECKING:
    from reinvent.runmodes.RL import terminator_callable

logger = logging.getLogger(__name__)

class ImpalaLearner(ReinventLearning):
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
        """Override parent's update() to apply V-trace off-policy correction.
        
        Computes importance weights to correct for trajectories generated
        by old policy (behavior policy) when updating current policy (target policy).
        
        Args:
            results: ScoreResults with pre-computed scores
            orig_smilies: Original SMILES strings
            
        Returns:
            Tuple of (agent_lls, prior_lls, augmented_nll, loss)
        """
        # 1. Compute NLLs for current agent and prior policies
        agent_nlls = self._state.agent.likelihood_smiles(self.sampled.items2)
        prior_nlls = self.prior.likelihood_smiles(self.sampled.items2)
        
        # 2. Compute target log probs for each trajectory (using current agent model)
        self._compute_target_log_probs()
        
        # 3. Compute V-trace importance weights for off-policy correction
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
        self._sigma = 120

        ### FROM RLReward.__call__()
        scores = torch.from_numpy(scores).to(prior_nlls)

        # FIXME: move NaN filtering before first use of scores in learning
        # FIXME: reconsider NaN/failure handling
        nan_idx = torch.isnan(scores)
        scores_nonnan = scores[~nan_idx]
        agent_lls = -agent_nlls[~nan_idx]  # negated because we need to minimize
        prior_lls = -prior_nlls[~nan_idx]
        importance_weights = importance_weights[~nan_idx]

        loss, augmented_lls = self._strategy(
            agent_lls,
            scores_nonnan,
            prior_lls,
            self._sigma,
        )

        loss = loss * importance_weights.detach()

        if self.inception is not None:
            inception_result = self.inception(
                np.array(orig_smilies)[mask_idx], scores_nonnan[mask_idx], prior_lls[mask_idx]
            )
            
            # Check if inception returned valid data (may be None if memory is empty)
            if inception_result is not None:
                _orig_smilies, _scores, _prior_lls = inception_result

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
        
        self.reward_nlls._optimizer.zero_grad()
        loss.backward()
        self.reward_nlls._optimizer.step()

        ### FROM RLReward.__call__()

        return agent_lls, prior_lls, augmented_lls, loss
    
    def _compute_target_log_probs(self):
        """Compute target policy log probabilities for each trajectory.
        
        Uses current agent model to compute log probs for each token in sequences.
        Stores results in trajectory.target_log_prob field.
        """
        for traj in self.trajectories:
            # Use agent model to compute log probs for this sequence
            # Delegate to model's method for computing token-level log probs
            target_log_probs = self._state.agent.model._compute_token_log_probs(traj.sequence)
            traj.target_log_prob = target_log_probs
            traj.target_nll = -target_log_probs.sum()
    
    def _compute_importance_weights(self, clip_rho=10.0, normalize=True):
        """Compute importance weights for off-policy correction using Clipped IS.
        
        Clipped Importance Sampling:
            w_i = min(clip_rho, π_target(τ_i) / π_behavior(τ_i))
            
        With normalization (Self-normalized IS):
            w̃_i = w_i / Σ_j w_j
            
        This prevents weights from being too small and provides more stable learning.
        
        Args:
            clip_rho: Clipping parameter for importance weight (default: 10.0)
            normalize: Whether to normalize weights (default: True)
            
        Returns:
            Tensor of importance weights [batch_size]
        """
        log_ratios = []
        
        for i, traj in enumerate(self.trajectories):
            if traj.target_log_prob is None:
                raise ValueError("Trajectory missing target_log_prob. Call _compute_target_log_probs first.")
            
            # Compute importance ratio for entire trajectory
            target_log_prob_sum = traj.target_log_prob.sum()
            behavior_log_prob_sum = traj.behavior_log_prob.sum()
            log_ratio = target_log_prob_sum - behavior_log_prob_sum
            log_ratios.append(log_ratio)
        
        # Convert log ratios to tensor for stable computation
        log_ratios = torch.stack(log_ratios)
        
        # For numerical stability, use log-sum-exp trick
        # w = exp(log_ratio - max(log_ratios)) to avoid overflow/underflow
        max_log_ratio = log_ratios.max()
        importance_ratios = torch.exp(log_ratios - max_log_ratio)
        
        # Clip importance weights
        clipped_weights = torch.clamp(importance_ratios, min=0.0, max=clip_rho)
        
        # Normalize weights if requested
        if normalize:
            weight_sum = clipped_weights.sum()
            if weight_sum > 0:
                normalized_weights = clipped_weights / weight_sum * len(clipped_weights)
            else:
                # If all weights are zero, use uniform weights
                normalized_weights = torch.ones_like(clipped_weights)
            weights_final = normalized_weights
        else:
            weights_final = clipped_weights
        
        return weights_final.to(self._state.agent.device)

    def optimize(self, converged: terminator_callable) -> bool:
        step = -1
        scaffolds = None
        self.start_time = time.time()

        for step in range(self.max_steps):
            # Get trajectories from learner node
            self.trajectories: list[Trajectory] = asyncio.run(
                self.learner_node.GetTrajectories(self.sampling_model.batch_size)
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