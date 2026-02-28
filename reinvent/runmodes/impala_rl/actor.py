"""
Actor for IMPALA

Actors sample molecules using a frozen copy of the policy
and send trajectories to the learner.
"""

import logging
import time
import numpy as np
import torch
from typing import Callable
from dataclasses import dataclass

from reinvent.runmodes import create_adapter
from reinvent.runmodes.samplers.reinvent import ReinventSampler
from reinvent.runmodes.setup_sampler import setup_sampler
from reinvent.runmodes.utils import disable_gradients
from reinvent.runmodes.impala_rl.trajectory import Trajectory

# Telemetry imports
from .telemetry import get_tracer, get_logger
from .telemetry.traces import create_span
from .telemetry.metrics import (
    record_duration_metric,
    record_value_metric,
    increment_counter_metric,
)

logger = logging.getLogger(__name__)
# Get telemetry logger (structured with trace context)
telem_logger = get_logger(__name__)
# Get tracer for creating spans
tracer = get_tracer(__name__)

@dataclass
class ActorConfig:
    """Configuration for IMPALA Actor"""
    actor_id: int = 0
    batch_size: int = 128
    device: str = "cpu"
    isomeric: bool = False
    randomize_smiles: bool = True
    unique_sequences: bool = False


class ImpalaActor:
    """
    IMPALA Actor
    
    Samples molecules using behavior policy and computes rewards.
    Sends trajectories to learner for training.
    """
    
    def __init__(
        self,
        config: ActorConfig,
        model_path: str,
        model_version: int,
        scoring_function: Callable,
    ):
        """
        Initialize actor
        
        Args:
            config: Actor configuration
            model_path: Path to initial model checkpoint
            scoring_function: Callable that takes List[str] and returns scores
        """
        with create_span("actor.initialize", tracer=tracer, attributes={"actor_id": config.actor_id}) as span:
            self.config = config
            self.actor_id = config.actor_id
            self.scoring_function = scoring_function
            self.device = torch.device(config.device)
            self.model_version = model_version
            
            telem_logger.info(
                "Actor initializing",
                attributes={
                    "actor_id": self.actor_id,
                    "device": str(self.device),
                    "batch_size": config.batch_size,
                    "model_version": model_version,
                }
            )
            
            span.add_event("loading_model")
            self.adapter, self.save_dict, self.model_type = create_adapter(
                model_path,
                mode="inference",
                device=self.device
            )
            span.set_attribute("model_type", self.model_type)
            
            disable_gradients(self.adapter)
            
            span.add_event("setting_up_sampler")
            sampler_params = {
                "batch_size": config.batch_size,
                "isomeric": config.isomeric,
                "randomize_smiles": config.randomize_smiles,
                "unique_sequences": config.unique_sequences,
            }
            self.sampler: ReinventSampler
            self.sampler, _ = setup_sampler(self.model_type, sampler_params, self.adapter)
            
            TRANSFORMERS = ["Mol2Mol", "LinkinventTransformer", "LibinventTransformer", "Pepinvent"]
            if self.model_type in TRANSFORMERS:
                self.rdkit_smiles_flags = dict(sanitize=True, isomericSmiles=True)
            else:
                self.rdkit_smiles_flags = dict(allowTautomers=True)
            
            self.step_count = 0
            
            telem_logger.info(
                "Actor initialized successfully",
                attributes={
                    "actor_id": self.actor_id,
                    "model_type": self.model_type,
                }
            )
    
    def collect_trajectory(self):
        """
        Collect a batch of trajectories by sampling and scoring molecules
        
        Returns:
            List[Trajectory]: Batch of trajectories
        """
        with create_span(
            "actor.collect_trajectory",
            tracer=tracer,
            attributes={
                "actor_id": self.actor_id,
                "batch_size": self.config.batch_size,
                "model_version": self.model_version,
            }
        ) as span:
            start_time = time.perf_counter()
            
            # Sample molecules
            span.add_event("sampling_molecules")
            sample_start = time.perf_counter()
            sampled = self.sampler.sample_impala([])
            sample_duration = time.perf_counter() - sample_start
            
            # Record sampling metrics
            record_duration_metric(
                "actor.sampling.duration",
                sample_duration,
                attributes={"actor_id": self.actor_id, "batch_size": len(sampled.smilies)}
            )
            
            # Score molecules
            span.add_event("scoring_molecules")
            score_start = time.perf_counter()
            scores = self.scoring_function(sampled.smilies)
            score_duration = time.perf_counter() - score_start
            
            # Record scoring metrics
            record_duration_metric(
                "actor.scoring.duration",
                score_duration,
                attributes={"actor_id": self.actor_id, "batch_size": len(sampled.smilies)}
            )
            
            # Create trajectories
            span.add_event("creating_trajectories")
            trajectories = []
            for i in range(len(sampled.smilies)):
                sequence = sampled.sequences[i]
                token_log_probs = sampled.token_log_probs[i]
                behavior_nll = sampled.nlls[i]
                
                trajectory = Trajectory(
                    sequence=sequence,
                    behavior_log_prob=token_log_probs,
                    target_log_prob=None,
                    behavior_nll=behavior_nll,
                    target_nll=None,
                    reward=torch.tensor(scores[i], dtype=torch.float32, device=self.device),
                    smiles=sampled.smilies[i],
                    state=sampled.states[i],
                    actor_id=self.actor_id,
                    step=self.step_count,
                    model_version=self.model_version
                )
                trajectories.append(trajectory)
                self.step_count += 1
            
            # Calculate metrics
            mean_reward = float(np.mean(scores))
            max_reward = float(np.max(scores))
            min_reward = float(np.min(scores))
            
            # Calculate validity ratio
            from reinvent.models.model_factory.sample_batch import SmilesState
            valid_count = sum(1 for s in sampled.states if s == SmilesState.VALID)
            validity_ratio = valid_count / len(sampled.states) if len(sampled.states) > 0 else 0.0
            
            total_duration = time.perf_counter() - start_time
            
            # Record comprehensive metrics
            attributes = {
                "actor_id": self.actor_id,
                "model_version": self.model_version,
            }
            
            record_duration_metric("actor.trajectory_generation.duration", total_duration, attributes)
            record_value_metric("actor.reward.mean", mean_reward, attributes)
            record_value_metric("actor.reward.max", max_reward, attributes)
            record_value_metric("actor.reward.min", min_reward, attributes)
            record_value_metric("actor.smiles.valid.ratio", validity_ratio, attributes)
            increment_counter_metric(
                "actor.trajectories_generated.total",
                len(trajectories),
                attributes
            )
            
            # Add metrics to span
            span.set_attribute("mean_reward", mean_reward)
            span.set_attribute("validity_ratio", validity_ratio)
            span.set_attribute("trajectory_count", len(trajectories))
            
            # Log progress
            if self.step_count % 100 == 0:
                telem_logger.info(
                    "Actor progress",
                    attributes={
                        "actor_id": self.actor_id,
                        "steps": self.step_count,
                        "mean_reward": mean_reward,
                        "validity_ratio": validity_ratio,
                    }
                )
        
        return trajectories
    
    def update_model(self, new_model_state: dict, model_version: int):
        """
        Update actor's model with new weights from learner
        
        Args:
            new_model_state: State dict from learner
            model_version: Version number of the new model
        """
        with create_span(
            "actor.update_model",
            tracer=tracer,
            attributes={
                "actor_id": self.actor_id,
                "old_version": self.model_version,
                "new_version": model_version,
            }
        ) as span:
            start_time = time.perf_counter()
            
            old_version = self.model_version
            version_lag = model_version - old_version
            
            self.adapter.model.network.load_state_dict(new_model_state)
            self.model_version = model_version
            
            duration = time.perf_counter() - start_time
            
            # Record metrics
            attributes = {"actor_id": self.actor_id}
            record_duration_metric("actor.model_update.duration", duration, attributes)
            record_value_metric("model.version.current", model_version, attributes)
            record_value_metric("model.version.lag", version_lag, attributes)
            
            span.set_attribute("duration", duration)
            span.set_attribute("version_lag", version_lag)
            
            telem_logger.info(
                "Model updated",
                attributes={
                    "actor_id": self.actor_id,
                    "step": self.step_count,
                    "old_version": old_version,
                    "new_version": model_version,
                    "version_lag": version_lag,
                    "duration_seconds": duration,
                }
            )
    
    def run(self, num_trajectories: int):
        """
        Run actor to collect trajectories
        
        This is a placeholder - in real IMPALA, actors run in separate processes
        and communicate with learner via queues/RPC.
        
        Args:
            num_trajectories: Number of trajectories to collect
        """
        trajectories = []
        
        for _ in range(num_trajectories):
            traj = self.collect_trajectory()
            trajectories.append(traj)
        
        return trajectories
