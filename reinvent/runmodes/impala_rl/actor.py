"""
Actor for IMPALA

Actors sample molecules using a frozen copy of the policy
and send trajectories to the learner.
"""

import logging
import torch
from typing import Callable
from dataclasses import dataclass

from reinvent.runmodes import create_adapter
from reinvent.runmodes.samplers.reinvent import ReinventSampler
from reinvent.runmodes.setup_sampler import setup_sampler
from reinvent.runmodes.utils import disable_gradients
from reinvent.runmodes.impala_rl.trajectory import Trajectory

logger = logging.getLogger(__name__)

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
        self.config = config
        self.actor_id = config.actor_id
        self.scoring_function = scoring_function
        self.device = torch.device(config.device)
        self.model_version = model_version
        
        logging.info(f"[Actor {self.actor_id}] Loading model on {self.device}...")
        
        self.adapter, self.save_dict, self.model_type = create_adapter(
            model_path,
            mode="inference",
            device=self.device
        )
        logging.info(f"[Actor {self.actor_id}] Model type: {self.model_type}")
        
        disable_gradients(self.adapter)
        
        sampler_params = {
            "batch_size": config.batch_size,
            "isomeric": config.isomeric,
            "randomize_smiles": config.randomize_smiles,
            "unique_sequences": config.unique_sequences,
        }
        self.sampler: ReinventSampler
        self.sampler, _ = setup_sampler(self.model_type, sampler_params, self.adapter)
        logging.info(f"[Actor {self.actor_id}] Sampler ready")
        
        TRANSFORMERS = ["Mol2Mol", "LinkinventTransformer", "LibinventTransformer", "Pepinvent"]
        if self.model_type in TRANSFORMERS:
            self.rdkit_smiles_flags = dict(sanitize=True, isomericSmiles=True)
        else:
            self.rdkit_smiles_flags = dict(allowTautomers=True)
        
        self.step_count = 0
        
        logging.info(f"[Actor {self.actor_id}] Ready")
    
    def collect_trajectory(self) -> list:
        """
        Collect trajectories for a batch of molecules
        
        Generates a batch of molecules, scores them, and uses pre-computed token-level
        log probabilities and NLLs for V-trace off-policy correction.
        
        Returns:
            List of Trajectory objects, one per molecule in batch
        """
        # Sample a batch of molecules with IMPALA-specific data
        # sampled.token_log_probs: list of [seq_len] tensors
        # sampled.token_nlls: [batch_size] tensor of scalar NLLs
        sampled = self.sampler.sample_impala([])
        
        # Score all molecules in the batch
        scores = self.scoring_function(sampled.smilies)
        
        # Return trajectories for each molecule in the batch
        trajectories = []
        for i in range(len(sampled.smilies)):
            # Use token sequences (tensor), not SMILES strings
            sequence = sampled.sequences[i]  # This is a tensor of token IDs
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
        
        # Log progress
        mean_reward = sum(scores) / len(scores)
        if self.step_count % 100 == 0:
            logging.info(
                f"[Actor {self.actor_id}] Steps: {self.step_count} | "
                f"Mean reward: {mean_reward:.3f}"
            )
        
        return trajectories
    
    def update_model(self, new_model_state: dict, model_version: int):
        """
        Update actor's model with new weights from learner
        
        Args:
            new_model_state: State dict from learner
        """
        self.adapter.model.network.load_state_dict(new_model_state)
        self.model_version = model_version
        logging.info(f"[Actor {self.actor_id}] Model updated at step {self.step_count} to version {self.model_version}")
    
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
