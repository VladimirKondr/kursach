import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class Trajectory:
    """
    Single trajectory for IMPALA
    
    Contains data for ONE generated molecule with all metadata needed
    for V-trace off-policy correction.
    """
    # Single sequence (token IDs) for one molecule
    sequence: torch.Tensor  # (seq_len,)
    
    # Behavior policy log probability for EACH TOKEN in the sequence
    behavior_log_prob: torch.Tensor  # (seq_len,) - log prob of each token
    
    # Target policy log probability for each token (computed by learner during training)
    target_log_prob: Optional[torch.Tensor]  # (seq_len,) - log prob of each token
    
    # Scalar NLL (negative log likelihood) for behavior policy
    # = -sum(behavior_log_prob) = sum(-log_probs)
    behavior_nll: Optional[torch.Tensor]  # scalar
    
    # Scalar NLL for target policy (computed by learner)
    # = -sum(target_log_prob) = sum(-log_probs)
    target_nll: Optional[torch.Tensor]  # scalar
    
    # Reward (score from scoring function) for this molecule
    reward: torch.Tensor  # scalar
    
    # SMILES string (for logging and reference)
    smiles: str
    
    # SMILES state (VALID, INVALID, DUPLICATE) - from sampler
    state: Optional[str]
    
    # Model version that generated this trajectory
    model_version: int
    
    # Metadata
    actor_id: int
    step: int
    
    def to(self, device: torch.device) -> 'Trajectory':
        """Move trajectory to device"""
        return Trajectory(
            sequence=self.sequence.to(device),
            behavior_log_prob=self.behavior_log_prob.to(device),
            target_log_prob=self.target_log_prob.to(device) if self.target_log_prob is not None else None,
            behavior_nll=self.behavior_nll.to(device) if self.behavior_nll is not None else None,
            target_nll=self.target_nll.to(device) if self.target_nll is not None else None,
            reward=self.reward.to(device),
            smiles=self.smiles,
            state=self.state,
            model_version=self.model_version,
            actor_id=self.actor_id,
            step=self.step
        )
    
    def is_valid(self) -> bool:
        """Check if trajectory contains a valid SMILES"""
        from reinvent.models.model_factory.sample_batch import SmilesState
        return self.state == SmilesState.VALID
    
    def to_dict(self) -> dict:
        """Convert trajectory to JSON-serializable dictionary"""
        from reinvent.models.model_factory.sample_batch import SmilesState
        
        # Convert state enum to string name
        state_value = self.state.name if isinstance(self.state, SmilesState) else self.state
        
        result = {
            'sequence': self.sequence.tolist() if isinstance(self.sequence, torch.Tensor) else self.sequence,
            'behavior_log_prob': self.behavior_log_prob.tolist() if isinstance(self.behavior_log_prob, torch.Tensor) else self.behavior_log_prob,
            'target_log_prob': self.target_log_prob.tolist() if self.target_log_prob is not None and isinstance(self.target_log_prob, torch.Tensor) else self.target_log_prob,
            'behavior_nll': float(self.behavior_nll) if self.behavior_nll is not None else None,
            'target_nll': float(self.target_nll) if self.target_nll is not None else None,
            'reward': float(self.reward) if isinstance(self.reward, torch.Tensor) else self.reward,
            'smiles': self.smiles,
            'state': state_value,
            'model_version': self.model_version,
            'actor_id': self.actor_id,
            'step': self.step
        }
        
        return result
    
    @classmethod
    def from_dict(cls, data: dict, device: torch.device = torch.device('cpu')) -> 'Trajectory':
        """Create trajectory from dictionary"""
        import json
        
        # Helper function to safely parse data that might be string or list
        def parse_field(field_data):
            if field_data is None:
                return None
            if isinstance(field_data, str):
                # If it's a JSON string, parse it
                try:
                    return json.loads(field_data)
                except (json.JSONDecodeError, TypeError):
                    # If not valid JSON, it might be a regular string (like SMILES)
                    return field_data
            return field_data
        
        sequence_data = parse_field(data['sequence'])
        behavior_log_prob_data = parse_field(data['behavior_log_prob'])
        target_log_prob_data = parse_field(data.get('target_log_prob'))
        
        return cls(
            sequence=torch.tensor(sequence_data, dtype=torch.long, device=device),
            behavior_log_prob=torch.tensor(behavior_log_prob_data, dtype=torch.float32, device=device),
            target_log_prob=torch.tensor(target_log_prob_data, dtype=torch.float32, device=device) if target_log_prob_data is not None else None,
            behavior_nll=torch.tensor(data['behavior_nll'], dtype=torch.float32, device=device) if data.get('behavior_nll') is not None else None,
            target_nll=torch.tensor(data['target_nll'], dtype=torch.float32, device=device) if data.get('target_nll') is not None else None,
            reward=torch.tensor(data['reward'], dtype=torch.float32, device=device),
            smiles=data['smiles'],
            state=data['state'],
            model_version=data['model_version'],
            actor_id=data['actor_id'],
            step=data['step']
        )
    