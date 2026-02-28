"""Reinvent adapter"""

__all__ = ["ReinventAdapter"]
from typing import List

import torch

from .sample_batch import SampleBatch
from reinvent.models.model_factory.model_adapter import ModelAdapter


class ReinventAdapter(ModelAdapter):
    def likelihood(self, sequences: torch.Tensor) -> torch.Tensor:
        return self.model.likelihood(sequences)

    def likelihood_smiles(self, smiles: List[str]) -> torch.Tensor:
        return self.model.likelihood_smiles(smiles)

    def sample(self, batch_size: int) -> SampleBatch:
        """Sample from Model

        :param batch_size: batch size
        :returns: token sequences, list of SMILES, NLLs
        """
        # torch.Tensor, List[str], torch.Tensor
        sequences, smilies, nlls = self.model.sample(batch_size)

        # NOTE: keep the sequences and nlls as Tensor as they are needed for
        #       later computations
        return SampleBatch(sequences, smilies, nlls)
    
    def sample_impala(self, batch_size: int) -> SampleBatch:
        """Sample molecules for IMPALA with token sequences and log probs"""
        sequences, smiles, nlls, token_log_probs = self.model.sample_impala(batch_size)
        
        # For Reinvent, items1 is None, items2 is SMILES list
        sample_batch = SampleBatch(None, smiles, nlls)
        
        # Set IMPALA-specific fields
        sample_batch.sequences = sequences  # List of token tensors
        sample_batch.token_log_probs = token_log_probs  # List of log prob tensors
        sample_batch.smilies = smiles  # Already set via items2, but explicit is better
        
        return sample_batch
        
        
