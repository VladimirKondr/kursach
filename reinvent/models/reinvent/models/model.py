"""Classical Reinvent de novo model

See:
https://doi.org/10.1186/s13321-017-0235-x (original publication)
https://doi.org/10.1021/acs.jcim.0c00915 (REINVENT 2.0)
"""

from __future__ import annotations
from typing import Any, Tuple, TypeVar, Iterator, TYPE_CHECKING

import torch
import torch.nn as tnn

from reinvent.models.reinvent.models import rnn, vocabulary as mv
from reinvent.models.reinvent.utils import collate_fn
from reinvent.models.model_mode_enum import ModelModeEnum

if TYPE_CHECKING:
    from reinvent.models.meta_data import ModelMetaData

M = TypeVar("M", bound="Model")


class Model:
    """
    Implements an RNN model using SMILES.
    """

    _model_type = "Reinvent"
    _version = 1

    def __init__(
        self,
        vocabulary: mv.Vocabulary,
        tokenizer: mv.SMILESTokenizer,
        meta_data: ModelMetaData,
        network_params: dict = None,
        max_sequence_length: int = 256,
        mode: str = "training",
        device=torch.device("cpu"),
    ):
        """
        Implements an RNN using either GRU or LSTM.

        :param vocabulary: vocabulary to use
        :param tokenizer: tokenizer to use
        :param meta_data: model meta data
        :param network_params: parameters required to initialize the RNN
        :param max_sequence_length: maximum length of sequence that can be generated
        :param mode: either "training" or "inference"
        :param device: the PyTorch device
        """

        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.meta_data = meta_data
        self.max_sequence_length = max_sequence_length
        self.device = device

        if not isinstance(network_params, dict):
            network_params = {}

        self._model_modes = ModelModeEnum()
        self.network = rnn.RNN(len(self.vocabulary), **network_params, device=self.device)
        self.set_mode(mode)

        self._nll_loss = tnn.NLLLoss(reduction="none")

    def set_mode(self, mode: str) -> None:
        """
        Set training or inference mode of the network.

        :param mode: Mode to be set.
        :raises ValueError: raised when unknown mode
        """

        if mode == self._model_modes.TRAINING:
            self.network.train()
        elif mode == self._model_modes.INFERENCE:
            self.network.eval()
        else:
            raise ValueError(f"Invalid model mode '{mode}")

    @classmethod
    def create_from_dict(cls: type[M], save_dict: dict, mode: str, device: torch.device) -> M:
        model_type = save_dict.get("model_type")

        if model_type and model_type != cls._model_type:
            raise RuntimeError(f"Wrong type: {model_type} but expected {cls._model_type}")

        if isinstance(save_dict["vocabulary"], dict):
            vocabulary = mv.Vocabulary.load_from_dictionary(save_dict["vocabulary"])
        else:
            vocabulary = save_dict["vocabulary"]

        model = cls(
            vocabulary=vocabulary,
            tokenizer=save_dict.get("tokenizer", mv.SMILESTokenizer()),
            meta_data=save_dict.get("metadata"),
            network_params=save_dict.get("network_params"),
            max_sequence_length=save_dict["max_sequence_length"],
            mode=mode,
            device=device,
        )

        model.network.load_state_dict(save_dict["network"])

        return model

    def get_save_dict(self):
        """Return the layout of the save dictionary"""

        save_dict = dict(
            model_type=self._model_type,
            version=self._version,
            metadata=self.meta_data,
            vocabulary=self.vocabulary.get_dictionary(),
            tokenizer=self.tokenizer,
            max_sequence_length=self.max_sequence_length,
            network=self.network.state_dict(),
            network_params=self.network.get_params(),
        )

        return save_dict

    def save(self, file_path: str) -> None:
        """Saves the model into a file

        :param file_path: Path to the model file.
        """

        save_dict = self.get_save_dict()
        torch.save(save_dict, file_path)

    save_to_file = save  # alias for backwards compatibility

    def likelihood_smiles(self, smiles: str) -> torch.Tensor:
        tokens = [self.tokenizer.tokenize(smile) for smile in smiles]
        encoded = [self.vocabulary.encode(token) for token in tokens]

        sequences = [
            torch.tensor(encode, dtype=torch.long, device=self.device) for encode in encoded
        ]
        padded_sequences = collate_fn(sequences)

        return self.likelihood(padded_sequences)

    def likelihood(self, sequences: torch.Tensor) -> torch.Tensor:
        """Retrieves the likelihood of a given sequence

        Used in training.

        :param sequences: a batch of sequences (batch_size, sequence_length)
        :returns: log likelihood for each example (batch_size)
        """

        logits, _ = self.network(sequences[:, :-1])  # all steps done at once
        log_probs = logits.log_softmax(dim=2)

        return self._nll_loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1)
    
    def likelihood_impala(self, sequences: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """Retrieves the likelihood of a given sequence

        Used in training.

        :param sequences: a batch of sequences (batch_size, sequence_length)
        :returns: log likelihood for each example (batch_size)
        """

        logits, _ = self.network(sequences[:, :-1])  # all steps done at once
        log_probs = logits.log_softmax(dim=2)

        return self._nll_loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1), log_probs

    # NOTE: needed for Reinvent TL

    @torch.no_grad()
    def sample(self, batch_size: int = 128) -> Tuple[torch.Tensor, list, torch.Tensor]:
        seqs, likelihoods = self._sample(batch_size=batch_size)

        # FIXME: this is potentially unnecessary in some cases
        smiles = [
            self.tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in seqs.cpu().numpy()
        ]

        return seqs, smiles, likelihoods

    @torch.no_grad()
    def sample_impala(self, batch_size: int = 128) -> Tuple[torch.Tensor, list, torch.Tensor, list[torch.Tensor]]:
        """Sample from RNN with IMPALA-specific data
        
        In addition to standard sampling, computes:
        - token_log_probs: log probability for each token in each sequence
        - token_nlls: scalar NLL for each sequence

        :param batch_size: batch size
        :returns: sequences, smiles, nlls, token_log_probs, token_nlls
        """
        seqs, likelihoods, token_log_probs = self._sample_impala(batch_size=batch_size)

        # FIXME: this is potentially unnecessary in some cases
        smiles = [
            self.tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in seqs.cpu().numpy()
        ]

        return seqs, smiles, likelihoods, token_log_probs
    
    def _compute_token_log_probs(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities for each token in a sequence.
        
        Args:
            sequence: Token sequence of shape [seq_len] including START token
            
        Returns:
            Log probabilities for each generated token (excluding START), shape [seq_len-1]
        """
        with torch.no_grad():
            # Skip the START token - we don't have a log prob for it
            # sequence[0] is START token, sequence[1:] are generated tokens
            if len(sequence) <= 1:
                return torch.tensor([], device=sequence.device)
            
            generated_tokens = sequence[1:]  # Exclude START token
            
            # Add batch dimension: [seq_len-1] -> [1, seq_len-1]
            seq_batch = generated_tokens.unsqueeze(0)
            
            # Get model output (logits)
            logits, _ = self.network(seq_batch)
            # logits shape: [1, seq_len-1, vocab_size]
            
            # Trim logits to match sequence length (needed for teacher forcing)
            # During teacher forcing, model outputs predictions for all input positions
            # We need to match each input token with its corresponding output logit
            min_len = min(logits.size(1), len(generated_tokens))
            logits = logits[:, :min_len, :]
            sequence_trimmed = generated_tokens[:min_len]
            
            log_probs = logits.log_softmax(dim=2)  # shape: [1, min_len, vocab_size]
            
            # Remove batch dimension: [1, min_len, vocab_size] -> [min_len, vocab_size]
            log_probs = log_probs.squeeze(dim=0)
            
            # Extract log prob for each actual token
            token_log_probs = log_probs[torch.arange(min_len), sequence_trimmed]
            
            return token_log_probs

    def _sample(self, batch_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a number of sequences from the RNN

        :param batch_size: batch size which is the number of sequences to sample
        :returns: sequences (2D) and associated NLLs (1D)
        """

        # NOTE: the first token never gets added in the loop so initialize with the start token
        sequences = [
            torch.full(
                (batch_size, 1),
                self.vocabulary[mv.START_TOKEN],
                dtype=torch.long,
                device=self.device,
            )
        ]
        input_vector = torch.full(
            (batch_size,), self.vocabulary[mv.START_TOKEN], dtype=torch.long, device=self.device
        )
        hidden_state = None
        nlls = torch.zeros(batch_size, device=self.device)

        for _ in range(self.max_sequence_length - 1):
            logits, hidden_state = self.network(input_vector.unsqueeze(1), hidden_state)
            logits = logits.squeeze(1)  # 2D
            log_probs = logits.log_softmax(dim=1)  # 2D
            probabilities = logits.softmax(dim=1)  # 2D
            input_vector = torch.multinomial(probabilities, num_samples=1).view(-1)  # 1D
            sequences.append(input_vector.view(-1, 1))
            nlls += self._nll_loss(log_probs, input_vector)

            if input_vector.sum() == 0:
                break

        concat_sequences = torch.cat(sequences, dim=1)

        return concat_sequences.data, nlls
    
    def _sample_impala(self, batch_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Sample a number of sequences from the RNN

        :param batch_size: batch size which is the number of sequences to sample
        :returns: sequences (2D) and associated NLLs (1D) and token_log_probs (list of 1D tensors, one per sample)
        """

        # NOTE: the first token never gets added in the loop so initialize with the start token
        sequences = [
            torch.full(
                (batch_size, 1),
                self.vocabulary[mv.START_TOKEN],
                dtype=torch.long,
                device=self.device,
            )
        ]
        input_vector = torch.full(
            (batch_size,), self.vocabulary[mv.START_TOKEN], dtype=torch.long, device=self.device
        )
        hidden_state = None
        nlls = torch.zeros(batch_size, device=self.device)

        # Store log probs for each token in each sequence
        # Will be list of [batch_size] tensors, then we'll split per sample
        all_token_log_probs = []

        for _ in range(self.max_sequence_length - 1):
            logits, hidden_state = self.network(input_vector.unsqueeze(1), hidden_state)
            logits = logits.squeeze(1)  # 2D: [batch_size, vocab_size]
            log_probs = logits.log_softmax(dim=1)  # 2D: [batch_size, vocab_size]
            probabilities = logits.softmax(dim=1)  # 2D
            input_vector = torch.multinomial(probabilities, num_samples=1).view(-1)  # 1D: [batch_size]
            sequences.append(input_vector.view(-1, 1))
            
            # Extract log prob for the chosen token for each sample
            # log_probs[i, input_vector[i]] gives log prob of chosen token for sample i
            chosen_log_probs = log_probs[torch.arange(batch_size), input_vector]  # [batch_size]
            all_token_log_probs.append(chosen_log_probs)
            
            nlls += self._nll_loss(log_probs, input_vector)

            if input_vector.sum() == 0:
                break

        concat_sequences = torch.cat(sequences, dim=1)
        
        # Convert all_token_log_probs from list of [batch_size] to list of [seq_len] per sample
        # all_token_log_probs is list of tensors, each [batch_size]
        # We need to transpose to get per-sample sequences
        token_log_probs_tensor = torch.stack(all_token_log_probs, dim=1)  # [batch_size, seq_len]
        token_log_probs = [token_log_probs_tensor[i] for i in range(batch_size)]  # List of [seq_len] tensors

        return concat_sequences.data, nlls, token_log_probs

    def get_network_parameters(self) -> Iterator[tnn.Parameter]:
        """
        Returns the configuration parameters of the network.

        :returns: network parameters of the RNN
        """

        return self.network.parameters()
