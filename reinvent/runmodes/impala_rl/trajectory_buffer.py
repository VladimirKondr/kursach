import random
from typing import Optional
from collections import deque

from reinvent.runmodes.impala_rl.trajectory import Trajectory

class TrajectoryBuffer:
    """
    Buffer for storing trajectories from multiple actors
    
    Used by learner to collect batches for V-trace updates.
    """
    
    def __init__(self, maxlen: Optional[int] = None):
        """
        Initialize buffer
        
        Args:
            maxlen: Maximum number of trajectories to store
        """
        self.buffer: deque = deque() if maxlen is None else deque(maxlen=maxlen)
        self.maxlen = maxlen
    
    def add(self, trajectory: Trajectory):
        """Add trajectory to buffer"""
        self.buffer.append(trajectory)
    
    def sample(self, batch_size: int) -> list[Trajectory]:
        """
        Sample trajectories from buffer
        
        Args:
            batch_size: Number of trajectories to sample
        
        Returns:
            List of sampled trajectories
        """
        if len(self.buffer) < batch_size:
            # Return all if not enough
            result = list(self.buffer)
            self.buffer.clear()
            return result

        # Random sampling without replacement: draw random indices,
        # remove them from the buffer to avoid training on the same
        # trajectory twice.  This breaks the LIFO temporal correlation
        # that caused loss variance to spike whenever actor timing changed.
        indices = sorted(random.sample(range(len(self.buffer)), batch_size), reverse=True)
        sampled: list[Trajectory] = []
        buf_list = list(self.buffer)
        for idx in sorted(indices):
            sampled.append(buf_list[idx])
        # Rebuild deque without sampled elements
        sampled_set = set(indices)
        self.buffer = deque(
            (item for i, item in enumerate(buf_list) if i not in sampled_set),
            maxlen=self.maxlen,
        )

        return sampled
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def __len__(self) -> int:
        """Number of trajectories in buffer"""
        return len(self.buffer)
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.maxlen is not None and len(self.buffer) >= self.maxlen
    
    @property
    def total_samples(self) -> int:
        """Total number of samples across all trajectories"""
        return sum(len(traj) for traj in self.buffer)
