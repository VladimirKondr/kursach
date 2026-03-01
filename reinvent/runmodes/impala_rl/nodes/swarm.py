"""
Swarm Manager for IMPALA distributed training.

Manages multiple actor nodes and coordinates model updates across the cluster.
"""

import asyncio
import logging
from typing import List, Optional
import nats
from nats.js import JetStreamContext

from .actor_node import ActorNode


logger = logging.getLogger(__name__)


class Swarm:
    """
    Swarm manager for coordinating IMPALA actors.
    
    Responsibilities:
    - Create and manage multiple actor nodes
    - Listen for model update signals from learner
    - Broadcast model updates to all actors
    - Coordinate actor lifecycle
    """
    
    def __init__(
        self,
        nats_url: str = "nats://localhost:4222",
        num_actors: int = 1,
        model_update_subject: str = "model.update",
        control_port: int = 8765,
    ):
        """
        Initialize Swarm manager.
        
        Args:
            nats_url: NATS server URL
            num_actors: Number of actor nodes to manage
            model_update_subject: NATS subject for model update notifications
            control_port: Port for control signals (unused in NATS version)
        """
        self.nats_url = nats_url
        self.num_actors = num_actors
        self.model_update_subject = model_update_subject
        self.control_port = control_port
        
        self.nc: Optional[nats.NATS] = None
        self.js: Optional[JetStreamContext] = None
        self.actors: List[ActorNode] = []
        self.running = False
        self.latest_model_version: int = 0  # Track latest model version for lag calculation

        logger.info(f"[Swarm] Initialized with {num_actors} actors")
    
    async def connect(self):
        """Connect to NATS server and setup subscriptions."""
        try:
            self.nc = await nats.connect(self.nats_url)
            self.js = self.nc.jetstream()
            
            # Subscribe to model update notifications
            await self.nc.subscribe(self.model_update_subject, cb=self._handle_model_update)
            
            logger.info(f"[Swarm] Connected to NATS at {self.nats_url}")
            logger.info(f"[Swarm] Listening for model updates on '{self.model_update_subject}'")
            
        except Exception as e:
            logger.error(f"[Swarm] Failed to connect to NATS: {e}")
            raise
    
    async def create_actors(
        self,
        model,
        sampler,
        scoring_function,
        publish_subject: str = "trajectories",
    ) -> List[ActorNode]:
        """
        Create and initialize actor nodes.
        
        Args:
            model: The generative model
            sampler: Sampler instance
            scoring_function: Scoring function for molecules
            publish_subject: NATS subject for publishing trajectories
            
        Returns:
            List of initialized actor nodes
        """
        logger.info(f"[Swarm] Creating {self.num_actors} actor(s)...")
        
        # NOTE: This method is currently not used - actors are created manually in tests
        # TODO: Update this method to match current ActorNode constructor signature
        logger.warning("[Swarm] create_actors() is deprecated and not functional")
        return self.actors
    
    async def _handle_model_update(self, msg):
        """
        Handle model update notification from learner.

        Pull-based design: swarm does NOT push/broadcast the model to actors.
        Instead it records the latest {version, bucket, key} directly on each
        ActorNode.  Every actor pulls the model on its own schedule, right after
        finishing a trajectory batch (in SendTrajectories → _maybe_pull_model).
        This eliminates broadcast timeouts, lock contention, and version skipping.
        """
        try:
            import json
            data = json.loads(msg.data.decode())
            version = data["version"]
            bucket = data["bucket"]
            key = data["key"]

            self.latest_model_version = version
            update_info = {"version": version, "bucket": bucket, "key": key}

            # Propagate to all actor nodes so they can pull when ready.
            for actor in self.actors:
                actor._swarm_latest_version = version
                actor._latest_model_info = update_info

            logger.info(
                f"[Swarm] Model v{version} available — "
                f"{len(self.actors)} actors will pull on next cycle"
            )

        except Exception as e:
            logger.error(f"[Swarm] Error handling model update: {e}")
    
    async def start(self):
        """Start the swarm and keep it running."""
        self.running = True
        logger.info("[Swarm] Started and listening for updates")
        
        # Keep running until explicitly stopped
        while self.running:
            await asyncio.sleep(1)
    
    async def stop(self):
        """Stop the swarm and close all actors."""
        logger.info("[Swarm] Stopping...")
        self.running = False
        
        # Close all actors
        close_tasks = [actor.Close() for actor in self.actors]
        await asyncio.gather(*close_tasks, return_exceptions=True)
        
        # Close NATS connection
        if self.nc:
            await self.nc.close()
        
        logger.info("[Swarm] Stopped")
    
    async def get_actor_stats(self) -> dict:
        """
        Get statistics from all actors.
        
        Returns:
            Dictionary with actor statistics
        """
        stats = {
            "num_actors": len(self.actors),
            "actors": []
        }
        
        for actor in self.actors:
            actor_stats = {
                "id": actor.worker_id,
                "model_version": actor.actor.model_version,
                "step_count": actor.actor.step_count,
            }
            stats["actors"].append(actor_stats)
        
        return stats
