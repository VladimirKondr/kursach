#!/usr/bin/env python
"""
Full IMPALA Integration Test: Actor(s) + Learner + NATS
Tests complete training loop with trajectory generation, NATS messaging,
scoring, V-trace correc    # ==================== Setup NATS ====================
    logger.info("\n📡 Setting up NATS...")
    if not await setup_nats(NATS_URL):
        logger.error("❌ Failed to setup NATS. Make sure NATS server is running with JetStream.")
        logger.error("   Start NATS with: nats-server --jetstream -D")
        return Falseand model updates.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List

import nats
import torch
import numpy as np
from nats.errors import NoServersError

from reinvent.runmodes.impala_rl.actor import ImpalaActor, ActorConfig
from reinvent.runmodes.impala_rl.learner import ImpalaLearner
from reinvent.runmodes.impala_rl.nodes.actor_node import ActorNode
from reinvent.runmodes.impala_rl.nodes.learner_node import LearnerNode
from reinvent.runmodes.impala_rl.nodes.swarm import Swarm
from reinvent.runmodes.RL.data_classes import ModelState
from reinvent.runmodes.RL.memories import Inception
from reinvent.runmodes.RL.memories.identical_murcko_scaffold import IdenticalMurckoScaffold
from reinvent.runmodes.RL.reward import RLReward, dap_strategy
from reinvent.runmodes.create_adapter import create_adapter
from reinvent.runmodes.setup_sampler import setup_sampler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class QEDScorer:
    """Simple QED (drug-likeness) scoring function"""
    
    def __call__(self, smiles_list: List[str]) -> np.ndarray:
        """Score molecules by QED"""
        from rdkit import Chem
        from rdkit.Chem import QED
        
        scores = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    score = QED.qed(mol)
                else:
                    score = 0.0
            except Exception:
                score = 0.0
            scores.append(score)
        
        return np.array(scores, dtype=np.float32)


async def setup_nats(nats_url: str = "localhost:4222", max_retries: int = 5) -> bool:
    """Setup NATS server and JetStream"""
    for attempt in range(max_retries):
        try:
            nc = await nats.connect(nats_url)
            js = nc.jetstream()
            
            # Delete existing stream if it exists (for clean test)
            try:
                await js.delete_stream("impala_trajectories")
                logger.info("🗑️ Deleted existing stream")
            except Exception:
                pass  # Stream doesn't exist, that's OK
            
            # Create stream for trajectories
            try:
                await js.add_stream(
                    name="impala_trajectories",
                    subjects=["jobs.result"],
                    max_msgs=10000,
                    storage="memory"  # Use memory storage for testing
                )
                logger.info("✅ JetStream stream created")
            except Exception as e:
                logger.error(f"❌ Failed to create stream: {e}")
                await nc.close()
                return False
            
            await nc.close()
            return True
        except NoServersError:
            logger.warning(f"⚠️ NATS not available (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
            continue
    
    logger.error("❌ Could not connect to NATS")
    return False


async def run_actor_loop(actor_node: ActorNode, num_iterations: int = 5):
    """Run actor to generate trajectories"""
    logger.info("🎬 Starting Actor Loop")
    
    for i in range(num_iterations):
        logger.info(f"📤 Actor iteration {i + 1}/{num_iterations}")
        await actor_node.SendTrajectories()
        await asyncio.sleep(0.5)  # Small delay between iterations
    
    await actor_node.Close()
    logger.info("✅ Actor Loop Complete")


async def run_learner_loop(learner: ImpalaLearner, learner_node: LearnerNode, num_iterations: int = 5):
    """Run learner to process trajectories and update model"""
    logger.info("📚 Starting Learner Loop")
    
    for i in range(num_iterations):
        logger.info(f"📥 Learner iteration {i + 1}/{num_iterations}")
        
        # Get trajectories from NATS
        trajectories = await learner_node.GetTrajectories(timeout=10.0)
        
        if trajectories:
            logger.info(f"  Received {len(trajectories)} trajectories")
            
            # Set trajectories in learner
            learner.trajectories = trajectories
            
            # Convert trajectories to sample batch
            learner.sampled = learner._trajectories_to_sample_batch(trajectories)
            
            # Score (uses pre-computed rewards from trajectories)
            results = learner.score()
            
            # Update model with V-trace correction
            orig_smilies = [traj.smiles for traj in trajectories]
            agent_lls, prior_lls, augmented_lls, loss = learner.update(results, orig_smilies)
            
            logger.info(f"  Loss: {loss:.4f}")
            logger.info(f"  Mean reward: {results.total_scores.mean():.4f}")
            logger.info(f"  Agent NLLs: {agent_lls.mean():.4f}")
            
            # Commit model to NATS and notify swarm
            logger.info(f"  💾 Committing model version {learner_node.model_version + 1}")
            await learner_node.commit_model(learner._state.agent.model)
            
        else:
            logger.warning("  No trajectories received (timeout)")
        
        await asyncio.sleep(0.5)
    
    await learner_node.close()
    logger.info("✅ Learner Loop Complete")


async def main():
    """Main integration test"""
    logger.info("=" * 70)
    logger.info("IMPALA Full Integration Test: Actors + Learner + NATS")
    logger.info("=" * 70)
    
    # Configuration
    NATS_URL = "localhost:4222"
    DEVICE = "cpu"  # Use CPU for testing
    BATCH_SIZE = 16  # Small batch for testing
    NUM_ACTORS = 5
    NUM_ITERATIONS = 15  # Number of actor/learner cycles
    
    logger.info("\nConfiguration:")
    logger.info(f"  NATS URL: {NATS_URL}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Batch Size: {BATCH_SIZE}")
    logger.info(f"  Number of Actors: {NUM_ACTORS}")
    logger.info(f"  Training Iterations: {NUM_ITERATIONS}")
    
    # ==================== Setup NATS ====================
    logger.info("\n📡 Setting up NATS...")
    if not await setup_nats(NATS_URL):
        logger.error("❌ Failed to setup NATS. Make sure NATS server is running.")
        logger.error("   Start NATS with: nats-server -D")
        return False
    
    # ==================== Load Models ====================
    logger.info("\n🔧 Loading models...")
    
    model_path = PROJECT_ROOT / "reinvent.prior"
    
    if not model_path.exists():
        logger.error(f"❌ Model not found at {model_path}")
        logger.error("   Please provide a valid model path")
        return False
    
    try:
        # Load agent model (will be updated during training)
        agent_adapter, agent_save_dict, model_type = create_adapter(
            str(model_path),
            mode="training",
            device=DEVICE
        )
        logger.info(f"✅ Agent model loaded (type: {model_type})")
        
        # Load prior model (frozen, used for regularization)
        prior_adapter, _, _ = create_adapter(
            str(model_path),
            mode="inference",
            device=DEVICE
        )
        logger.info("✅ Prior model loaded")
        
        # Freeze prior
        for param in prior_adapter.model.network.parameters():
            param.requires_grad = False
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ==================== Setup Scoring ====================
    logger.info("\n🎯 Setting up scoring function...")
    
    scoring_function = QEDScorer()
    logger.info("✅ QED scoring function ready")
    
    # ==================== Setup Reward Strategy ====================
    logger.info("\n💰 Setting up reward strategy...")
    
    try:
        # Create optimizer for the agent model
        optimizer = torch.optim.Adam(
            agent_adapter.model.network.parameters(),
            lr=0.0001
        )
        logger.info("✅ Optimizer ready")
        
        # Create reward strategy
        reward_strategy = RLReward(
            optimizer=optimizer,
            sigma=120,
            strategy=dap_strategy
        )
        logger.info("✅ Reward strategy (DAP) ready")
        
        # Setup inception (memory for diversity)
        inception = Inception(
            memory_size=100,
            sample_size=10,
            seed_smilies=[],  # No seed SMILES for testing
            scoring_function=scoring_function,
            prior=prior_adapter
        )
        logger.info("✅ Inception memory ready")
        
    except Exception as e:
        logger.error(f"❌ Failed to setup reward: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ==================== Setup Sampler ====================
    logger.info("\n🎲 Setting up sampler...")
    
    sampler_params = {
        "batch_size": BATCH_SIZE,
        "isomeric": False,
        "randomize_smiles": True,
        "unique_sequences": False,
    }
    
    sampler, _ = setup_sampler(model_type, sampler_params, agent_adapter)
    logger.info("✅ Sampler ready")
    
    # ==================== Setup Learner ====================
    logger.info("\n📚 Setting up Learner...")
    
    try:
        # Create diversity filter
        diversity_filter = IdenticalMurckoScaffold(
            bucket_size=25,
            minscore=0.4,
            minsimilarity=0.0,
            penalty_multiplier=0.5,
            rdkit_smiles_flags=dict(allowTautomers=True)
        )
        logger.info("✅ Diversity filter ready")
        
        # Create model state
        model_state = ModelState(
            agent=agent_adapter,
            diversity_filter=diversity_filter
        )
        logger.info("✅ Model state ready")
        
        # Create dummy scorer for learner (we override score() to use trajectory rewards)
        class DummyScorer:
            def __call__(self, smiles_list):
                # This won't be called - learner.score() is overridden
                return np.zeros(len(smiles_list))
        
        dummy_scorer = DummyScorer()
        
        # Create learner
        learner = ImpalaLearner(
            max_steps=1000,
            stage_no=0,
            prior=prior_adapter,
            state=model_state,
            scoring_function=dummy_scorer,
            reward_strategy=reward_strategy,
            sampling_model=sampler,
            smilies=[],
            distance_threshold=0,
            rdkit_smiles_flags=dict(allowTautomers=True),
            inception=inception,
            responder_config=None,
            tb_logdir=None,
            tb_isim=False,
            intrinsic_penalty=None
        )
        logger.info("✅ Learner created")
        
        # Create learner node
        learner_node = LearnerNode(
            queue_url=NATS_URL,
            publish_sibject="jobs.result"
        )
        
        if not await learner_node.Connect():
            logger.error("❌ Learner failed to connect to NATS")
            return False
        
        # Subscribe learner to node
        learner.subscribe(learner_node)
        
        logger.info("✅ Learner connected to NATS")
        
    except Exception as e:
        logger.error(f"❌ Failed to setup learner: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ==================== Setup Actors with Swarm ====================
    logger.info(f"\n🎬 Setting up {NUM_ACTORS} Actor(s) with Swarm...")
    
    try:
        # Create Swarm manager
        swarm = Swarm(
            nats_url=NATS_URL,
            num_actors=NUM_ACTORS,
            model_update_subject="model.update"
        )
        
        await swarm.connect()
        logger.info("✅ Swarm manager connected to NATS")
        
        # Create actors through Swarm
        for actor_id in range(NUM_ACTORS):
            # Create actor config
            actor_config = ActorConfig(
                actor_id=actor_id,
                batch_size=BATCH_SIZE,
                device=DEVICE,
                isomeric=False,
                randomize_smiles=True,
                unique_sequences=False
            )
            
            # Create actor
            actor = ImpalaActor(
                config=actor_config,
                model_path=str(model_path),
                model_version=0,
                scoring_function=scoring_function
            )
            
            # Create actor node and add to swarm
            actor_node = ActorNode(
                actor=actor,
                worker_id=f"actor_{actor_id}",
                queue_url=NATS_URL
            )
            
            if not await actor_node.Connect():
                logger.error(f"❌ Actor {actor_id} failed to connect to NATS")
                return False
            
            swarm.actors.append(actor_node)
            logger.info(f"✅ Actor {actor_id} ready and registered with Swarm")
        
        logger.info(f"✅ Swarm managing {len(swarm.actors)} actors")
        
    except Exception as e:
        logger.error(f"❌ Failed to setup actors with Swarm: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ==================== Run Training Loop ====================
    logger.info("\n🚀 Starting training loop...")
    logger.info("=" * 70)
    
    try:
        # Run actor(s) and learner concurrently
        actor_tasks = [
            asyncio.create_task(run_actor_loop(actor_node, num_iterations=NUM_ITERATIONS))
            for actor_node in swarm.actors
        ]
        
        learner_task = asyncio.create_task(
            run_learner_loop(learner, learner_node, num_iterations=NUM_ITERATIONS)
        )
        
        # Wait for all tasks to complete
        await asyncio.gather(*actor_tasks, learner_task)
        
        logger.info("=" * 70)
        logger.info("✅ Training loop completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("Starting IMPALA Full Integration Test\n")
    
    try:
        success = asyncio.run(main())
        
        if success:
            logger.info("\n" + "=" * 70)
            logger.info("✅ IMPALA FULL INTEGRATION TEST PASSED")
            logger.info("=" * 70)
            sys.exit(0)
        else:
            logger.info("\n" + "=" * 70)
            logger.info("❌ IMPALA FULL INTEGRATION TEST FAILED")
            logger.info("=" * 70)
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
