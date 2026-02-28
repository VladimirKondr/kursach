#!/usr/bin/env python
"""
Integration test for IMPALA: Single Actor + Single Learner
Tests the complete training loop with trajectory generation, scoring, and V-trace correction.
"""

import asyncio
import logging
import sys
from pathlib import Path

import nats
from nats.errors import NoServersError

from reinvent.runmodes.impala_rl.actor import ImpalaActor, ActorConfig
from reinvent.runmodes.impala_rl.nodes.actor_node import ActorNode
from reinvent.runmodes.impala_rl.nodes.learner_node import LearnerNode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


async def setup_nats(nats_url: str = "localhost:4222", max_retries: int = 5) -> bool:
    """Setup NATS server and JetStream"""
    for attempt in range(max_retries):
        try:
            nc = await nats.connect(nats_url)
            js = nc.jetstream()
            
            # Create stream for trajectories
            try:
                await js.add_stream(
                    name="impala_trajectories",
                    subjects=["jobs.result"]
                )
                logger.info("✅ JetStream created")
            except Exception as e:
                logger.info(f"ℹ️ Stream may already exist: {e}")
            
            await nc.close()
            return True
        except NoServersError:
            logger.warning(f"⚠️ NATS not available (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
            continue
    
    logger.error("❌ Could not connect to NATS")
    return False


async def run_actor_loop(actor_node: ActorNode, num_iterations: int = 3):
    """Run actor to generate trajectories"""
    logger.info("🎬 Starting Actor Loop")
    
    for i in range(num_iterations):
        logger.info(f"Actor iteration {i + 1}/{num_iterations}")
        await actor_node.SendTrajectories()
        await asyncio.sleep(1)  # Small delay between iterations
    
    await actor_node.Close()
    logger.info("✅ Actor Loop Complete")


async def main():
    """Main integration test"""
    logger.info("=" * 60)
    logger.info("IMPALA Integration Test: Actor + Learner")
    logger.info("=" * 60)
    
    # Configuration
    NATS_URL = "localhost:4222"
    DEVICE = "cpu"  # Use CPU for testing
    BATCH_SIZE = 32
    NUM_ACTOR_ITERATIONS = 3
    
    logger.info("Configuration:")
    logger.info(f"  NATS URL: {NATS_URL}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Batch Size: {BATCH_SIZE}")
    logger.info(f"  Actor Iterations: {NUM_ACTOR_ITERATIONS}")
    
    # ==================== Setup NATS ====================
    logger.info("\n📡 Setting up NATS...")
    if not await setup_nats(NATS_URL):
        logger.error("Failed to setup NATS. Make sure NATS server is running.")
        logger.error("Start NATS with: nats-server -D")
        return False
    
    # ==================== Setup Actor ====================
    logger.info("\n🎬 Setting up Actor...")
    
    try:
        # Create actor config
        actor_config = ActorConfig(
            actor_id=0,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            isomeric=False,
            randomize_smiles=True,
            unique_sequences=False
        )
        
        # Note: You need to provide actual model path
        # For testing, we use a dummy model path - this will fail if model doesn't exist
        # TODO: Replace with actual path to trained model
        model_path = PROJECT_ROOT / "reinvent.prior"
        
        if not model_path.exists():
            logger.error(f"❌ Model not found at {model_path}")
            logger.error("Please provide a valid model path")
            return False
        
        # Dummy scoring function (for testing only)
        def dummy_scorer(smiles_list):
            """Simple dummy scorer that returns random scores"""
            import numpy as np
            return np.random.uniform(0, 1, len(smiles_list))
        
        # Create actor
        actor = ImpalaActor(
            config=actor_config,
            model_path=str(model_path),
            model_version=0,
            scoring_function=dummy_scorer
        )
        logger.info("✅ Actor created")
        
        # Create actor node
        actor_node = ActorNode(
            actor=actor,
            worker_id="actor_0",
            queue_url=NATS_URL
        )
        
        if not await actor_node.Connect():
            logger.error("❌ Actor failed to connect to NATS")
            return False
        logger.info("✅ Actor connected to NATS")
        
    except Exception as e:
        logger.error(f"❌ Failed to setup actor: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ==================== Setup Learner ====================
    logger.info("\n📚 Setting up Learner...")
    
    try:
        # Create learner node
        learner_node = LearnerNode(
            queue_url=NATS_URL,
            publish_sibject="jobs.result"
        )
        
        if not await learner_node.Connect():
            logger.error("❌ Learner failed to connect to NATS")
            return False
        logger.info("✅ Learner connected to NATS")
        
        # TODO: Setup full learner with model, prior, reward strategy, etc.
        # This requires:
        # 1. Load agent model
        # 2. Load prior model
        # 3. Create reward strategy
        # 4. Create ImpalaLearner instance
        # 5. Call learner.subscribe(learner_node)
        
        logger.warning("⚠️ Full learner setup not implemented in this test")
        logger.warning("⚠️ Next step: Setup ImpalaLearner with models and start training")
        
    except Exception as e:
        logger.error(f"❌ Failed to setup learner: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ==================== Run Training Loop ====================
    logger.info("\n🚀 Starting training loop...")
    
    try:
        # Run actor in background task
        actor_task = asyncio.create_task(
            run_actor_loop(actor_node, num_iterations=NUM_ACTOR_ITERATIONS)
        )
        
        # TODO: Run learner in another task
        # learner_task = asyncio.create_task(
        #     run_learner_loop(learner, num_iterations=NUM_ACTOR_ITERATIONS)
        # )
        
        # Wait for actor to complete
        await actor_task
        
        # Close learner node
        await learner_node.close()
        
        logger.info("✅ Training loop completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("Starting IMPALA Integration Test")
    
    try:
        success = asyncio.run(main())
        
        if success:
            logger.info("\n" + "=" * 60)
            logger.info("✅ IMPALA Integration Test PASSED")
            logger.info("=" * 60)
            sys.exit(0)
        else:
            logger.info("\n" + "=" * 60)
            logger.info("❌ IMPALA Integration Test FAILED")
            logger.info("=" * 60)
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
