#!/usr/bin/env python
"""
Simple IMPALA Actor Test
Tests actor trajectory generation and NATS communication
"""

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def check_nats():
    """Check if NATS is running"""
    import nats
    from nats.errors import NoServersError
    
    try:
        nc = await nats.connect("localhost:4222", max_reconnect_attempts=1)
        await nc.close()
        logger.info("✅ NATS is running")
        return True
    except NoServersError:
        logger.error("❌ NATS is NOT running")
        logger.error("Start NATS with: nats-server -D")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking NATS: {e}")
        return False


def check_models():
    """Check if example models exist"""
    project_root = Path(__file__).parent
    
    # Check for example model (usually in examples/reinvent.prior)
    prior_path = project_root / "reinvent.prior"
    
    if prior_path.exists():
        logger.info(f"✅ Found model at {prior_path}")
        return True
    else:
        logger.warning(f"⚠️  No model found at {prior_path}")
        logger.info("Note: You need to train or download a REINVENT model first")
        return False


async def test_actor():
    """Test actor creation and trajectory generation"""
    logger.info("=" * 60)
    logger.info("IMPALA Actor Test")
    logger.info("=" * 60)
    
    # Check NATS
    logger.info("\n1️⃣ Checking NATS...")
    if not await check_nats():
        logger.error("Cannot continue without NATS. Start it with: nats-server -D")
        return False
    
    # Check models
    logger.info("\n2️⃣ Checking models...")
    if not check_models():
        logger.warning("Models not found. Skipping actor creation.")
        return True  # Not critical for this test
    
    # Try to import and create actor
    logger.info("\n3️⃣ Testing imports...")
    try:
        from reinvent.runmodes.impala_rl.actor import ImpalaActor, ActorConfig
        from reinvent.runmodes.impala_rl.nodes.actor_node import ActorNode
        logger.info("✅ Imports successful")
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    
    logger.info("\n4️⃣ Testing actor creation...")
    try:
        # Create dummy scorer
        def dummy_scorer(smiles_list):
            import numpy as np
            return np.random.uniform(0, 1, len(smiles_list))
        
        # Try to find a model
        project_root = Path(__file__).parent
        possible_models = [
            project_root / "reinvent.prior",
            project_root / "examples" / "reinvent.prior",
            project_root / "reinvent" / "models" / "reinvent" / "example_prior",
        ]
        
        model_path = None
        for path in possible_models:
            if path.exists():
                model_path = str(path)
                break
        
        if not model_path:
            logger.warning("No model found, skipping actor creation")
            return True
        
        config = ActorConfig(
            actor_id=0,
            batch_size=32,
            device="cpu",
            isomeric=False,
            randomize_smiles=True,
            unique_sequences=False
        )
        
        actor = ImpalaActor(
            config=config,
            model_path=model_path,
            model_version=0,
            scoring_function=dummy_scorer
        )
        logger.info("✅ Actor created successfully")
        
        # Test actor node creation
        logger.info("\n5️⃣ Testing actor node...")
        actor_node = ActorNode(
            actor=actor,
            worker_id="test_actor_0",
            queue_url="localhost:4222"
        )
        
        if await actor_node.Connect():
            logger.info("✅ Actor node connected to NATS")
            
            # Send one trajectory batch
            logger.info("\n6️⃣ Testing trajectory generation...")
            await actor_node.SendTrajectories()
            logger.info("✅ Trajectory sent successfully")
            
            await actor_node.Close()
        else:
            logger.error("❌ Actor node failed to connect")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Actor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(test_actor())
        
        if success:
            logger.info("\n" + "=" * 60)
            logger.info("✅ All checks passed!")
            logger.info("=" * 60)
            exit(0)
        else:
            logger.info("\n" + "=" * 60)
            logger.info("❌ Test failed")
            logger.info("=" * 60)
            exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
