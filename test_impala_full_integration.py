#!/usr/bin/env python
"""
Full IMPALA Integration Test: Actor(s) + Learner + NATS
Tests complete training loop with trajectory generation, NATS messaging,
scoring, V-trace correction, and model updates.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

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

# Global variable to track NATS process
_nats_process: Optional[subprocess.Popen] = None


def cleanup_old_nats():
    """Kill any existing NATS server processes to free port 4222."""
    try:
        # Find all nats-server processes
        result = subprocess.run(
            ["pgrep", "-f", "nats-server"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            logger.info(f"🧹 Found {len(pids)} existing NATS process(es), cleaning up...")
            
            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", pid], check=False, capture_output=True)
                except Exception:
                    pass
            
            time.sleep(1)
            logger.info("✅ Old NATS processes cleaned up")
        
    except FileNotFoundError:
        # pgrep not available, try alternative
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True
            )
            for line in result.stdout.split('\n'):
                if 'nats-server' in line and 'grep' not in line:
                    parts = line.split()
                    if len(parts) > 1:
                        pid = parts[1]
                        try:
                            subprocess.run(["kill", "-9", pid], check=False, capture_output=True)
                            logger.info(f"🧹 Killed old NATS process (PID: {pid})")
                        except Exception:
                            pass
            time.sleep(1)
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"Error during cleanup: {e}")


def start_nats_server() -> Optional[subprocess.Popen]:
    """Start NATS server with configuration file.
    
    Returns:
        subprocess.Popen object or None if failed
    """
    # Clean up any old NATS processes first
    cleanup_old_nats()
    
    config_path = PROJECT_ROOT / "nats-server.conf"
    
    if not config_path.exists():
        logger.warning(f"⚠️  NATS config not found: {config_path}")
        logger.warning("   Starting NATS without config (default 1MB payload limit)")
        config_args = []
    else:
        logger.info(f"📋 Using NATS config: {config_path}")
        config_args = ["-c", str(config_path)]
    
    try:
        # Start NATS server
        process = subprocess.Popen(
            ["nats-server"] + config_args + ["-js", "-D"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT)
        )
        
        # Wait for NATS to start
        logger.info("⏳ Waiting for NATS to start...")
        time.sleep(2)
        
        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            logger.error(f"❌ NATS failed to start: {stderr.decode()}")
            return None
        
        logger.info(f"✅ NATS server started (PID: {process.pid})")
        return process
        
    except FileNotFoundError:
        logger.error("❌ nats-server not found. Install with:")
        logger.error("   brew install nats-server  # macOS")
        logger.error("   sudo apt install nats-server  # Ubuntu/Debian")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to start NATS: {e}")
        return None


def stop_nats_server(process: Optional[subprocess.Popen]):
    """Stop NATS server gracefully.
    
    Args:
        process: NATS server process to stop
    """
    if process is None:
        return
    
    try:
        logger.info("🛑 Stopping NATS server...")
        
        # Try graceful shutdown first
        process.terminate()
        
        # Wait up to 5 seconds for graceful shutdown
        try:
            process.wait(timeout=5)
            logger.info("✅ NATS server stopped gracefully")
        except subprocess.TimeoutExpired:
            # Force kill if not responding
            logger.warning("⚠️  NATS not responding, forcing shutdown...")
            process.kill()
            process.wait()
            logger.info("✅ NATS server stopped (forced)")
            
    except Exception as e:
        logger.error(f"❌ Error stopping NATS: {e}")


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


async def run_actor_loop(actor_node: ActorNode, num_iterations: int = 5, delay: float = 3.0):
    """Run actor to generate trajectories
    
    Args:
        actor_node: Actor node to run
        num_iterations: Number of iterations
        delay: Delay in seconds between iterations (to slow down actors)
    """
    actor_id = actor_node.actor.config.actor_id
    logger.info(f"🎬 Starting Actor Loop (Actor {actor_id})")
    
    try:
        for i in range(num_iterations):
            current_version = actor_node.actor.model_version
            logger.info(f"📤 [Actor {actor_id}] Iteration {i + 1}/{num_iterations} | Model version: {current_version}")
            
            await actor_node.SendTrajectories()
            
            # Check if model was updated
            new_version = actor_node.actor.model_version
            if new_version > current_version:
                logger.info(f"🔄 [Actor {actor_id}] Model updated: v{current_version} → v{new_version}")
            
            # Slow down to allow learner to catch up
            await asyncio.sleep(delay)
        
        await actor_node.Close()
        logger.info(f"✅ [Actor {actor_id}] Loop Complete | Final model version: {actor_node.actor.model_version}")
        
    except asyncio.CancelledError:
        logger.info(f"⚠️  [Actor {actor_id}] Cancelled, cleaning up...")
        try:
            await actor_node.Close()
        except Exception:
            pass
        raise


# Learner batch size: how many trajectories to collect before each gradient step.
# With BATCH_SIZE=16 per actor and NUM_ACTORS=5 this gives 2-3 actor rounds per
# learner update, keeping gradient estimates stable.
LEARNER_BATCH_SIZE = 32


async def run_learner_loop(learner: ImpalaLearner, learner_node: LearnerNode,
                           learner_batch_size: int = LEARNER_BATCH_SIZE,
                           max_empty_fetches: int = 2,
                           max_nats_drains: int = 3):
    """Run learner loop until actors are done.

    ``GetTrajectories`` drains all incoming NATS messages into a replay buffer
    and returns a random sample of ``learner_batch_size`` trajectories from
    it.  Nothing is discarded — this keeps the IMPALA off-policy semantics
    intact.

    The loop stops when NATS has been silent for ``max_nats_drains``
    consecutive ``GetTrajectories`` calls (i.e. no new trajectories arrived
    from any actor in that time), which signals that all actors have finished.
    """
    logger.info("📚 Starting Learner Loop")

    initial_loss = None
    losses = []
    rewards = []
    step = 0

    try:
        while True:
            # Drain NATS into replay buffer; return a random sample.
            # max_retries controls how long to wait for the buffer to fill
            # on each individual call.
            trajectories = await learner_node.GetTrajectories(
                timeout=3.0,
                min_trajectories=learner_batch_size,
                max_retries=max_empty_fetches,
            )

            if not trajectories:
                logger.warning("  ⚠️  No trajectories received — buffer empty, stopping learner")
                break

            # Stop when NATS has been silent for max_nats_drains calls in a row
            # (all actors have finished publishing).
            if learner_node.consecutive_nats_drains >= max_nats_drains:
                logger.info(
                    f"  ℹ️  NATS queue drained {learner_node.consecutive_nats_drains}×"
                    " in a row — actors done, stopping learner"
                )
                break

            buf_size = len(learner_node._replay_buffer)
            logger.info(f"  🗂️  Replay buffer size: {buf_size}")

            step += 1
            logger.info(
                f"📥 [Learner] Iteration {step} | Current model version: {learner_node.model_version}"
            )
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

            mean_reward = results.total_scores.mean()
            losses.append(float(loss))
            rewards.append(float(mean_reward))

            if initial_loss is None:
                initial_loss = float(loss)

            logger.info(f"  📉 Loss: {loss:.4f}")
            logger.info(f"  📊 Mean reward: {mean_reward:.4f}")
            logger.info(f"  📈 Agent NLLs: {agent_lls.mean():.4f}")

            # Commit model to NATS and notify swarm
            new_version = learner_node.model_version + 1
            logger.info(f"  💾 Committing model version {new_version}")
            await learner_node.commit_model(learner._state.agent.model)
            logger.info(f"  ✅ Model v{new_version} published to actors")

            await asyncio.sleep(0.5)
        
        # Summary statistics
        if losses:
            logger.info("\n📊 Training Summary:")
            logger.info(f"  Initial loss: {initial_loss:.4f}")
            logger.info(f"  Final loss: {losses[-1]:.4f}")
            logger.info(f"  Loss change: {((losses[-1] - initial_loss) / initial_loss * 100):.2f}%")
            logger.info(f"  Mean reward (first): {rewards[0]:.4f}")
            logger.info(f"  Mean reward (last): {rewards[-1]:.4f}")
            logger.info(f"  Reward improvement: {((rewards[-1] - rewards[0]) / max(abs(rewards[0]), 0.001) * 100):.2f}%")
            logger.info(f"  Total model updates: {learner_node.model_version}")

        await learner_node.close()
        logger.info("✅ Learner Loop Complete")

    except asyncio.CancelledError:
        logger.info("⚠️  [Learner] Cancelled, cleaning up...")
        try:
            await learner_node.close()
        except Exception:
            pass
        raise


async def main():
    """Main integration test"""
    global _nats_process
    
    # Generate unique run ID for filtering in SigNoz
    import uuid
    run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    os.environ["IMPALA_RUN_ID"] = run_id
    
    logger.info("=" * 70)
    logger.info("IMPALA Full Integration Test: Actors + Learner + NATS")
    logger.info("=" * 70)
    logger.info(f"🆔 Run ID: {run_id}")
    logger.info("   Use this ID to filter data in SigNoz")
    logger.info("=" * 70)
    
    # ==================== Start NATS Server ====================
    logger.info("\n📡 Starting NATS Server...")
    _nats_process = start_nats_server()
    if _nats_process is None:
        logger.error("❌ Failed to start NATS server")
        return False
    
    try:
        # ==================== Initialize Telemetry ====================
        logger.info("\n📊 Initializing Telemetry...")
        try:
            from reinvent.runmodes.impala_rl.telemetry import setup_telemetry, shutdown_telemetry
            
            # Check if telemetry should be enabled (can disable with env var)
            telemetry_enabled = os.getenv("IMPALA_TELEMETRY_ENABLED", "true").lower() != "false"
            
            if telemetry_enabled:
                # Try to initialize telemetry
                # It will gracefully fallback to console/noop if SigNoz is not available
                config_path = PROJECT_ROOT / "reinvent" / "runmodes" / "impala_rl" / "config" / "telemetry.dev.yaml"
                
                if config_path.exists():
                    success = setup_telemetry(config_path=str(config_path))
                else:
                    success = setup_telemetry(env="dev")
                
                if success:
                    logger.info("✅ Telemetry initialized (SigNoz available)")
                    logger.info("   View traces at: http://localhost:3301")
                else:
                    logger.info("⚠️  Telemetry fallback mode (SigNoz not available)")
                    logger.info("   To enable full telemetry, start SigNoz:")
                    logger.info("   cd examples && docker-compose -f docker-compose.signoz.yml up -d")
            else:
                logger.info("ℹ️  Telemetry disabled via IMPALA_TELEMETRY_ENABLED=false")
        except ImportError:
            logger.warning("⚠️  Telemetry package not available. Install with:")
            logger.warning("   pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")
        except Exception as e:
            logger.warning(f"⚠️  Telemetry initialization failed: {e}")
            logger.warning("   Continuing without telemetry...")
        
        # Configuration
        NATS_URL = "localhost:4222"
        DEVICE = "cpu"  # Use CPU for testing
        BATCH_SIZE = 16  # Small batch for testing
        NUM_ACTORS = 5
        NUM_ITERATIONS = 30  # Number of actor/learner cycles (increased for more training)
        ACTOR_DELAY = 3.0  # Delay between actor iterations in seconds
        
        logger.info("\nConfiguration:")
        logger.info(f"  NATS URL: {NATS_URL}")
        logger.info(f"  Device: {DEVICE}")
        logger.info(f"  Batch Size: {BATCH_SIZE}")
        logger.info(f"  Number of Actors: {NUM_ACTORS}")
        logger.info(f"  Training Iterations: {NUM_ITERATIONS}")
        logger.info(f"  Actor Delay: {ACTOR_DELAY}s (slow down to allow learner to catch up)")
        
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
            # LR=1e-5: 3e-5 still overshoots — the inverted-V in the loss chart
            # means the agent crossed the augmented target (prior + sigma*score)
            # and is now correcting from the other side.  1e-5 is the standard
            # REINVENT4 RL learning rate.
            optimizer = torch.optim.Adam(
                agent_adapter.model.network.parameters(),
                lr=1e-5
            )
            logger.info("✅ Optimizer ready")
            
            # Create reward strategy
            # sigma=50: further reduced from 80.
            # DAP target deviation = sigma * mean_score ≈ 50 * 0.55 = 27.5 nats.
            # At LR=1e-5, sigma=80 still overshoots (inverted-V arc).
            # sigma=50 keeps the reward gradient but halves the step magnitude
            # relative to sigma=80, which should produce a monotone decrease.
            reward_strategy = RLReward(
                optimizer=optimizer,
                sigma=50,
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
                    queue_url=NATS_URL,
                    swarm_manager=swarm
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
                asyncio.create_task(run_actor_loop(actor_node, num_iterations=NUM_ITERATIONS, delay=ACTOR_DELAY))
                for actor_node in swarm.actors
            ]
            
            learner_task = asyncio.create_task(
                run_learner_loop(learner, learner_node)
            )
            
            # Wait for all tasks to complete
            await asyncio.gather(*actor_tasks, learner_task)
            
            logger.info("=" * 70)
            logger.info("✅ Training loop completed successfully")
            
            # Shutdown telemetry to flush remaining data
            try:
                from reinvent.runmodes.impala_rl.telemetry import shutdown_telemetry
                shutdown_telemetry()
            except Exception:
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    finally:
        # Always stop NATS, even on error
        stop_nats_server(_nats_process)
        _nats_process = None


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
        # Ensure NATS is stopped on Ctrl+C
        stop_nats_server(_nats_process)
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        # Ensure NATS is stopped on error
        stop_nats_server(_nats_process)
        sys.exit(1)
