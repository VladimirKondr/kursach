#!/usr/bin/env python
"""
Full IMPALA Integration Test: Actor(s) + Learner + NATS
Tests complete training loop with trajectory generation, NATS messaging,
scoring, V-trace correction, and model updates.
"""

import asyncio
import logging
import os
import socket
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
from reinvent.runmodes.RL.intrinsic_penalty import IdenticalMurckoScaffoldRND
from reinvent.runmodes.RL.reward import RLReward, dap_strategy
from reinvent.runmodes.create_adapter import create_adapter
from reinvent.runmodes.setup_sampler import setup_sampler

# Configuration
NATS_URL = "localhost:4222"
DEVICE = "cpu"  # Use CPU for testing
BATCH_SIZE = 16  # Small batch for testing
NUM_ACTORS = 5
NUM_ITERATIONS = 200  # Number of actor/learner cycles (increased for more training)
ACTOR_DELAY = 2.0  # Delay between actor iterations in seconds

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


def _wait_port_free(port: int = 4222, timeout: float = 10.0, interval: float = 0.3) -> bool:
    """Block until the TCP port is free (connection refused) or timeout expires.

    Returns True if port is free, False if still occupied after timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                pass
            # Connection succeeded → something is still listening
            time.sleep(interval)
        except (ConnectionRefusedError, OSError):
            # Port is free
            return True
    return False


def cleanup_old_nats():
    """Kill any process holding port 4222 (NATS default)."""

    def _kill_pids(pids: list[str], source: str) -> None:
        for pid in pids:
            pid = pid.strip()
            if not pid:
                continue
            killed = False
            for cmd in (["kill", "-9", pid], ["sudo", "-n", "kill", "-9", pid]):
                try:
                    r = subprocess.run(cmd, check=False, capture_output=True)
                    if r.returncode == 0:
                        logger.debug(f"🧹 Killed PID {pid} ({source}) via {cmd[0]}")
                        killed = True
                        break
                except Exception:
                    pass
            if not killed:
                logger.debug(f"⚠️  Could not kill PID {pid} ({source})")

    killed_any = False

    # 1. Kill by port using fuser (Linux) — most reliable
    for sudo_prefix in ([], ["sudo", "-n"]):
        try:
            result = subprocess.run(
                sudo_prefix + ["fuser", "-k", "-9", "4222/tcp"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info(f"🧹 Freed port 4222 via fuser {('(sudo)' if sudo_prefix else '')}")
                killed_any = True
                break
        except FileNotFoundError:
            break

    # 2. Kill by port using lsof (macOS / Linux fallback)
    for sudo_prefix in ([], ["sudo", "-n"]):
        try:
            result = subprocess.run(
                sudo_prefix + ["lsof", "-ti", "tcp:4222"],
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                logger.info(f"🧹 Found {len(pids)} process(es) on port 4222 via lsof, killing...")
                _kill_pids(pids, "lsof:4222")
                killed_any = True
                break
        except FileNotFoundError:
            break

    # 3. Kill all nats-server processes by name as extra safety net
    try:
        result = subprocess.run(
            ["pgrep", "-f", "nats-server"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            logger.info(f"🧹 Found {len(pids)} nats-server process(es) by name, killing...")
            _kill_pids(pids, "pgrep")
            killed_any = True
    except FileNotFoundError:
        pass

    # 4. Poll until the port is actually free (handles TIME_WAIT / slow OS cleanup)
    if _wait_port_free(4222, timeout=10.0):
        logger.info("✅ Port 4222 is free")
    else:
        logger.warning("⚠️  Port 4222 still occupied after 10 s — proceeding anyway")


def _nats_is_alive(host: str = "127.0.0.1", port: int = 4222, timeout: float = 2.0) -> bool:
    """Return True if something is already accepting TCP connections on port 4222."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def start_nats_server() -> Optional[subprocess.Popen]:
    """Start NATS server, or reuse it if already running on port 4222.

    Returns:
        subprocess.Popen object, or None if failed.
        If an existing server is reused the return value is a sentinel
        Popen-like object that does nothing on terminate/kill.
    """
    # ── If NATS is already up (e.g. started by another user / system service)
    # ── just use it; don't try to kill + restart.
    if _nats_is_alive():
        logger.info("✅ NATS already running on port 4222 — reusing it")

        class _FakeProcess:
            """Sentinel so stop_nats_server() is a no-op."""
            pid = None
            def poll(self): return None
            def terminate(self): pass
            def kill(self): pass
            def wait(self, timeout=None): pass

        return _FakeProcess()

    # ── Port is free — clean up stale processes and start fresh ──────────────
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
        process = subprocess.Popen(
            ["nats-server"] + config_args + ["-js", "-D"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT)
        )

        # Wait until NATS actually accepts connections (up to 10 s)
        logger.info("⏳ Waiting for NATS to start...")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                # Process already exited — collect stderr and report
                _, stderr = process.communicate()
                logger.error(f"❌ NATS failed to start: {stderr.decode()}")
                return None
            if _nats_is_alive():
                logger.info(f"✅ NATS server started (PID: {process.pid})")
                return process
            time.sleep(0.2)

        # Timed out waiting
        _, stderr = process.communicate(timeout=1)
        logger.error(f"❌ NATS did not become ready in time: {stderr.decode()}")
        process.kill()
        return None

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
        process: NATS server process to stop (None or _FakeProcess → no-op)
    """
    if process is None:
        return

    # _FakeProcess sentinel: we reused an external NATS — don't stop it
    if getattr(process, "pid", None) is None:
        logger.info("ℹ️  NATS server was external — not stopping it")
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


async def setup_nats(nats_url: str = "localhost:4222", max_retries: int = 5,
                     stream_name: str = "impala_trajectories",
                     jobs_subject: str = "jobs.result") -> bool:
    """Setup NATS server and JetStream"""
    for attempt in range(max_retries):
        try:
            nc = await nats.connect(nats_url)
            js = nc.jetstream()
            
            # Delete existing stream if it exists (for clean test)
            try:
                await js.delete_stream(stream_name)
                logger.info("🗑️ Deleted existing stream")
            except Exception:
                pass  # Stream doesn't exist, that's OK
            
            # Create stream for trajectories
            try:
                await js.add_stream(
                    name=stream_name,
                    subjects=[jobs_subject],
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
LEARNER_BATCH_SIZE = 64


async def run_learner_loop(learner: ImpalaLearner, learner_node: LearnerNode,
                           actors_done: asyncio.Event,
                           learner_batch_size: int = LEARNER_BATCH_SIZE,
                           max_empty_fetches: int = 2):
    """Run learner loop until actors are done.

    ``GetTrajectories`` drains all incoming NATS messages into a replay buffer
    and returns a random sample of ``learner_batch_size`` trajectories from it.
    Nothing is discarded — this keeps the IMPALA off-policy semantics intact.

    The loop runs as long as actors are alive (``actors_done`` not set).  Once
    all actor tasks complete they set ``actors_done``; the learner then does a
    final drain of whatever is left in NATS / the replay buffer and exits.
    """
    logger.info("📚 Starting Learner Loop")

    initial_loss = None
    losses = []
    rewards = []   # stale: mean score of batch trajectories
    step = 0

    try:
        while True:
            actors_finished = actors_done.is_set()

            # After actors finish use a shorter timeout so we don't hang;
            # allow only 1 consecutive empty fetch before stopping.
            fetch_timeout = 1.5 if actors_finished else 3.0
            fetch_retries = 1  if actors_finished else max_empty_fetches

            trajectories = await learner_node.GetTrajectories(
                timeout=fetch_timeout,
                min_trajectories=learner_batch_size,
                max_retries=fetch_retries,
            )

            if not trajectories:
                if actors_finished:
                    logger.info("  ✅ Queue fully consumed after actors finished — stopping learner")
                else:
                    logger.warning("  ⚠️  No trajectories in queue yet — waiting...")
                    await asyncio.sleep(0.5)
                    continue
                break

            # Log buffer remaining AFTER destructive sample (shows headroom for next step)
            buf_remaining = len(learner_node._replay_buffer)
            logger.info(f"  🗂️  Replay buffer remaining: {buf_remaining}")

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

            mean_reward = results.total_scores.mean()  # stale: from buffer
            losses.append(float(loss))
            rewards.append(float(mean_reward))

            if initial_loss is None:
                initial_loss = float(loss)

            logger.info(f"  📉 Loss: {loss:.4f}")
            logger.info(f"  📊 Stale reward (batch): {mean_reward:.4f}")
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
            logger.info(f"  Stale reward (first batch): {rewards[0]:.4f}")
            logger.info(f"  Stale reward (last batch):  {rewards[-1]:.4f}")
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


async def main(args):
    """Main integration test"""
    global _nats_process

    # Derive per-experiment NATS config from args so parallel runs don't mix
    nats_url = NATS_URL
    jobs_subject = f"jobs.{args.exp_name}.result"
    model_subject = f"model.{args.exp_name}.update"
    stream_name = f"impala_{args.exp_name}"

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
        
        logger.info("\nConfiguration:")
        logger.info(f"  Experiment: {args.exp_name}")
        logger.info(f"  NATS URL: {nats_url}  subjects: {jobs_subject} / {model_subject}")
        logger.info(f"  Device: {DEVICE}")
        logger.info(f"  Batch Size: {BATCH_SIZE}")
        logger.info(f"  Number of Actors: {NUM_ACTORS}")
        logger.info(f"  Training Iterations: {args.num_iterations}")
        logger.info(f"  Actor Delay: {args.actor_delay}s")
        logger.info(f"  sigma={args.sigma}  lr={args.lr}  ema_alpha={args.ema_alpha}")
        
        # ==================== Setup NATS ====================
        logger.info("\n📡 Setting up NATS...")
        if not await setup_nats(nats_url, stream_name=stream_name, jobs_subject=jobs_subject):
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
                lr=args.lr
            )
            logger.info("✅ Optimizer ready")
            
            # Create reward strategy
            reward_strategy = RLReward(
                optimizer=optimizer,
                sigma=args.sigma,
                strategy=dap_strategy
            )
            logger.info("✅ Reward strategy (DAP) ready")
            
            # Setup inception (memory for diversity)
            inception = Inception(
                memory_size=100,
                sample_size=16,  # ~25% of LEARNER_BATCH_SIZE=64
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
            # RND: scaffold penalty + intrinsic novelty bonus via Random Network Distillation.
            # Keeps two copies of the prior (frozen target + trainable prediction);
            # prediction error = novelty signal added to the extrinsic score.
            rnd_penalty = IdenticalMurckoScaffoldRND(
                penalty_function="Sigmoid",
                bucket_size=25,
                minscore=0.4,
                device=torch.device(DEVICE),
                prior_model_file_path=str(model_path),
                learning_rate=1e-4,
                rdkit_smiles_flags=dict(allowTautomers=True),
            )
            logger.info("✅ RND intrinsic penalty ready")

            # diversity_filter goes into ModelState; we use intrinsic_penalty instead
            model_state = ModelState(
                agent=agent_adapter,
                diversity_filter=None,
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
                intrinsic_penalty=rnd_penalty,
            )
            learner._score_ema_alpha = args.ema_alpha
            logger.info("✅ Learner created")
            
            # Create learner node
            learner_node = LearnerNode(
                queue_url=nats_url,
                publish_sibject=jobs_subject,
                stream_name=stream_name,
                model_update_subject=model_subject,           # must match what Swarm subscribes to
                model_bucket=f"models_{args.exp_name}",       # namespace per experiment to avoid cross-contamination
                # Evict trajectories older than 50 learner updates.
                # IS clipping at rho=2.0 covers ~log(2)=0.69 nats of policy
                # drift; beyond ~50 steps the correction becomes negligible.
                # Raise this value in production (more iterations = more budget).
                max_staleness=50,
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
                nats_url=nats_url,
                num_actors=NUM_ACTORS,
                model_update_subject=model_subject
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
                    queue_url=nats_url,
                    publish_sibject=jobs_subject,
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
            # Run actor(s) and learner concurrently.
            # actors_done is set once every actor finishes; the learner uses it
            # to know when to stop waiting for new trajectories.
            actors_done = asyncio.Event()

            actor_tasks = [
                asyncio.create_task(run_actor_loop(actor_node, num_iterations=args.num_iterations, delay=args.actor_delay))
                for actor_node in swarm.actors
            ]

            async def _wait_actors_then_signal():
                await asyncio.gather(*actor_tasks)
                logger.info("🏁 All actors finished — signalling learner to drain and stop")
                actors_done.set()

            learner_task = asyncio.create_task(
                run_learner_loop(learner, learner_node, actors_done=actors_done)
            )

            # Wait for actors first, then wait for learner to drain
            await asyncio.gather(
                asyncio.create_task(_wait_actors_then_signal()),
                learner_task,
            )
            
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
    import argparse
    parser = argparse.ArgumentParser(description="IMPALA integration test")
    parser.add_argument("--exp-name", default="default",
                        help="Experiment name — namespaces NATS subjects so parallel runs don't mix (default: default)")
    parser.add_argument("--sigma", type=float, default=50.0,
                        help="DAP sigma: larger = stronger gradient (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Adam learning rate (default: 1e-5)")
    parser.add_argument("--ema-alpha", type=float, default=0.05,
                        help="Score EMA alpha; smaller = slower baseline (default: 0.05)")
    parser.add_argument("--num-iterations", type=int, default=NUM_ITERATIONS,
                        help=f"Actor iterations per run (default: {NUM_ITERATIONS})")
    parser.add_argument("--actor-delay", type=float, default=ACTOR_DELAY,
                        help=f"Delay between actor steps in seconds (default: {ACTOR_DELAY})")
    _args = parser.parse_args()

    logger.info("Starting IMPALA Full Integration Test\n")
    
    try:
        success = asyncio.run(main(_args))
        
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
