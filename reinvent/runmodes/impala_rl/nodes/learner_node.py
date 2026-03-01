import asyncio
import collections
import json
import logging
import random
import time
import io
import torch
import nats

from nats.errors import NoServersError

from reinvent.runmodes.impala_rl.trajectory import Trajectory
from reinvent.runmodes.impala_rl.nodes.model_compression import compress_model_state

# Telemetry imports
from reinvent.runmodes.impala_rl.telemetry import get_tracer, get_logger
from reinvent.runmodes.impala_rl.telemetry.traces import create_span
from reinvent.runmodes.impala_rl.telemetry.propagation import extract_trace_context
from reinvent.runmodes.impala_rl.telemetry.metrics import (
    record_duration_metric,
    record_value_metric,
    increment_counter_metric,
    set_gauge_metric,
)

logger = logging.getLogger(__name__)
telem_logger = get_logger(__name__)
tracer = get_tracer(__name__)

MAX_COMMIT_RETRIES = 3   # max attempts to upload model to NATS KV
COMMIT_RETRY_BASE_DELAY = 1.0   # seconds; doubles on each retry

class LearnerNode:
    def __init__(
        self, 
        queue_url: str = "localhost:4222", 
        publish_sibject: str = "jobs.result",
        stream_name: str = "impala_trajectories",
        model_update_subject: str = "model.update",
        model_bucket: str = "models",
        model_key: str = "current_model",
        buffer_capacity: int = 2000,
        max_staleness: int | None = None,
    ):
        self.queue_url = queue_url
        self.publish_sibject = publish_sibject
        self.stream_name = stream_name
        self.model_update_subject = model_update_subject
        self.model_bucket = model_bucket
        self.model_key = model_key
        self.max_retries = 10
        self.model_version = 0

        # Maximum allowed staleness (learner_version - traj.model_version).
        # Trajectories older than this are evicted before sampling.
        # None = disabled (keep all trajectories regardless of age).
        # Typical value: ~50 learner steps (beyond that IS clipping at rho=2.0
        # makes the correction too weak to be useful).
        self.max_staleness = max_staleness

        self.psub = None
        self.nc = None
        self.js = None

        # Trajectory queue: все поступающие траектории
        # накапливаются здесь; лёрнер **потребляет** их (удаляет
        # после сэмплирования), так что каждая траектория
        # обрабатывается ровно один раз.
        # IS-веса корректируют staleness старых записей.
        # maxlen=buffer_capacity гарантирует, что при переполнении
        # самые старые вытесняются (не должно возникать на практике).
        self._replay_buffer: collections.deque["Trajectory"] = collections.deque(
            maxlen=buffer_capacity
        )
        # Сколько раз подряд GetTrajectories получило 0 новых
        # сообщений из NATS (для диагностики).
        self.consecutive_nats_drains: int = 0
    async def Connect(self):
        with create_span("learner_node.connect", tracer=tracer) as _:
            try:
                self.nc = await nats.connect(
                    self.queue_url,
                    name="learner",
                    reconnect_time_wait=2,
                    max_reconnect_attempts=-1 
                )
                self.js = self.nc.jetstream()
                self.psub = await self.js.pull_subscribe(self.publish_sibject, durable="center_processor", stream=self.stream_name)
                
                telem_logger.info(
                    "Learner node connected to NATS",
                    attributes={"queue_url": self.queue_url}
                )
                set_gauge_metric("nats.connection.state", 1, {"component": "learner"})
                return True
            except NoServersError:
                telem_logger.error("Failed to find NATS server", attributes={"queue_url": self.queue_url})
                set_gauge_metric("nats.connection.state", 0, {"component": "learner"})
                return False
            except Exception as e:
                telem_logger.error(f"Connection error: {str(e)}")
                increment_counter_metric("nats.errors.total", 1, {"component": "learner", "error_type": "connection"})
                set_gauge_metric("nats.connection.state", 0, {"component": "learner"})
                return False

    async def GetTrajectories(
        self,
        timeout: float = 10.0,
        batch_size: int = 16,
        max_retries: int | None = None,
        min_trajectories: int = 1,
    ) -> list[Trajectory]:
        """Drain NATS into the trajectory queue, then consume a random batch.

        All received trajectories are appended to ``self._replay_buffer``.
        The method blocks until the buffer contains at least
        ``min_trajectories`` items, then **removes and returns** a random
        sample of exactly ``min_trajectories`` trajectories.

        Each trajectory is processed exactly once.  IS-weights in the learner
        correct for policy staleness (off-policy IMPALA semantics).

        Returns an empty list when NATS is silent AND the buffer is empty —
        i.e. all work is done.

        Args:
            timeout:          Per-fetch NATS timeout in seconds.
            batch_size:       Number of NATS messages requested per fetch call.
            max_retries:      Consecutive timeouts tolerated while waiting for
                              the buffer to fill (defaults to self.max_retries).
            min_trajectories: Desired batch size returned to the caller.

        Returns:
            Random batch of ``min_trajectories`` Trajectory objects (removed
            from the internal queue), or fewer if the queue is fully drained,
            or [] when nothing is left.
        """
        effective_max_retries = self.max_retries if max_retries is None else max_retries

        with create_span(
            "learner_node.get_trajectories",
            tracer=tracer,
            attributes={
                "batch_size": batch_size,
                "timeout": timeout,
                "min_trajectories": min_trajectories,
            },
        ) as span:
            if not self.psub:
                telem_logger.error("No JetStream connection")
                return []

            total_message_size = 0
            new_from_nats = 0
            start_time = time.perf_counter()

            # ── Phase 1: Eager non-blocking drain ────────────────────────────
            # Always try to pull everything currently sitting in NATS into the
            # replay buffer, regardless of buffer size.  This keeps the buffer
            # fresh and the nats.queue.depth metric accurate.
            try:
                while True:
                    msgs = await self.psub.fetch(batch_size, timeout=0.05)
                    for msg in msgs:
                        headers = msg.header if hasattr(msg, "header") else None
                        _parent_context = extract_trace_context(headers)
                        data_list = json.loads(msg.data.decode())
                        total_message_size += len(msg.data)
                        for traj_dict in data_list:
                            self._replay_buffer.append(Trajectory.from_dict(traj_dict))
                            new_from_nats += 1
                        await msg.ack()
            except nats.errors.TimeoutError:
                pass  # queue is empty — that's fine
            except Exception as e:
                telem_logger.error(f"Eager drain error: {str(e)}")

            # ── Phase 2: Blocking wait (only if buffer still too small) ──────
            consecutive_empty = 0
            while len(self._replay_buffer) < min_trajectories:
                if consecutive_empty >= effective_max_retries:
                    buf_size = len(self._replay_buffer)
                    if buf_size > 0:
                        telem_logger.info(
                            "GetTrajectories: buffer has partial batch, returning",
                            attributes={"buffer_size": buf_size,
                                        "min_trajectories": min_trajectories},
                        )
                    else:
                        telem_logger.error(
                            "GetTrajectories: queue drained and buffer empty",
                            attributes={"max_retries": effective_max_retries},
                        )
                        increment_counter_metric(
                            "learner.get_trajectories.failures.total", 1,
                            {"component": "learner"}
                        )
                    break

                try:
                    span.add_event("waiting_for_buffer",
                                   {"buffer_size": len(self._replay_buffer),
                                    "consecutive_empty": consecutive_empty})
                    msgs = await self.psub.fetch(batch_size, timeout=timeout)
                    consecutive_empty = 0
                    for msg in msgs:
                        headers = msg.header if hasattr(msg, "header") else None
                        _parent_context = extract_trace_context(headers)
                        data_list = json.loads(msg.data.decode())
                        total_message_size += len(msg.data)
                        for traj_dict in data_list:
                            self._replay_buffer.append(Trajectory.from_dict(traj_dict))
                            new_from_nats += 1
                        await msg.ack()
                except nats.errors.TimeoutError:
                    consecutive_empty += 1
                    telem_logger.debug(
                        "Timeout waiting for trajectories",
                        attributes={
                            "consecutive_empty": consecutive_empty,
                            "buffer_size": len(self._replay_buffer),
                        },
                    )
                    increment_counter_metric(
                        "nats.fetch.timeouts.total", 1, {"component": "learner"}
                    )
                except Exception as e:
                    telem_logger.error(f"Fetch error: {str(e)}")
                    increment_counter_metric(
                        "nats.errors.total", 1,
                        {"component": "learner", "error_type": "fetch"}
                    )
                    raise

            # Update drain counter.
            if new_from_nats > 0:
                self.consecutive_nats_drains = 0
            else:
                self.consecutive_nats_drains += 1

            # ── Phase 3: Staleness eviction + Sample + metrics ────────────────
            # Evict trajectories that are too stale for IS correction to be useful.
            stale_evicted = 0
            if self.max_staleness is not None:
                buf_before = len(self._replay_buffer)
                fresh = [
                    t for t in self._replay_buffer
                    if (self.model_version - getattr(t, "model_version", self.model_version))
                    <= self.max_staleness
                ]
                stale_evicted = buf_before - len(fresh)
                if stale_evicted > 0:
                    self._replay_buffer = collections.deque(fresh, maxlen=self._replay_buffer.maxlen)
                    telem_logger.info(
                        "Evicted stale trajectories from replay buffer",
                        attributes={"evicted": stale_evicted, "remaining": len(fresh)},
                    )

            buf_size = len(self._replay_buffer)
            if buf_size == 0:
                return []

            sample_size = min(min_trajectories, buf_size)

            # Destructive random sample: pick indices, remove from buffer.
            buf_list = list(self._replay_buffer)
            indices = sorted(random.sample(range(buf_size), sample_size), reverse=True)
            sample = [buf_list[i] for i in reversed(indices)]
            for i in indices:
                del buf_list[i]
            self._replay_buffer = collections.deque(buf_list, maxlen=self._replay_buffer.maxlen)

            # Get accurate pending count from JetStream consumer info
            nats_pending = 0
            try:
                info = await self.js.consumer_info(self.stream_name, "center_processor")
                nats_pending = info.num_pending
            except Exception:
                pass
            # Accurate combined "pending trajectories" metric:
            #   = trajectories still sitting in NATS (not yet pulled)
            #   + trajectories already in replay buffer (pulled but not trained on)
            # Each NATS message carries batch_size trajectories (approximate).
            nats_pending_trajs = nats_pending * batch_size
            buffer_remaining = len(self._replay_buffer)  # after destructive sample
            pending_trajectories = nats_pending_trajs + buffer_remaining

            total_duration = time.perf_counter() - start_time

            learner_attrs = {"component": "learner"}
            record_duration_metric("learner.batch_collection.duration", total_duration, learner_attrs)
            record_value_metric("nats.queue.depth", nats_pending, learner_attrs)
            record_value_metric("nats.pending_trajectories", nats_pending_trajs, learner_attrs)
            # pending_trajectories: total generated-but-not-consumed across NATS + replay buffer.
            # Use this to check if actors outpace the learner (rising value)
            # or if the learner starves (near zero).
            record_value_metric("pending_trajectories", pending_trajectories, learner_attrs)
            record_value_metric("learner.batch.size", sample_size, learner_attrs)
            record_value_metric("learner.replay_buffer.size", buf_size, learner_attrs)
            record_value_metric("learner.new_trajectories_from_nats", new_from_nats, learner_attrs)
            record_value_metric("nats.message.total_size_bytes", total_message_size, learner_attrs, unit="bytes")
            record_value_metric("learner.stale_evicted", stale_evicted, learner_attrs)
            increment_counter_metric("learner.stale_evicted.total", stale_evicted, learner_attrs)
            increment_counter_metric("learner.trajectories_fetched.total", sample_size, learner_attrs)

            span.set_attribute("num_trajectories", sample_size)
            span.set_attribute("buffer_size", buf_size)
            span.set_attribute("nats_pending", nats_pending)

            telem_logger.info(
                "Trajectories sampled from replay buffer",
                attributes={
                    "sample_size": sample_size,
                    "buffer_size": buf_size,
                    "new_from_nats": new_from_nats,
                    "nats_pending_msgs": nats_pending,
                    "nats_pending_trajs": nats_pending_trajs,
                    "duration_seconds": total_duration,
                },
            )

            return sample
    
    async def close(self):
        if self.nc:
            await self.nc.close()
            print("[Learner] Соединение закрыто")
    
    async def commit_model(self, model):
        """
        Save model to NATS KV store and notify swarm.
        
        Args:
            model: The model to save
        """
        with create_span(
            "learner_node.commit_model",
            tracer=tracer,
            attributes={"version": self.model_version + 1}
        ) as span:
            try:
                # Increment version
                self.model_version += 1
                
                telem_logger.info(
                    "Committing model",
                    attributes={"version": self.model_version}
                )
                
                start_time = time.perf_counter()
                
                # Serialize and compress model state dict (gzip + float16)
                span.add_event("compressing_model")
                serialize_start = time.perf_counter()
                state_dict = model.network.state_dict()
                
                # Get uncompressed size for metrics
                buffer = io.BytesIO()
                torch.save(state_dict, buffer)
                uncompressed_size = len(buffer.getvalue())
                
                # Compress with gzip + float16
                model_bytes = compress_model_state(state_dict, compresslevel=6)
                serialize_duration = time.perf_counter() - serialize_start
                
                model_size_mb = len(model_bytes) / (1024 * 1024)
                uncompressed_size_mb = uncompressed_size / (1024 * 1024)
                compression_ratio = uncompressed_size / len(model_bytes) if len(model_bytes) > 0 else 0
                span.set_attribute("model_size_mb", model_size_mb)
                
                # Upload to NATS KV with retry + exponential back-off
                last_upload_exc = None
                for attempt in range(1, MAX_COMMIT_RETRIES + 1):
                    try:
                        # Re-acquire KV bucket handle on each attempt (handles reconnects)
                        try:
                            kv = await self.js.key_value(self.model_bucket)
                        except Exception:
                            kv = await self.js.create_key_value(
                                bucket=self.model_bucket,
                                history=1,
                                max_bytes=100 * 1024 * 1024
                            )

                        upload_start = time.perf_counter()
                        span.add_event("direct_upload")
                        await asyncio.wait_for(kv.put(self.model_key, model_bytes), timeout=30.0)

                        upload_duration = time.perf_counter() - upload_start
                        last_upload_exc = None
                        break  # success

                    except Exception as upload_exc:
                        last_upload_exc = upload_exc
                        delay = COMMIT_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        telem_logger.warning(
                            f"Model upload attempt {attempt}/{MAX_COMMIT_RETRIES} failed, "
                            f"retrying in {delay:.1f}s",
                            attributes={"version": self.model_version, "error": str(upload_exc)}
                        )
                        if attempt < MAX_COMMIT_RETRIES:
                            await asyncio.sleep(delay)

                if last_upload_exc is not None:
                    raise last_upload_exc
                
                total_duration = time.perf_counter() - start_time
                
                # Record metrics (including compression stats)
                learner_attrs = {"component": "learner"}
                record_duration_metric("learner.model_serialize.duration", serialize_duration, learner_attrs)
                record_duration_metric("learner.model_upload.duration", upload_duration, learner_attrs)
                record_duration_metric("learner.model_commit.duration", total_duration, learner_attrs)
                record_value_metric("model.size_bytes", len(model_bytes), learner_attrs, unit="bytes")
                record_value_metric("model.size_uncompressed_bytes", uncompressed_size, learner_attrs, unit="bytes")
                record_value_metric("model.compression_ratio", compression_ratio, learner_attrs)
                record_value_metric("model.version.current", self.model_version, {"component": "learner"}, snapshot=True)
                increment_counter_metric("learner.model_commits.total", 1, learner_attrs)
                
                span.set_attribute("duration", total_duration)
                span.set_attribute("compression_ratio", compression_ratio)
                span.set_attribute("uncompressed_size_mb", uncompressed_size_mb)
                
                telem_logger.info(
                    "Model committed to NATS KV",
                    attributes={
                        "version": self.model_version,
                        "bucket": self.model_bucket,
                        "key": self.model_key,
                        "size_mb": model_size_mb,
                        "duration_seconds": total_duration,
                    }
                )
                
                # Notify swarm about new model
                await self._notify_swarm_update()
                
            except Exception as e:
                telem_logger.error(f"Failed to commit model: {str(e)}")
                increment_counter_metric("learner.model_commit.failures.total", 1, {"component": "learner"})
                raise
    
    async def _notify_swarm_update(self):
        """Send notification to swarm about new model version."""
        with create_span(
            "learner_node.notify_swarm",
            tracer=tracer,
            attributes={"version": self.model_version}
        ) as _:
            try:
                import json
                
                notification = {
                    "version": self.model_version,
                    "bucket": self.model_bucket,
                    "key": self.model_key
                }
                
                await self.nc.publish(
                    self.model_update_subject,
                    json.dumps(notification).encode()
                )
                
                increment_counter_metric("learner.swarm_notifications.total", 1, {"component": "learner"})
                
                telem_logger.info(
                    "Notified swarm about model update",
                    attributes={"version": self.model_version}
                )
                
            except Exception as e:
                telem_logger.error(f"Failed to notify swarm: {str(e)}")
                increment_counter_metric("learner.swarm_notification.failures.total", 1, {"component": "learner"})
