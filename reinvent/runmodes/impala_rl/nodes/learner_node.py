import asyncio
import json
import logging
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
        model_update_subject: str = "model.update",
        model_bucket: str = "models",
        model_key: str = "current_model"
    ):
        self.queue_url = queue_url
        self.publish_sibject = publish_sibject
        self.model_update_subject = model_update_subject
        self.model_bucket = model_bucket
        self.model_key = model_key
        self.max_retries = 10
        self.model_version = 0

        self.psub = None
        self.nc = None
        self.js = None

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
                self.psub = await self.js.pull_subscribe("jobs.result", durable="center_processor")
                
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
        """Fetch trajectories from NATS queue.

        Keeps pulling messages until at least ``min_trajectories`` trajectories
        are accumulated, then returns the whole collected batch.  If the queue
        goes quiet (``max_retries`` consecutive timeouts with still fewer than
        ``min_trajectories`` collected) the method returns whatever has been
        gathered so far (possibly empty).

        Args:
            timeout:          Timeout in seconds for each individual fetch attempt.
            batch_size:       Number of NATS messages to request per fetch call.
            max_retries:      Max consecutive empty fetches before giving up
                              (defaults to self.max_retries).
            min_trajectories: Keep accumulating until this many trajectories are
                              in hand.  Use 1 to preserve the original one-shot
                              behaviour.

        Returns:
            List of Trajectory objects; empty list means the queue is drained.
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

            accumulated: list[Trajectory] = []
            total_message_size = 0
            consecutive_empty = 0
            start_time = time.perf_counter()

            while len(accumulated) < min_trajectories:
                if consecutive_empty >= effective_max_retries:
                    # Queue appears drained — return what we have (may be empty)
                    queue_depth = self.psub.pending_msgs if self.psub else 0
                    if accumulated:
                        telem_logger.info(
                            "GetTrajectories: returning partial batch before drain",
                            attributes={"num_trajectories": len(accumulated),
                                        "min_trajectories": min_trajectories},
                        )
                    else:
                        telem_logger.error(
                            "GetTrajectories ran out of retries",
                            attributes={"queue_depth": queue_depth,
                                        "max_retries": effective_max_retries},
                        )
                        increment_counter_metric(
                            "learner.get_trajectories.failures.total", 1,
                            {"component": "learner"}
                        )
                    break

                try:
                    span.add_event("fetching_messages",
                                   {"accumulated": len(accumulated),
                                    "consecutive_empty": consecutive_empty})
                    msgs = await self.psub.fetch(batch_size, timeout=timeout)

                    consecutive_empty = 0  # got at least one message — reset

                    for msg in msgs:
                        headers = msg.header if hasattr(msg, "header") else None
                        _parent_context = extract_trace_context(headers)

                        data_list = json.loads(msg.data.decode())
                        total_message_size += len(msg.data)

                        for traj_dict in data_list:
                            accumulated.append(Trajectory.from_dict(traj_dict))

                        await msg.ack()

                except nats.errors.TimeoutError:
                    consecutive_empty += 1
                    queue_depth = self.psub.pending_msgs if self.psub else 0
                    telem_logger.warning(
                        "Timeout fetching trajectories",
                        attributes={
                            "consecutive_empty": consecutive_empty,
                            "accumulated": len(accumulated),
                            "queue_depth": queue_depth,
                        },
                    )
                    increment_counter_metric(
                        "nats.fetch.timeouts.total", 1, {"component": "learner"}
                    )
                    continue
                except Exception as e:
                    telem_logger.error(f"Fetch error: {str(e)}")
                    increment_counter_metric(
                        "nats.errors.total", 1,
                        {"component": "learner", "error_type": "fetch"}
                    )
                    raise

            total_duration = time.perf_counter() - start_time
            queue_depth = self.psub.pending_msgs if self.psub else 0

            learner_attrs = {"component": "learner"}
            record_duration_metric("learner.batch_collection.duration", total_duration, learner_attrs)
            record_value_metric("nats.queue.depth", queue_depth, learner_attrs)
            record_value_metric("learner.batch.size", len(accumulated), learner_attrs)
            record_value_metric("nats.message.total_size_bytes", total_message_size, learner_attrs, unit="bytes")
            increment_counter_metric("learner.trajectories_fetched.total", len(accumulated), learner_attrs)

            span.set_attribute("num_trajectories", len(accumulated))
            span.set_attribute("queue_depth", queue_depth)

            if accumulated:
                telem_logger.info(
                    "Trajectories fetched",
                    attributes={
                        "num_trajectories": len(accumulated),
                        "queue_depth": queue_depth,
                        "duration_seconds": total_duration,
                    },
                )

            return accumulated
    
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
