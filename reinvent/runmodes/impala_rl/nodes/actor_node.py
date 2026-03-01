import asyncio
import json
import logging
import time
import nats
from nats.errors import TimeoutError

from reinvent.runmodes.impala_rl.actor import ImpalaActor
from nats.errors import NoServersError
from reinvent.runmodes.impala_rl.nodes.model_compression import decompress_model_state

# Telemetry imports
from reinvent.runmodes.impala_rl.telemetry import get_tracer, get_logger
from reinvent.runmodes.impala_rl.telemetry.traces import create_span
from reinvent.runmodes.impala_rl.telemetry.propagation import inject_trace_context
from reinvent.runmodes.impala_rl.telemetry.metrics import (
    record_duration_metric,
    record_value_metric,
    increment_counter_metric,
    set_gauge_metric,
)

logger = logging.getLogger(__name__)
telem_logger = get_logger(__name__)
tracer = get_tracer(__name__)

class ActorNode:
    def __init__(self, actor: ImpalaActor, worker_id: str, queue_url: str = "localhost:4222", publish_sibject: str = "jobs.result", swarm_manager=None):
        self.queue_url = queue_url
        self.actor = actor
        self.publish_sibject = publish_sibject
        self.worker_id = worker_id

        self.nc = None
        self.js = None
        
        # Reference to Swarm for latest_model_version tracking
        self._swarm_manager = swarm_manager
        self._swarm_latest_version = 0

        # Pull-based model update: swarm writes latest info here, actor pulls after each trajectory batch.
        self._latest_model_info: dict | None = None  # {version, bucket, key}

        # Cached KV bucket handle — avoids a StreamInfo roundtrip on every broadcast.
        # Invalidated on reconnect (set back to None).
        self._kv_cache: dict = {}  # bucket_name -> kv handle

    async def Connect(self):
        with create_span("actor_node.connect", tracer=tracer, attributes={"worker_id": self.worker_id}) as _:
            try:
                self.nc = await nats.connect(
                    self.queue_url,
                    name=self.worker_id,
                    reconnect_time_wait=2,
                    max_reconnect_attempts=-1 
                )
                self.js = self.nc.jetstream()
                
                telem_logger.info(
                    "Actor node connected to NATS",
                    attributes={"worker_id": self.worker_id, "queue_url": self.queue_url}
                )
                set_gauge_metric("nats.connection.state", 1, {"worker_id": self.worker_id, "component": "actor"})
                return True
            except NoServersError:
                telem_logger.error(
                    "Failed to find NATS server",
                    attributes={"worker_id": self.worker_id, "queue_url": self.queue_url}
                )
                set_gauge_metric("nats.connection.state", 0, {"worker_id": self.worker_id, "component": "actor"})
                return False
            except Exception as e:
                telem_logger.error(
                    "Connection error",
                    attributes={"worker_id": self.worker_id, "error": str(e)}
                )
                increment_counter_metric("nats.errors.total", 1, {"worker_id": self.worker_id, "error_type": "connection"})
                set_gauge_metric("nats.connection.state", 0, {"worker_id": self.worker_id, "component": "actor"})
                return False

    async def SendTrajectories(self):
        with create_span(
            "actor_node.send_trajectories",
            tracer=tracer,
            attributes={
                "worker_id": self.worker_id,
                "subject": self.publish_sibject,
            }
        ) as span:
            if not self.js:
                telem_logger.error("No JetStream connection")
                return
            
            start_time = time.perf_counter()
            
            # Collect trajectories — run in executor so the event loop is free
            # while the LSTM sampling (CPU-intensive) is happening. This allows
            # concurrent load_model_from_nats tasks to proceed without timing out.
            span.add_event("collecting_trajectories")
            loop = asyncio.get_event_loop()
            trajectories = await loop.run_in_executor(None, self.actor.collect_trajectory)
            
            # Inject trace context into NATS headers
            headers = inject_trace_context()
            
            # Convert Trajectory objects to dictionaries for JSON serialization
            span.add_event("serializing_trajectories")
            serialize_start = time.perf_counter()
            trajectories_data = [traj.to_dict() for traj in trajectories]
            data_bytes = json.dumps(trajectories_data).encode('utf-8')
            serialize_duration = time.perf_counter() - serialize_start
            
            message_size = len(data_bytes)
            
            # Retry publish up to 3 times on timeout to avoid silently dropping batches
            _max_publish_retries = 3
            _publish_attempt = 0
            _published = False
            while _publish_attempt < _max_publish_retries and not _published:
                _publish_attempt += 1
                try:
                    span.add_event("publishing_to_nats", {"attempt": _publish_attempt})
                    publish_start = time.perf_counter()
                    ack = await self.js.publish(self.publish_sibject, data_bytes, timeout=5, headers=headers)
                    publish_duration = time.perf_counter() - publish_start
                    _published = True

                    total_duration = time.perf_counter() - start_time

                    # Record metrics
                    attributes = {"worker_id": self.worker_id, "subject": self.publish_sibject}
                    record_duration_metric("nats.publish.duration", publish_duration, attributes)
                    record_duration_metric("nats.serialize.duration", serialize_duration, attributes)
                    record_value_metric("nats.message.size_bytes", message_size, attributes, unit="bytes")
                    increment_counter_metric("nats.publish.total", 1, attributes)

                    # Record model version lag at the time of sending trajectories
                    # This is the meaningful staleness metric: how old is the actor's model
                    # compared to the latest version published by the learner
                    if hasattr(self, '_swarm_latest_version'):
                        version_lag = self._swarm_latest_version - self.actor.model_version
                        record_value_metric("model.version.lag", version_lag, {"worker_id": self.worker_id}, snapshot=True)
                        span.set_attribute("model_version_lag", version_lag)

                    span.set_attribute("stream_sequence", ack.seq)
                    span.set_attribute("message_size", message_size)
                    span.set_attribute("num_trajectories", len(trajectories))
                    span.set_attribute("total_duration", total_duration)

                    telem_logger.debug(
                        "Trajectories published",
                        attributes={
                            "worker_id": self.worker_id,
                            "stream_seq": ack.seq,
                            "num_trajectories": len(trajectories),
                            "message_size_kb": message_size / 1024,
                        }
                    )
                except TimeoutError:
                    telem_logger.warning(
                        "NATS publish timeout",
                        attributes={"worker_id": self.worker_id, "attempt": _publish_attempt}
                    )
                    increment_counter_metric("nats.publish.failures.total", 1, {"worker_id": self.worker_id, "error_type": "timeout"})
                    if _publish_attempt < _max_publish_retries:
                        await asyncio.sleep(0.5 * _publish_attempt)  # brief back-off
                except Exception as e:
                    telem_logger.error(
                        f"Publish error: {str(e)}",
                        attributes={"worker_id": self.worker_id, "error": str(e)}
                    )
                    increment_counter_metric("nats.publish.failures.total", 1, {"worker_id": self.worker_id, "error_type": "other"})
                    break  # non-timeout errors are not retriable

        # --- Pull-based model update ---
        # After every trajectory batch, check if a newer model version is available
        # and pull it.  This runs while the lock is guaranteed to be free (sampling
        # just finished), so there is zero lock contention.
        await self._maybe_pull_model()

    async def _maybe_pull_model(self):
        """Pull the latest model from NATS KV if a newer version is available."""
        info = self._latest_model_info
        if info is None:
            return
        if info["version"] <= self.actor.model_version:
            return

        logger.debug(
            f"[ActorNode {self.worker_id}] Pulling model v{info['version']} "
            f"(current: v{self.actor.model_version})"
        )
        await self.load_model_from_nats(info["version"], info["bucket"], info["key"])

    async def Close(self):
        if self.nc:
            await self.nc.close()
            print(f"[{self.worker_id}] Соединение закрыто")
    
    async def load_model_from_nats(self, version: int, bucket: str, key: str):
        """
        Load new model version from NATS KV store.

        Args:
            version: Model version number
            bucket: KV bucket name
            key: Key in the bucket
        """
        with create_span(
            "actor_node.load_model_from_nats",
            tracer=tracer,
            attributes={
                "worker_id": self.worker_id,
                "version": version,
                "bucket": bucket,
            }
        ) as span:
            try:
                telem_logger.info(
                    "Loading model from NATS KV",
                    attributes={"worker_id": self.worker_id, "version": version, "bucket": bucket}
                )
                
                load_start = time.perf_counter()

                # Always re-derive the JetStream context from the live connection so
                # that a transparent NATS reconnect doesn't leave self.js stale.
                js = self.nc.jetstream()

                # Get KV bucket — use cached handle, re-open only on error / first use.
                KV_MAX_RETRIES = 3
                KV_RETRY_DELAY = 1.0
                kv = self._kv_cache.get(bucket)
                if kv is None:
                    for _attempt in range(1, KV_MAX_RETRIES + 1):
                        try:
                            kv = await asyncio.wait_for(js.key_value(bucket), timeout=10.0)
                            self._kv_cache[bucket] = kv
                            break
                        except (nats.errors.ConnectionClosedError, asyncio.CancelledError):
                            return
                        except Exception as _e:
                            if _attempt == KV_MAX_RETRIES:
                                raise
                            await asyncio.sleep(KV_RETRY_DELAY * _attempt)

                # Direct download with retry
                span.add_event("downloading_direct")
                for _attempt in range(1, KV_MAX_RETRIES + 1):
                    try:
                        entry = await asyncio.wait_for(kv.get(key), timeout=30.0)
                        model_bytes = entry.value
                        break
                    except (nats.errors.ConnectionClosedError, asyncio.CancelledError):
                        return
                    except Exception as _e:
                        if _attempt == KV_MAX_RETRIES:
                            raise
                        # KV handle may be stale — evict cache and re-open
                        self._kv_cache.pop(bucket, None)
                        try:
                            kv = await asyncio.wait_for(js.key_value(bucket), timeout=10.0)
                            self._kv_cache[bucket] = kv
                        except Exception:
                            pass
                        await asyncio.sleep(KV_RETRY_DELAY * _attempt)

                download_duration = time.perf_counter() - load_start

                # Decompress in thread pool — gzip.decompress + torch.load + float16→float32
                # are all CPU-bound and would block the event loop for 2-4s otherwise.
                span.add_event("decompressing_model")
                decompress_start = time.perf_counter()

                loop = asyncio.get_event_loop()
                state_dict = await loop.run_in_executor(
                    None,
                    lambda: decompress_model_state(model_bytes, device=self.actor.device)
                )

                decompress_duration = time.perf_counter() - decompress_start

                # Delegate to ImpalaActor.update_model() — run in executor so that
                # acquiring the model lock doesn't block the event loop.
                await loop.run_in_executor(None, self.actor.update_model, state_dict, version)
                
                total_duration = time.perf_counter() - load_start
                model_size_mb = len(model_bytes) / (1024 * 1024)
                
                # Record metrics (including decompression time)
                attributes = {"worker_id": self.worker_id}
                record_duration_metric("actor.model_download.duration", download_duration, attributes)
                record_duration_metric("actor.model_decompress.duration", decompress_duration, attributes)
                record_duration_metric("actor.model_load.duration", total_duration, attributes)
                record_value_metric("model.size_bytes", len(model_bytes), attributes, unit="bytes")
                record_value_metric("model.version.current", version, {"component": "actor", "worker_id": self.worker_id}, snapshot=True)
                
                span.set_attribute("model_size_mb", model_size_mb)
                span.set_attribute("duration", total_duration)
                
                telem_logger.info(
                    "Model loaded successfully",
                    attributes={
                        "worker_id": self.worker_id,
                        "version": version,
                        "size_mb": model_size_mb,
                        "duration_seconds": total_duration,
                    }
                )
                
            except Exception as e:
                # Log the real error to the console so it appears in plain stdout,
                # not just hidden inside telemetry span attributes.
                logger.error(
                    f"[{self.worker_id}] Failed to load model v{version}: {type(e).__name__}: {e}",
                    exc_info=True
                )
                telem_logger.error(
                    "Failed to load model",
                    attributes={"worker_id": self.worker_id, "version": version, "error": str(e)}
                )
                increment_counter_metric("actor.model_load.failures.total", 1, {"worker_id": self.worker_id})
                raise
