import json
import logging
import time
import nats

from nats.errors import NoServersError

from reinvent.runmodes.impala_rl.trajectory import Trajectory

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
        with create_span("learner_node.connect", tracer=tracer) as span:
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

    async def GetTrajectories(self, timeout: float = 10.0, batch_size: int = 16) -> list[Trajectory]:
        """Fetch trajectories from NATS queue
        
        Args:
            timeout: Timeout in seconds for fetching
            batch_size: Number of trajectory messages to fetch
            
        Returns:
            List of Trajectory objects
        """
        with create_span(
            "learner_node.get_trajectories",
            tracer=tracer,
            attributes={"batch_size": batch_size, "timeout": timeout}
        ) as span:
            if not self.psub:
                telem_logger.error("No JetStream connection")
                return []

            retry: int = 0
            start_time = time.perf_counter()
            
            while retry < self.max_retries:
                retry += 1
                try:
                    span.add_event("fetching_messages", {"retry": retry})
                    fetch_start = time.perf_counter()
                    msgs = await self.psub.fetch(batch_size, timeout=timeout)
                    fetch_duration = time.perf_counter() - fetch_start
                    
                    trajectories: list[Trajectory] = []
                    total_message_size = 0

                    for msg in msgs:
                        # Extract trace context from message headers (not currently used but prepared for future linked spans)
                        headers = msg.header if hasattr(msg, 'header') else None
                        _parent_context = extract_trace_context(headers)
                        
                        # Deserialize list of trajectory dicts
                        data_list = json.loads(msg.data.decode())
                        total_message_size += len(msg.data)
                        
                        # Convert each dict to Trajectory object
                        for traj_dict in data_list:
                            traj = Trajectory.from_dict(traj_dict)
                            trajectories.append(traj)
                        
                        await msg.ack()
                    
                    total_duration = time.perf_counter() - start_time
                    
                    # Check queue depth
                    queue_depth = self.psub.pending_msgs
                    
                    # Record metrics
                    record_duration_metric("learner.batch_collection.duration", total_duration)
                    record_duration_metric("nats.fetch.duration", fetch_duration)
                    record_value_metric("nats.queue.depth", queue_depth)
                    record_value_metric("learner.batch.size", len(trajectories))
                    record_value_metric("nats.message.total_size_bytes", total_message_size, unit="bytes")
                    increment_counter_metric("learner.trajectories_fetched.total", len(trajectories))
                    
                    span.set_attribute("num_messages", len(msgs))
                    span.set_attribute("num_trajectories", len(trajectories))
                    span.set_attribute("queue_depth", queue_depth)
                    span.set_attribute("retries", retry)
                    
                    telem_logger.info(
                        "Trajectories fetched",
                        attributes={
                            "num_trajectories": len(trajectories),
                            "num_messages": len(msgs),
                            "queue_depth": queue_depth,
                            "duration_seconds": total_duration,
                        }
                    )
                    
                    return trajectories
                    
                except nats.errors.TimeoutError:
                    queue_depth = self.psub.pending_msgs if self.psub else 0
                    telem_logger.warning(
                        "Timeout fetching trajectories",
                        attributes={"retry": retry, "queue_depth": queue_depth}
                    )
                    increment_counter_metric("nats.fetch.timeouts.total", 1, {"component": "learner"})
                    continue
                except Exception as e:
                    telem_logger.error(f"Fetch error: {str(e)}")
                    increment_counter_metric("nats.errors.total", 1, {"component": "learner", "error_type": "fetch"})
                    raise e
                    
            queue_depth = self.psub.pending_msgs if self.psub else 0
            telem_logger.error(
                "GetTrajectories ran out of retries",
                attributes={"queue_depth": queue_depth, "max_retries": self.max_retries}
            )
            increment_counter_metric("learner.get_trajectories.failures.total", 1)
            return []
    
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
                import torch
                import io
                
                # Increment version
                self.model_version += 1
                
                telem_logger.info(
                    "Committing model",
                    attributes={"version": self.model_version}
                )
                
                start_time = time.perf_counter()
                
                # Create or get KV bucket
                try:
                    kv = await self.js.key_value(self.model_bucket)
                except Exception:
                    # Create bucket if it doesn't exist
                    kv = await self.js.create_key_value(
                        bucket=self.model_bucket,
                        history=1,  # Keep only latest version
                        max_bytes=100 * 1024 * 1024  # 100MB limit
                    )
                
                # Serialize model state dict
                span.add_event("serializing_model")
                serialize_start = time.perf_counter()
                buffer = io.BytesIO()
                torch.save(model.network.state_dict(), buffer)
                model_bytes = buffer.getvalue()
                serialize_duration = time.perf_counter() - serialize_start
                
                model_size_mb = len(model_bytes) / (1024*1024)
                span.set_attribute("model_size_mb", model_size_mb)
                
                # For large models (>10MB), use chunked upload
                if model_size_mb > 10:
                    span.add_event("chunked_upload")
                    upload_start = time.perf_counter()
                    
                    chunk_size = 5 * 1024 * 1024  # 5MB chunks
                    num_chunks = (len(model_bytes) + chunk_size - 1) // chunk_size
                    
                    span.set_attribute("num_chunks", num_chunks)
                    
                    # Save number of chunks first
                    await kv.put(f"{self.model_key}_chunks", str(num_chunks).encode())
                    
                    # Upload chunks
                    for i in range(num_chunks):
                        start = i * chunk_size
                        end = min(start + chunk_size, len(model_bytes))
                        chunk = model_bytes[start:end]
                        await kv.put(f"{self.model_key}_chunk_{i}", chunk)
                        telem_logger.debug(
                            "Uploaded chunk",
                            attributes={"chunk": i+1, "total": num_chunks}
                        )
                    
                    upload_duration = time.perf_counter() - upload_start
                else:
                    # Small model - direct upload
                    span.add_event("direct_upload")
                    upload_start = time.perf_counter()
                    await kv.put(self.model_key, model_bytes)
                    upload_duration = time.perf_counter() - upload_start
                
                total_duration = time.perf_counter() - start_time
                
                # Record metrics
                record_duration_metric("learner.model_serialize.duration", serialize_duration)
                record_duration_metric("learner.model_upload.duration", upload_duration)
                record_duration_metric("learner.model_commit.duration", total_duration)
                record_value_metric("model.size_bytes", len(model_bytes), unit="bytes")
                record_value_metric("model.version.current", self.model_version)
                increment_counter_metric("learner.model_commits.total", 1)
                
                span.set_attribute("duration", total_duration)
                
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
                increment_counter_metric("learner.model_commit.failures.total", 1)
                raise
    
    async def _notify_swarm_update(self):
        """Send notification to swarm about new model version."""
        with create_span(
            "learner_node.notify_swarm",
            tracer=tracer,
            attributes={"version": self.model_version}
        ) as span:
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
                
                increment_counter_metric("learner.swarm_notifications.total", 1)
                
                telem_logger.info(
                    "Notified swarm about model update",
                    attributes={"version": self.model_version}
                )
                
            except Exception as e:
                telem_logger.error(f"Failed to notify swarm: {str(e)}")
                increment_counter_metric("learner.swarm_notification.failures.total", 1)
