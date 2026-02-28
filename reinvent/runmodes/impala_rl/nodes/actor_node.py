import json
import logging
import time
import nats
from nats.errors import TimeoutError

from reinvent.runmodes.impala_rl.actor import ImpalaActor
from nats.errors import NoServersError

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
    def __init__(self, actor: ImpalaActor, worker_id: str, queue_url: str = "localhost:4222", publish_sibject: str = "jobs.result"):
        self.queue_url = queue_url
        self.actor = actor
        self.publish_sibject = publish_sibject
        self.worker_id = worker_id

        self.nc = None
        self.js = None

    async def Connect(self):
        with create_span("actor_node.connect", tracer=tracer, attributes={"worker_id": self.worker_id}) as span:
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
            
            # Collect trajectories
            span.add_event("collecting_trajectories")
            trajectories = self.actor.collect_trajectory()
            
            # Inject trace context into NATS headers
            headers = inject_trace_context()
            
            # Convert Trajectory objects to dictionaries for JSON serialization
            span.add_event("serializing_trajectories")
            serialize_start = time.perf_counter()
            trajectories_data = [traj.to_dict() for traj in trajectories]
            data_bytes = json.dumps(trajectories_data).encode('utf-8')
            serialize_duration = time.perf_counter() - serialize_start
            
            message_size = len(data_bytes)
            
            try:
                span.add_event("publishing_to_nats")
                publish_start = time.perf_counter()
                ack = await self.js.publish(self.publish_sibject, data_bytes, timeout=5, headers=headers)
                publish_duration = time.perf_counter() - publish_start
                
                total_duration = time.perf_counter() - start_time
                
                # Record metrics
                attributes = {"worker_id": self.worker_id, "subject": self.publish_sibject}
                record_duration_metric("nats.publish.duration", publish_duration, attributes)
                record_duration_metric("nats.serialize.duration", serialize_duration, attributes)
                record_value_metric("nats.message.size_bytes", message_size, attributes, unit="bytes")
                increment_counter_metric("nats.publish.total", 1, attributes)
                
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
                telem_logger.warning("NATS publish timeout", attributes={"worker_id": self.worker_id})
                increment_counter_metric("nats.publish.failures.total", 1, {"worker_id": self.worker_id, "error_type": "timeout"})
            except Exception as e:
                telem_logger.error(
                    f"Publish error: {str(e)}",
                    attributes={"worker_id": self.worker_id, "error": str(e)}
                )
                increment_counter_metric("nats.publish.failures.total", 1, {"worker_id": self.worker_id, "error_type": "other"})

    
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
                import torch
                import io
                
                telem_logger.info(
                    "Loading model from NATS KV",
                    attributes={"worker_id": self.worker_id, "version": version, "bucket": bucket}
                )
                
                load_start = time.perf_counter()
                
                # Get KV bucket
                kv = await self.js.key_value(bucket)
                
                # Try to get chunked model first
                try:
                    span.add_event("downloading_chunked_model")
                    chunks_entry = await kv.get(f"{key}_chunks")
                    num_chunks = int(chunks_entry.value.decode())
                    
                    span.set_attribute("num_chunks", num_chunks)
                    
                    # Download all chunks
                    model_bytes = b""
                    for i in range(num_chunks):
                        chunk_entry = await kv.get(f"{key}_chunk_{i}")
                        model_bytes += chunk_entry.value
                        telem_logger.debug(
                            "Downloaded chunk",
                            attributes={"worker_id": self.worker_id, "chunk": i+1, "total": num_chunks}
                        )
                    
                except Exception:
                    # No chunks, try direct download
                    span.add_event("downloading_direct")
                    entry = await kv.get(key)
                    model_bytes = entry.value
                
                download_duration = time.perf_counter() - load_start
                
                # Load model state dict
                span.add_event("loading_state_dict")
                buffer = io.BytesIO(model_bytes)
                state_dict = torch.load(buffer, map_location=self.actor.device)
                
                # Update actor's adapter model (ImpalaActor uses adapter, not model)
                self.actor.adapter.network.load_state_dict(state_dict)
                self.actor.model_version = version
                
                total_duration = time.perf_counter() - load_start
                model_size_mb = len(model_bytes) / (1024 * 1024)
                
                # Record metrics
                attributes = {"worker_id": self.worker_id}
                record_duration_metric("actor.model_download.duration", download_duration, attributes)
                record_duration_metric("actor.model_load.duration", total_duration, attributes)
                record_value_metric("model.size_bytes", len(model_bytes), attributes, unit="bytes")
                record_value_metric("model.version.current", version, attributes)
                
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
                telem_logger.error(
                    "Failed to load model",
                    attributes={"worker_id": self.worker_id, "version": version, "error": str(e)}
                )
                increment_counter_metric("actor.model_load.failures.total", 1, {"worker_id": self.worker_id})
                raise
