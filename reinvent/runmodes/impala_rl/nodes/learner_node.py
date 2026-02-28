import json
import logging
import torch
import io
import nats
from nats.errors import TimeoutError

from nats.errors import NoServersError

from reinvent.runmodes.impala_rl.trajectory import Trajectory

logger = logging.getLogger(__name__)

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
        try:
            self.nc = await nats.connect(
                self.queue_url,
                name="learner",
                reconnect_time_wait=2,
                max_reconnect_attempts=-1 
            )
            self.js = self.nc.jetstream()
            self.psub = await self.js.pull_subscribe("jobs.result", durable="center_processor")
            print("[Learner] Успешно подключен к NATS")
        except NoServersError:
            print("[Learner] Не удалось найти сервер NATS")
            return False
        except Exception as e:
            print(f"[Learner] Ошибка подключения: {e}")
            return False
        return True

    async def GetTrajectories(self, timeout: float = 10.0, batch_size: int = 16) -> list[Trajectory]:
        """Fetch trajectories from NATS queue
        
        Args:
            timeout: Timeout in seconds for fetching
            batch_size: Number of trajectory messages to fetch
            
        Returns:
            List of Trajectory objects
        """
        if not self.psub:
            print("Ошибка: Нет соединения с JetStream")
            return []

        retry: int = 0
        
        while retry < self.max_retries:
            retry += 1
            try:
                msgs = await self.psub.fetch(batch_size, timeout=timeout)
                trajectories: list[Trajectory] = []

                for msg in msgs:
                    # Deserialize list of trajectory dicts
                    data_list = json.loads(msg.data.decode())
                    
                    # Convert each dict to Trajectory object
                    for traj_dict in data_list:
                        traj = Trajectory.from_dict(traj_dict)
                        trajectories.append(traj)
                    
                    await msg.ack()
                
                return trajectories
                
            except nats.errors.TimeoutError:
                logging.log(level=1, msg=f"Timeout on fetching data. Current retry: {retry}. Current size of queue: {self.psub.pending_msgs}")
                continue
            except Exception as e:
                logging.log(level=2, msg=f"Ошибка: {e}")
                raise e
                
        logging.log(level=1, msg=f"GetTrajectories ran out of retries. Current size of queue: {self.psub.pending_msgs}")
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
        try:
            import torch
            import io
            
            # Increment version
            self.model_version += 1
            
            logger.info(f"[Learner] Committing model version {self.model_version}")
            
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
            buffer = io.BytesIO()
            torch.save(model.network.state_dict(), buffer)
            model_bytes = buffer.getvalue()
            
            model_size_mb = len(model_bytes) / (1024*1024)
            logger.info(f"[Learner] Model size: {model_size_mb:.2f} MB")
            
            # For large models (>10MB), use chunked upload
            if model_size_mb > 10:
                logger.info(f"[Learner] Using chunked upload for large model")
                chunk_size = 5 * 1024 * 1024  # 5MB chunks
                num_chunks = (len(model_bytes) + chunk_size - 1) // chunk_size
                
                # Save number of chunks first
                await kv.put(f"{self.model_key}_chunks", str(num_chunks).encode())
                
                # Upload chunks
                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min(start + chunk_size, len(model_bytes))
                    chunk = model_bytes[start:end]
                    await kv.put(f"{self.model_key}_chunk_{i}", chunk)
                    logger.info(f"[Learner] Uploaded chunk {i+1}/{num_chunks}")
            else:
                # Small model - direct upload
                await kv.put(self.model_key, model_bytes)
            
            logger.info(f"[Learner] Model saved to NATS KV: {self.model_bucket}/{self.model_key}")
            
            # Notify swarm about new model
            await self._notify_swarm_update()
            
        except Exception as e:
            logger.error(f"[Learner] Failed to commit model: {e}")
            raise
    
    async def _notify_swarm_update(self):
        """Send notification to swarm about new model version."""
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
            
            logger.info(f"[Learner] Notified swarm about model v{self.model_version}")
            
        except Exception as e:
            logger.error(f"[Learner] Failed to notify swarm: {e}")
