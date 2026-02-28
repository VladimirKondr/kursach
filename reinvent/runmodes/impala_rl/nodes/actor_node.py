import json
import logging
import torch
import nats
from nats.errors import TimeoutError

from reinvent.runmodes.impala_rl.actor import ImpalaActor
from nats.errors import NoServersError

logger = logging.getLogger(__name__)

class ActorNode:
    def __init__(self, actor: ImpalaActor, worker_id: str, queue_url: str = "localhost:4222", publish_sibject: str = "jobs.result"):
        self.queue_url = queue_url
        self.actor = actor
        self.publish_sibject = publish_sibject
        self.worker_id = worker_id

        self.nc = None
        self.js = None

    async def Connect(self):
        try:
            self.nc = await nats.connect(
                self.queue_url,
                name=self.worker_id,
                reconnect_time_wait=2,
                max_reconnect_attempts=-1 
            )
            self.js = self.nc.jetstream()
            print(f"[{self.worker_id}] Успешно подключен к NATS")
        except NoServersError:
            print(f"[{self.worker_id}] Не удалось найти сервер NATS")
            return False
        except Exception as e:
            print(f"[{self.worker_id}] Ошибка подключения: {e}")
            return False
        return True

    async def SendTrajectories(self):
        if not self.js:
            print("Ошибка: Нет соединения с JetStream")
            return
        trajectories = self.actor.collect_trajectory()
        
        # Convert Trajectory objects to dictionaries for JSON serialization
        trajectories_data = [traj.to_dict() for traj in trajectories]

        data_bytes = json.dumps(trajectories_data).encode('utf-8')

        try:
            ack = await self.js.publish(self.publish_sibject, data_bytes, timeout=5)
            logging.log(level=0, msg=f"Объект отправлен! Stream sequence: {ack.seq}")
        except TimeoutError:
            logging.log(level=2, msg="Ошибка: Сервер не подтвердил прием сообщения (таймаут)")
        except Exception as e:
            logging.log(level=1, msg=f"Ошибка отправки: {e}")

    
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
        try:
            import torch
            import io
            
            logger.info(f"[{self.worker_id}] Loading model version {version} from NATS KV")
            
            # Get KV bucket
            kv = await self.js.key_value(bucket)
            
            # Try to get chunked model first
            try:
                chunks_entry = await kv.get(f"{key}_chunks")
                num_chunks = int(chunks_entry.value.decode())
                logger.info(f"[{self.worker_id}] Downloading model in {num_chunks} chunks")
                
                # Download all chunks
                model_bytes = b""
                for i in range(num_chunks):
                    chunk_entry = await kv.get(f"{key}_chunk_{i}")
                    model_bytes += chunk_entry.value
                    logger.info(f"[{self.worker_id}] Downloaded chunk {i+1}/{num_chunks}")
                
            except Exception:
                # No chunks, try direct download
                logger.info(f"[{self.worker_id}] Downloading model directly")
                entry = await kv.get(key)
                model_bytes = entry.value
            
            # Load model state dict
            buffer = io.BytesIO(model_bytes)
            state_dict = torch.load(buffer, map_location=self.actor.device)
            
            # Update actor's adapter model (ImpalaActor uses adapter, not model)
            self.actor.adapter.network.load_state_dict(state_dict)
            self.actor.model_version = version
            
            logger.info(f"[{self.worker_id}] Successfully loaded model version {version}")
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] Failed to load model: {e}")
            raise
