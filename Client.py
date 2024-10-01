import os
from queue import Queue
import threading
import uuid
from typing import Any, Optional

# Client class for each connected Client to handle the data separately.
class Client:
    def __init__(self, client: Any) -> None:
        self.id: str = str(uuid.uuid4())
        self.mutex: threading.Lock = threading.Lock()
        self._client: Any = client
        self._instance: Optional[str] = None

    def send(self, data: Any) -> None:
        self._client.send_message(data)

    def stop(self) -> None:
        with self.mutex:
            self._client.stop()
            self.data_queue: Queue = Queue()
            self.last_sample: bytes = bytes()
