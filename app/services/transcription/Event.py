import logging
import threading
from typing import Any, Callable, Dict


class EventHandler:
    """Class responsible for managing callbacks."""

    def __init__(self) -> None:
        logging.debug("Initializing EventHandler.")
        self._callbacks: Dict[int, Callable[..., None]] = {}
        self._event_lock = threading.Lock()
        self._next_id: int = 0

    def add_event(self, callback: Callable[..., None]) -> int:
        """Add a new event callback and return its unique ID."""
        with self._event_lock:
            event_id = self._next_id
            self._next_id += 1
            logging.debug(f"Adding event with ID: {event_id}")
            self._callbacks[event_id] = callback
            return event_id

    def remove_event(self, event_id: int) -> None:
        """Remove an event callback using its ID."""
        logging.debug(f"Removing event with ID: {event_id}")
        with self._event_lock:
            self._callbacks.pop(event_id, None)

    def emit(self, *args: Any) -> None:
        """Trigger all the registered callbacks with the provided arguments."""
        threads = []  # To keep track of the threads

        with self._event_lock:
            for event_id, callback in self._callbacks.items():
                # logging.debug(f"Emitting event with ID: {event_id}")
                # Wrap the callback execution in a thread
                t = threading.Thread(target=callback, args=args)
                t.daemon = True  # This will ensure the thread stops when the main thread exits
                threads.append(t)
                t.start()

        # Wait for all threads to finish
        for t in threads:
            t.join()
