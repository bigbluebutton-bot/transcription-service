import os
import socket
import threading
import logging
import inspect
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import Event as event
EventHandler = event.EventHandler

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class Client:
    def __init__(self, on_remove: Callable[[Tuple[str, int]], None], host: str, encryption: bool = False, aes_key: bytes = b'\x05', aes_initkey: bytes = b'\x05'):
        self._on_remove: Callable[[Tuple[str, int]], None] = on_remove
        self._host: str = host
        self._port: Optional[int] = None
        self._encryption: bool = encryption
        self._aes_key: bytes = aes_key
        self._aes_initkey: bytes = aes_initkey
        self._message_callback: EventHandler = EventHandler()

    def address(self) -> Tuple[str, Optional[int]]:
        """Return the server's address."""
        return (self._host, self._port)

    def stop(self) -> None:
        logging.debug(f"Removing UDP client {self._host} from whitelist.")
        if self._port is not None:
            self._message_callback = EventHandler()
            self._on_remove((self._host, self._port))

    def on_event(self, event_type: str, callback: Callable[[ 'Client', bytes], None]) -> int:
        # Get the number of parameters the callback has
        num_params = len(inspect.signature(callback).parameters)

        if event_type == "message":
            if num_params == 2:
                return self._message_callback.add_event(callback)
            else:
                logging.error(f"Invalid number of parameters for 'on_message' event. Expected 2, got {num_params}.")
        else:
            logging.warning(f"Unsupported event type: {event_type}")
            
        return -1

    def remove_event(self, event_type: str, callback_id: int) -> None:
        if event_type == "message":
            self._message_callback.remove_event(callback_id)
        else:
            logging.warning(f"Unsupported event type: {event_type}")


class Server:
    def __init__(self, host: str, port: int, encryption: int = 0, buffer_size: int = 1024):
        logging.debug("Initializing UDP server.")
        self._host: str = host
        self._port: int = port
        self._running: bool = False
        self._socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._encryption: int = encryption
        self._main_thread: Optional[threading.Thread] = None
        self._clients: Dict[str, List[Client]] = {}  # {host: [Client, ...]}
        self._clients_lock: threading.Lock = threading.Lock()
        self._connected_callbacks: EventHandler = EventHandler()
        self._buffer_size: int = buffer_size

    def start(self) -> None:
        """Start the server."""
        if self._running:
            logging.warning("UDP server is already running.")
            return

        logging.debug("Starting UDP server.")
        self._running = True
        self._socket.bind((self._host, self._port))

        self._main_thread = threading.Thread(target=self._listen)
        self._main_thread.daemon = True  # This will ensure the thread stops when the main thread exits
        self._main_thread.start()

        logging.info(f"Server started at {self._host}:{self._port}")

    def stop(self) -> None:
        """Stop the server."""
        if not self._running:
            logging.warning("UDP server is not running.")
            return
        logging.debug("Stopping UDP server.")
        self._running = False
        self._socket.sendto(b"exit", (self._host, self._port))
        self._socket.close()

        # stop all clients
        tempclients = None
        with self._clients_lock:
            tempclients = self._clients.copy()
        for clientslistaddr in tempclients:
            clientslist = tempclients[clientslistaddr]
            for client in clientslist:
                client.stop()

        if self._main_thread is not None:
            self._main_thread.join()
            logging.debug("UDP server stopped.")

    def add_client(self, host: str, aes_key: bytes = b'\x05', aes_initkey: bytes = b'\x05') -> Client:
        """Add a client to the whitelist."""
        logging.debug(f"Adding UDP client {host} to whitelist.")
        udp_encryption = False
        if self._encryption:
            udp_encryption = True

        client = Client(self._remove_client, host, udp_encryption, aes_key, aes_initkey)

        with self._clients_lock:
            if host in self._clients:
                self._clients[host].append(client)
            else:
                self._clients[host] = [client]
            return client

    def remove_client(self, address: Tuple[str, int]) -> None:
        """Remove a client from the whitelist."""
        logging.debug(f"Removing UDP client {address}.")
        host = address[0]
        with self._clients_lock:
            clientslist = self._clients.get(host)
            if clientslist is None:
                logging.debug(f"UDP client {address} not found.")
                return
            for client in clientslist:
                if client.address() == address:
                    client.stop()
                    return
            logging.debug(f"UDP client {address} not found in list.")


    def _remove_client(self, address: Tuple[str, int]) -> None:
        """Remove a client from the whitelist. Internal use only."""
        logging.debug(f"Removing UDP client {address} from whitelist.")
        host = address[0]
        with self._clients_lock:
            clientslist = self._clients.get(host)
            if clientslist is None:
                logging.debug(f"UDP client {address} not found.")
                return
            for client in clientslist:
                if client.address() == address:
                    clientslist.remove(client)
                    break

            if len(clientslist) == 0:
                self._clients.pop(host)

    def on_connected(self, callback: Callable[[Client], None]) -> int:
        """Register a callback for when a client connects."""
        # Get the number of parameters the callback has
        num_params = len(inspect.signature(callback).parameters)

        if num_params == 1:
            return self._connected_callbacks.add_event(callback)
        else:
            logging.error(f"Invalid number of parameters for 'on_connected' event. Expected 1, got {num_params}.")
            
        return -1

    def remove_on_connected(self, callback_id: int) -> None:
        self._connected_callbacks.remove_event(callback_id)

    def _decrypt(self, encrypted_data: bytes, aes_key: bytes, aes_initkey: bytes) -> bytes:
        """Decrypt the received data."""
        cipher = Cipher(algorithms.AES(aes_key), modes.CFB(aes_initkey), backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(encrypted_data) + decryptor.finalize()
        return plaintext

    def _handle_socket_errors(self, error: Exception) -> None:
        """Centralize error handling for socket-related errors."""
        logging.debug(f"UDP socket error: {error}")
        self.stop()

    def _listen(self) -> None:
        """Listen for incoming messages."""
        logging.debug("Listening for incoming UDP messages.")
        while self._running:
            try:
                data, address = self._socket.recvfrom(self._buffer_size)
                host = address[0]
                port = address[1]
            except socket.error as e:
                self._handle_socket_errors(e)
                break

            clientlist = None
            with self._clients_lock:
                clientlist = self._clients.get(host)
            if clientlist is None:
                logging.debug(f"Received UDP message from {address}, which is not in the whitelist/clientlist.")
                continue

            client = None
            for c in clientlist:
                if c.address() == address:
                    client = c
                    break

            if client is None:
                for c in clientlist:
                    if c._port is None:
                        client = c
                        client._port = port
                        break

            if client is None:
                logging.debug(f"Received UDP message from {address}, which is not in the whitelist/clientlist.")
                continue

            if client._port is None:
                client._port = port
                self._connected_callbacks.emit(client)

            if self._encryption:
                data = self._decrypt(data, client._aes_key, client._aes_initkey)

            client._message_callback.emit(client, data)


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    srv = Server("127.0.0.1", 5001, True)

    srv.on_connected(lambda client: print(f"Client {client.address()} connected."))

    aes_key = os.urandom(32)  # Generate a random 32 bytes AES key
    aes_initkey = os.urandom(16)  # Generate a random 16 bytes AES init key
    print(f"AES Key: {len(aes_key)}")
    print(f"AES Init Key: {len(aes_initkey)}")
    c1 = srv.add_client("127.0.0.1", aes_key, aes_initkey)

    c1.on_event("message", lambda c, data: print(f"Client {c.address()}: {len(data)}"))

    srv.start()

if __name__ == "__main__":
    main()
