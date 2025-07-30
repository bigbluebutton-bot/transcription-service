import socket
import threading
import logging
import time
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, Any

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
import Event as event

EventHandler = event.EventHandler


# Convert the byte string to a list of integers for logging
def byte_string_to_int_list(byte_string: bytes) -> list[int]:
    return [byte for byte in byte_string]



class Client:
    """Class representing a client."""

    def __init__(
        self,
        on_remove: Callable[['Client'], None],
        _connected_callbacks: EventHandler,
        conn: socket.socket,
        addr: tuple[str, int],
        timeout: int = 5,
        encryption: int = 0,
        public_key: Optional[rsa.RSAPublicKey] = None,
        private_key: Optional[rsa.RSAPrivateKey] = None,
        secretToken: str = "",
        buffer_size: int = 1024,
    ) -> None:
        logging.debug("Initializing Client.")
        self._on_remove: Callable[['Client'], None] = on_remove
        self._connected_callbacks: EventHandler = _connected_callbacks
        self.conn: Optional[socket.socket] = conn
        self.conn.settimeout(timeout)
        self._ping_timeout_time: int = timeout
        self.addr: tuple[str, int] = addr
        self._disconnected_callbacks: EventHandler = EventHandler()
        self._timeout_callbacks: EventHandler = EventHandler()
        self._message_callbacks: EventHandler = EventHandler()
        self._running: bool = False
        self._encryption: int = encryption
        self._last_ping: float = 0
        self._ping_callbacks: EventHandler = EventHandler()

        self.server_publickey: Optional[rsa.RSAPublicKey] = public_key
        self.server_privatekey: Optional[rsa.RSAPrivateKey] = private_key

        self.client_key: Optional[bytes] = None
        self.client_initkey: Optional[bytes] = None

        self._ping_message: bytes = b"PING"

        self.secretToken: str = secretToken
        self.buffer_size: int = buffer_size

    def address(self) -> tuple[str, int]:
        """Return the client's address."""
        return self.addr

    def start(self) -> None:
        """Start the client listener."""
        logging.debug(f"Client[{self.addr}] Starting client.")

        if not self.conn or self._running:
            return
        self._running = True

        self._reset_ping()

        if self._encryption:
            self._send_server_publickey()
            if not self._listen_for_clientkey():  # if returns false = error
                return

        self.send(b"OK")

        self._reset_ping()

        if not self._validate_token():
            logging.warning(f"Invalid token from {self.addr}. Closing connection.")
            self.stop()
            return

        logging.info(f"Valid token received from {self.addr}. Connection authorized.")

        self._connected_callbacks.emit(self)

        self._listen()

    def _validate_token(self) -> bool:
        """Validate the token sent by the client."""
        logging.debug(f"Client[{self.addr}] Validating token.")

        if self.conn is None:
            return False

        while self._running:
            try:
                current_time = time.time()
                if current_time - self._last_ping > self._ping_timeout_time:
                    self._ping_timeout()
                    return False

                data = self.conn.recv(self.buffer_size)
                if data:
                    if self._encryption:
                        logging.debug(f"Client[{self.addr}] Received encrypted data: {byte_string_to_int_list(data)}")
                        data = self._decrypt(data)

                    logging.debug(f"Client[{self.addr}] Received data: {byte_string_to_int_list(data)}")

                    if data.decode('utf-8') != self.secretToken:
                        return False
                    else:
                        return True

            except (socket.timeout, socket.error, OSError) as e:
                if isinstance(e, socket.timeout):
                    self._ping_timeout()
                else:
                    self._handle_socket_errors(e)
                return False

        return False

    def _handle_ping(self) -> None:
        """Handle the ping message."""
        logging.debug(f"Received ping from {self.addr}")
        self._reset_ping()
        self._ping_callbacks.emit(self)
        self.send(b"PONG")

    def _reset_ping(self) -> None:
        """Reset the ping timer."""
        logging.debug(f"Client[{self.addr}] Resetting ping timer.")
        self._last_ping = time.time()

    def _ping_timeout(self) -> None:
        """Emit timeout and stop connection."""
        logging.warning(f"Client[{self.addr}] Ping interval exceeded. Closing connection.")
        self._timeout_callbacks.emit(self)
        self.stop()

    def _listen_for_clientkey(self) -> bool:
        """Listen for the client's key."""
        logging.debug(f"Client[{self.addr}] Listening for client key.")

        if self.conn is None or self.server_privatekey is None:
            return False

        while self._running:
            try:
                current_time = time.time()
                if current_time - self._last_ping > self._ping_timeout_time:
                    self._ping_timeout()
                    return False

                data = self.conn.recv(self.buffer_size)
                if data:
                    logging.debug(f"Client[{self.addr}] Received client key: {byte_string_to_int_list(data)}")
                    init_and_key = self.server_privatekey.decrypt(
                        data,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )

                    self.client_initkey = init_and_key[:16]  # the first 16 bytes are the init vector
                    self.client_key = init_and_key[16:]      # the rest is the key

                    logging.debug(f"Client[{self.addr}] Decrypted AES Key: {byte_string_to_int_list(self.client_key)}")
                    logging.debug(f"Client[{self.addr}] Decrypted AES IV: {byte_string_to_int_list(self.client_initkey)}")

                    return True

                else:
                    logging.debug(f"Client[{self.addr}] No data received. Closing connection.")
                    self.stop()
                    return False

            except (socket.timeout, socket.error, OSError, Exception) as e:
                if isinstance(e, socket.timeout):
                    self._ping_timeout()
                else:
                    self._handle_socket_errors(e)
                return False

        return False

    def _send_server_publickey(self) -> None:
        """Send the server's public key to the client."""
        logging.debug(f"Client[{self.addr}] Sending server public key.")

        if self.server_publickey is None:
            return

        server_publickey = self.server_publickey.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        self.send(server_publickey)

    def _decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt the received data."""
        logging.debug(f"Client[{self.addr}] Decrypting data")

        if self.client_key is None or self.client_initkey is None:
            return b""

        cipher = Cipher(algorithms.AES(self.client_key), modes.CFB(self.client_initkey), backend=default_backend())

        decryptor = cipher.decryptor()

        plaintext = decryptor.update(encrypted_data) + decryptor.finalize()
        return plaintext

    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt the data to be sent."""
        logging.debug(f"Client[{self.addr}] Encrypting data: {len(data)}")

        if self.client_key is None or self.client_initkey is None:
            return b""

        cipher = Cipher(algorithms.AES(self.client_key), modes.CFB(self.client_initkey), backend=default_backend())

        encryptor = cipher.encryptor()

        ciphertext = encryptor.update(data) + encryptor.finalize()
        return ciphertext

    def _handle_socket_errors(self, error: Exception) -> None:
        """Centralize error handling for socket-related errors."""
        logging.debug(f"Client[{self.addr}] Socket error: {error}")
        self.stop()

    def _listen(self) -> None:
        """Private method to listen for incoming data from the client."""
        logging.debug(f"Client[{self.addr}] Listening for data.")

        if self.conn is None:
            return

        while self._running:
            try:
                current_time = time.time()
                if current_time - self._last_ping > self._ping_timeout_time:
                    self._ping_timeout()
                    return

                data = self.conn.recv(self.buffer_size)
                if data:
                    if self._encryption:
                        logging.debug(f"Client[{self.addr}] Received encrypted data: {byte_string_to_int_list(data)}")
                        data = self._decrypt(data)

                    logging.debug(f"Client[{self.addr}] Received data: {byte_string_to_int_list(data)}")

                    if data == self._ping_message:
                        self._handle_ping()
                    else:
                        self._message_callbacks.emit(self, data)

            except (socket.timeout, socket.error, OSError) as e:
                if isinstance(e, socket.timeout):
                    self._ping_timeout()
                else:
                    self._handle_socket_errors(e)

    def stop(self) -> None:
        """Stop the client and close its connection."""
        if not self._running:
            logging.warning(f"Client[{self.addr}] already stopped.")
            return

        logging.debug(f"Client[{self.addr}] Stopping client.")
        self._running = False

        if not self.conn:
            return
        self._disconnected_callbacks.emit(self)

        try:
            self.conn.shutdown(socket.SHUT_RDWR)
            self.conn.close()
        except Exception as e:
            logging.error(f"Error while closing client connection: {e}")
        self.conn = None
        self._on_remove(self)

        logging.debug(f"Thread: Stopped for client: {self.addr}")

    def send(self, data: bytes) -> None:
        """Send data to the client."""
        try:
            logging.debug(f"Client[{self.addr}] Sending data: {len(data)}")

            if self.conn is None:
                raise Exception("Connection is None.")

            if self._encryption and self.client_key and self.client_initkey:
                data = self._encrypt(data)

            self.conn.sendall(data)
        except (OSError, Exception) as e:
            self._handle_socket_errors(e)

    def on_event(self, event_type: str, callback: Callable) -> Optional[int]:
        """Register an event callback based on the event type."""
        num_params = len(inspect.signature(callback).parameters)

        if event_type == "disconnected":
            if num_params == 1:
                return self._disconnected_callbacks.add_event(callback)
            else:
                logging.error(f"Invalid number of parameters for 'disconnected' event. Expected 1, got {num_params}.")
        elif event_type == "timeout":
            if num_params == 1:
                return self._timeout_callbacks.add_event(callback)
            else:
                logging.error(f"Invalid number of parameters for 'timeout' event. Expected 1, got {num_params}.")
        elif event_type == "message":
            if num_params == 2:
                return self._message_callbacks.add_event(callback)
            else:
                logging.error(f"Invalid number of parameters for 'message' event. Expected 2, got {num_params}.")
        elif event_type == "ping":
            if num_params == 1:
                return self._ping_callbacks.add_event(callback)
            else:
                logging.error(f"Invalid number of parameters for 'ping' event. Expected 1, got {num_params}.")
        else:
            logging.warning(f"Unsupported event type: {event_type}")
        return None

    def remove_event(self, event_type: str, event_id: int) -> None:
        """Remove an event callback based on the event type."""
        if event_type == "disconnected":
            self._disconnected_callbacks.remove_event(event_id)
        elif event_type == "timeout":
            self._timeout_callbacks.remove_event(event_id)
        elif event_type == "message":
            self._message_callbacks.remove_event(event_id)
        elif event_type == "ping":
            self._ping_callbacks.remove_event(event_id)
        else:
            logging.warning(f"Unsupported event type: {event_type}")


class Server:
    """Class representing a TCP server."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout: int = 5,
        encryption: int = 0,
        backlog: int = 5,
        max_threads: int = 10,
        secretToken: str = "",
        buffer_size: int = 1024,
    ) -> None:
        logging.debug("Initializing Server.")
        self.host: str = host
        self.port: int = port
        self.backlog: int = backlog
        self.timeout: int = timeout
        self._connected_callbacks: EventHandler = EventHandler()
        self._clients: list[Client] = []
        self._clients_lock: threading.Lock = threading.Lock()
        self._socket: Optional[socket.socket] = None
        self._running: bool = False
        self.max_threads: int = max_threads
        self.main_accept_clients_thread: Optional[threading.Thread] = None
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self.active_clients_count: int = 0
        self._encryption: int = encryption
        self.public_key: Optional[rsa.RSAPublicKey] = None
        self.private_key: Optional[rsa.RSAPrivateKey] = None
        self.secretToken: str = secretToken
        self.buffer_size: int = buffer_size

    def generate_keys(self) -> None:
        """Generate RSA keys."""
        logging.debug("Generating RSA keys.")
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

    def start(self) -> None:
        """Start the server."""
        if self._encryption:
            self.generate_keys()

        logging.debug("Starting server.")
        if self._socket:
            logging.warning("Server already started.")
            return

        self._running = True

        self.main_accept_clients_thread = threading.Thread(target=self._accept_clients)
        self.main_accept_clients_thread.daemon = True
        self._thread_pool = ThreadPoolExecutor(self.max_threads)

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.settimeout(self.timeout)
            self._socket.bind((self.host, self.port))
            self._socket.listen(self.backlog)

            logging.debug("Thread: Starting client acceptance.")
            self.main_accept_clients_thread.start()

        except Exception as e:
            logging.error(f"Failed to start server: {e}")
            self._running = False

        logging.info(f"Server started: {self.host}:{self.port}")

    def _accept_clients(self) -> None:
        """Private method to accept incoming clients."""
        logging.debug("Accepting clients.")
        while True:
            if not self._running:
                break

            try:
                logging.debug("Waiting for client...")

                if self._socket is None:
                    logging.error("Server socket is None. Stopping client acceptance.")
                    break

                conn, addr = self._socket.accept()
                if conn and self._running:
                    logging.debug(f"Accepted client: {addr}")
                    client = Client(
                        self._remove_client,
                        self._connected_callbacks,
                        conn,
                        addr,
                        self.timeout,
                        self._encryption,
                        self.public_key,
                        self.private_key,
                        self.secretToken,
                        self.buffer_size,
                    )
                    with self._clients_lock:
                        self._clients.append(client)

                    if self._thread_pool is None:
                        logging.error("Thread pool is None. Stopping client acceptance.")
                        break

                    logging.debug(f"Thread: Starting for client: {addr}")
                    self._thread_pool.submit(client.start)
                    self.active_clients_count += 1
            except socket.timeout:
                logging.debug("Main socket timeout. But can be ignored.")
                pass
            except socket.error as e:
                if e.errno == 10038:
                    logging.info("Server socket closed. Stopping client acceptance.")
                else:
                    logging.error(f"Error accepting clients: {e}")

    def _remove_client(self, client: Client) -> None:
        """Private method to remove a client from the server's client list."""
        with self._clients_lock:
            logging.debug(f"Removing TCP client: {client.addr}")
            self._clients.remove(client)
            self.active_clients_count -= 1

    def stop(self) -> None:
        """Stop the server."""
        if not self._running:
            logging.warning("Server already stopped.")
            return

        logging.debug("Stopping server.")

        self._running = False

        logging.debug("Stopping clients.")
        for client in self._clients[:]:
            client.stop()

        logging.debug("Stopping server socket.")
        if self._socket:
            self._socket.close()
            self._socket = None

        logging.debug("Thread: Stopping client acceptance.")
        if self.main_accept_clients_thread and self.main_accept_clients_thread.is_alive():
            self.main_accept_clients_thread.join()

        logging.debug("Thread: Shutting down thread pool.")
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)

    def on_connected(self, callback: Callable[[Client], Any]) -> int:
        """Register a callback for when a client connects."""
        logging.debug("Registering 'connected' event.")
        num_params = len(inspect.signature(callback).parameters)

        if num_params != 1:
            logging.error(f"Invalid number of parameters for 'connected' event. Expected 1, got {num_params}.")
            return -1

        return self._connected_callbacks.add_event(callback)

    def remove_connected_event(self, event_id: int) -> None:
        """Remove the connected event callback using its ID."""
        logging.debug("Removing 'connected' event.")
        self._connected_callbacks.remove_event(event_id)




# EXAMPLE USAGE
SECRET_TOKEN = "your_secret_token"

def handle_client_message(client: Client, data: bytes) -> None:
    """Handle received message after token validation."""
    logging.info(f"Received from {client.addr}: {data.decode('utf-8')}")
    client.send(b"OK")

def on_connected(client: Client) -> None:
    """Handle new client connection."""
    logging.info(f"Connected by {client.addr}")
    client.on_event("disconnected", lambda c: logging.info(f"Disconnected by {c.addr}"))
    client.on_event("timeout", lambda c: logging.info(f"Timeout by {c.addr}"))
    client.on_event("message", handle_client_message)
    client.on_event("ping", lambda c: logging.info(f"Ping from {c.addr}"))

def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    srv = Server('localhost', 5000, 5, True, 5, 10, SECRET_TOKEN)

    srv.on_connected(on_connected)

    logging.info("Starting server: 127.0.0.1:5000...")
    srv.start()
    logging.info("Waiting for connections...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping server...")
        logging.info(f"Disconnecting from {srv.active_clients_count} clients...")
        srv.stop()

        while srv.active_clients_count > 0:
            logging.info(f"Waiting for {srv.active_clients_count} clients to disconnect...")
            time.sleep(1)
        logging.info("Server stopped.")

        logging.info("THE END")

if __name__ == '__main__':
    main()
