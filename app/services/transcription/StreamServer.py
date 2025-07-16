import inspect
import logging
import time
import json
from typing import Callable, Dict, Optional, Tuple
import TCPserver as TCPserver
import UDPserver as UDPserver
import Event as event

EventHandler = event.EventHandler

class Client:
    def __init__(self, on_remove: Callable[['Client'], None], tcpclient: TCPserver.Client, udpclient: UDPserver.Client) -> None:
        self._on_remove = on_remove
        self._tcpclient = tcpclient
        self._udpclient = udpclient

        # if the tcp client disconnects or timeouts, remove the client
        self._tcpclient.on_event("disconnected", lambda c: self.stop())
        self._tcpclient.on_event("timeout", lambda c: self.stop())

    def udp_address(self) -> Tuple[str, int | None]:
        """Return the client's UDP address."""
        return self._udpclient.address()

    def tcp_address(self) -> Tuple[str, int]:
        """Return the client's TCP address."""
        address = self._tcpclient.address()
        return (address[0], address[1] or 0)

    def stop(self) -> None:
        logging.debug(f"Stopping client {self._tcpclient.address()}, {self._udpclient.address()}.")
        self._tcpclient.stop()
        self._udpclient.stop()
        self._on_remove(self)

    def send_message(self, message: bytes) -> None:
        """Send a TCP message to the client."""
        self._tcpclient.send(message)

    def on_tcp_message(self, callback: Callable[['Client', bytes], None]) -> Optional[int]:
        """Register a new TCP message callback."""
        return self._tcpclient.on_event("message", lambda c, d: callback(self, d))

    def remove_on_tcp_message(self, callback_id: int) -> None:
        """Remove a TCP message callback using its ID."""
        return self._tcpclient.remove_event("message", callback_id)

    def on_udp_message(self, callback: Callable[['Client', bytes], None]) -> Optional[int]:
        """Register a new UDP message callback."""
        return self._udpclient.on_event("message", lambda c, d: callback(self, d))

    def remove_on_udp_message(self, callback_id: int) -> None:
        """Remove a UDP message callback using its ID."""
        return self._udpclient.remove_event("message", callback_id)

    def on_disconnected(self, callback: Callable[['Client'], None]) -> Optional[int]:
        """Register a new disconnected callback."""
        return self._tcpclient.on_event("disconnected", lambda c: callback(self))

    def remove_on_disconnected(self, callback_id: int) -> None:
        """Remove a disconnected callback using its ID."""
        return self._tcpclient.remove_event("disconnected", callback_id)

    def on_timeout(self, callback: Callable[['Client'], None]) -> Optional[int]:
        """Register a new timeout callback."""
        return self._tcpclient.on_event("timeout", lambda c: callback(self))

    def remove_on_timeout(self, callback_id: int) -> None:
        """Remove a timeout callback using its ID."""
        return self._tcpclient.remove_event("timeout", callback_id)

class Server:
    def __init__(
        self, 
        host: str, 
        tcpport: int, 
        udpport: int, 
        secrettoken: str = "", 
        encryption: int = 0, 
        timeout: int = 5, 
        maxclients: int = 10, 
        buffer_size: int = 1024, 
        external_host: str = "", 
        external_udpport: int = 0
    ) -> None:
        self._host = host
        self._tcpport = tcpport
        self._udpport = udpport
        self._secrettoken = secrettoken
        self._encryption = encryption
        self._timeout = timeout
        self._maxclients = maxclients
        self._buffer_size = buffer_size

        if external_host == "":
            self._external_host = host
        else:
            self._external_host = external_host
        if external_udpport == 0:
            self._external_udpport = udpport
        else:
            self._external_udpport = external_udpport

        self._clients: Dict[tuple[str, int], Client] = {}  # {tcpaddr: client}

        self._tcpserver = TCPserver.Server(
            self._host, 
            self._tcpport, 
            self._timeout, 
            self._encryption, 
            5, 
            self._maxclients, 
            self._secrettoken, 
            self._buffer_size
        )
        self._udpserver = UDPserver.Server(
            self._host, 
            self._udpport, 
            self._encryption, 
            self._buffer_size
        )

        self._connected_callbacks = EventHandler()

        # event. tcp client on connect:
        def _on_tcp_connected(tcpclient: TCPserver.Client) -> None:
            # 1. add client to udpserver whitelist
            clienthost = tcpclient.address()[0]
            aes_key = tcpclient.client_key
            aes_initkey = tcpclient.client_initkey

            if aes_key is None or aes_initkey is None:
                raise ValueError("AES key or initialization key is None.")

            udpclient = self._udpserver.add_client(clienthost, aes_key, aes_initkey)

            # 2. add tcp and udp client to self._clients
            client = Client(self._remove_client, tcpclient, udpclient)
            self._clients[tcpclient.address()] = client

            # 3. send udp server address to client
            udpencryption = udpclient._encryption
            jsondata = json.dumps({
                "type": "init_udpaddr", 
                "msg": {
                    "udp": {
                        "host": self._external_host, 
                        "port": self._external_udpport, 
                        "encryption": udpencryption
                    }
                }
            }).encode()
            tcpclient.send(jsondata)

            # 4. emit event connected
            self._connected_callbacks.emit(client)

        self._tcpserver.on_connected(_on_tcp_connected)

    def start(self) -> None:
        """Start the server."""
        if self._tcpserver._running:
            logging.warning("TCP server is already running.")
            return
        if self._udpserver._running:
            logging.warning("UDP server is already running.")
            return
        self._tcpserver.start()
        self._udpserver.start()

    def stop(self) -> None:
        """Stop the server."""
        self._tcpserver.stop()
        self._udpserver.stop()

    def on_connected(self, callback: Callable[[Client], None]) -> int:
        """Register a new connected callback."""
        # Get the number of parameters the callback has
        num_params = len(inspect.signature(callback).parameters)

        if num_params != 1:
            logging.error(f"Invalid number of parameters for 'connected' event. Expected 1, got {num_params}.")
            return -1

        return self._connected_callbacks.add_event(callback)

    def remove_on_connected(self, callback_id: int) -> None:
        """Remove a connected callback using its ID."""
        return self._connected_callbacks.remove_event(callback_id)

    def _remove_client(self, client: Client) -> None:
        """Remove a client from the server's client list."""
        logging.debug(f"Removing client: {client.tcp_address()}, {client.udp_address()}")
        self._clients.pop(client.tcp_address(), None)

# EXAMPLE USAGE
SECRET_TOKEN = "your_secret_token"

def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    srv = Server(
        "0.0.0.0", 
        5000, 
        5001, 
        SECRET_TOKEN, 
        4096, 
        5, 
        10, 
        1024, 
        "127.0.0.1", 
        5001
    )

    def _on_connected(client: Client) -> None:
        print(f"Client connected: {client.tcp_address()}, {client.udp_address()}")
        client.send_message(b"Hello from server! Connected")

        def _on_tcp_message(c: Client, message: bytes) -> None:
            print(f"Received TCP message from client: {c.tcp_address()}: {len(message)}")
            client.send_message(b"Hello from server! TCP")

        def _on_udp_message(c: Client, message: bytes) -> None:
            print(f"Received UDP message from client: {c.udp_address()}: {len(message)}")
            client.send_message(b"Hello from server! UDP")

        def _on_disconnected(c: Client) -> None:
            print(f"Client disconnected: {c.tcp_address()}, {c.udp_address()}")

        client.on_tcp_message(_on_tcp_message)
        client.on_udp_message(_on_udp_message)
        client.on_disconnected(_on_disconnected)

    srv.on_connected(_on_connected)

    srv.start()

    try:
        while True:  # Keep the server running until a keyboard interrupt
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping server...")
        srv.stop()

        logging.info("Server stopped.")
        logging.info("THE END")

if __name__ == "__main__":
    main()
