from typing import List, Dict, Any
from redis.asyncio.cluster import RedisCluster, ClusterNode
from app.core.config import CONFIG
import logging

async def get_redis_cluster(
    nodes: List[Dict[str, Any]] = CONFIG.REDIS_CLUSTER_NODES,
    decode_responses: bool = True,
    encoding: str = "utf-8",
    **redis_kwargs
) -> RedisCluster:
    """
    Create and return a RedisCluster object from a list of node dicts.

    Args:
      nodes: List of {"host": <hostname>, "port": <port>} dicts.
      decode_responses: Whether to decode responses to str (default True).
      encoding: Encoding to use when decoding (default "utf-8").
      **redis_kwargs: Any other RedisCluster keyword arguments
                     (e.g. socket_timeout, max_connections, etc.)

    Returns:
      An instance of redis.cluster.RedisCluster.
    """
    # Simple approach: use standard Redis connection
    logging.info(f"Connecting to Redis using URL: {CONFIG.REDIS_URL}")

    if not isinstance(nodes, list) or not nodes:
        raise ValueError("`nodes` must be a non-empty list of dicts")

    startup_nodes = []
    for idx, node in enumerate(nodes):
        host = node.get("host")
        port = node.get("port")
        if host is None or port is None:
            raise ValueError(f"node at index {idx} is missing 'host' or 'port'")
        startup_nodes.append(ClusterNode(host, port))

    return RedisCluster(
        startup_nodes=startup_nodes,
        decode_responses=decode_responses,
        encoding=encoding,
        **redis_kwargs
    )


if __name__ == "__main__":
    # Example usage
    config = [
        {"host": "localhost", "port": 6381},
        # you can add more cluster nodes here
    ]

    rc = get_redis_cluster(
        nodes=config,
        decode_responses=True,
        encoding="utf-8",
        # any other RedisCluster kwargs:
        # socket_timeout=5,
        # max_connections=100,
    )

    # test
    rc.set("foo", "bar")
    print(rc.get("foo"))  # → "bar"