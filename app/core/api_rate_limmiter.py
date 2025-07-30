from app.db.redis.redis import get_redis_cluster
from fastapi import Request, Response
from http import HTTPStatus
from fastapi_limiter import FastAPILimiter
from starlette.datastructures import Headers
import logging
from math import ceil
from fastapi import HTTPException
from typing import Union

async def service_name_identifier(request: Request) -> Union[str, Headers]:
    if request.client is None:
        return "unknown"
    return request.headers.get("Service-Name") or request.client.host  # Identify by IP if no header

async def rate_limit_exceeded_callback(request: Request, response: Response, pexpire: int) -> None:
    """
    default callback when too many requests
    :param request:
    :param pexpire: The remaining milliseconds
    :param response:
    :return:
    """
    expire = ceil(pexpire / 1000)

    raise HTTPException(
        HTTPStatus.TOO_MANY_REQUESTS,
        f"Too Many Requests. Retry after {expire} seconds.",
        headers={"Retry-After": str(expire)},
    )

async def init_api_rate_limiter():
    # Initialize FastAPILimiter with Redis
    try:        
        # Create Redis client
        redis_connection = await get_redis_cluster()
        
        # Test the connection
        await redis_connection.ping()
        logging.info("Redis ping successful")
        
        # Initialize FastAPILimiter with the Redis client
        await FastAPILimiter.init(
            redis_connection,
            identifier=service_name_identifier,
            http_callback=rate_limit_exceeded_callback,
        )
        logging.info("Successfully connected to Redis and initialized FastAPILimiter")
    except Exception as e:
        logging.error(f"Error connecting to Redis: {e}")
        raise