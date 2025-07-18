from fastapi.openapi.models import Response
import asyncio
import uvicorn
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
from starlette.datastructures import Headers
from app.api.router import api_router
from app.core.config import CONFIG, STATUS
from fastapi import Request
from typing import Union
from math import ceil
from fastapi import HTTPException
from http import HTTPStatus

from app.db.redis.redis import get_redis_cluster

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

@asynccontextmanager
async def lifespan(app: FastAPI):
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
        
    yield
    # Cleanup (if needed)
    # You can add cleanup code here if necessary


app = FastAPI(
    title=CONFIG.PROJECT_NAME,
    openapi_url=f"{CONFIG.API_V1_PREFIX}/openapi.json",
    debug=CONFIG.DEBUG,
    lifespan=lifespan
)

# Set all CORS enabled origins
if CONFIG.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in CONFIG.BACKEND_CORS_ORIGINS] + [CONFIG.EXTERNAL_URL],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=CONFIG.API_V1_PREFIX)

# Mount static files directory
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "app" / "web"), name="static")


# The startup event has been replaced with the lifespan context manager above


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    with open(Path(__file__).parent / "app" / "web" / "index.html") as f:
        return f.read()

# ---------------------------
# Main
# ---------------------------
async def main() -> None:
    # Configuration settings already loaded at module level
    # Configure the server (this does not call asyncio.run() internally)
    config = uvicorn.Config(app, host=CONFIG.HOST, port=CONFIG.PORT, log_level="info")
    server = uvicorn.Server(config)
    # Run the server asynchronously
    await asyncio.gather(
        server.serve()
    )

if __name__ == "__main__":
    STATUS = "running"
    asyncio.run(main())
    STATUS = "stopped"