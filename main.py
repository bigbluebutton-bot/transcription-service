import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from app.api.router import api_router
from app.core.config import CONFIG, STATUS
from typing import AsyncIterator

from app.core.api_rate_limmiter import init_api_rate_limiter
from app.db.mongo.connect import connect_to_mongo, create_default_boss, disconnect_from_mongo



@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        # Rate Limiter
        await init_api_rate_limiter()
        
        # MongoDB
        await connect_to_mongo()
        await create_default_boss()

        await STATUS.set("running")

        yield
    finally:
        await disconnect_from_mongo()


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
    asyncio.run(main())
    STATUS.set("stopped")