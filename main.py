import asyncio
import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

from app.api.api import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    debug=settings.DEBUG,
)

# Set all CORS enabled origins
if settings.backend_cors_origins:
    app.add_middleware(  # type: ignore
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.backend_cors_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# Mount static files directory
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    with open(Path(__file__).parent / "static" / "index.html") as f:
        return f.read()

# ---------------------------
# Main
# ---------------------------
async def main() -> None:
    # Configuration settings already loaded at module level
    # Configure the server (this does not call asyncio.run() internally)
    config = uvicorn.Config(app, host=settings.HOST, port=settings.PORT, log_level="info")
    server = uvicorn.Server(config)
    # Run the server asynchronously
    await asyncio.gather(
        server.serve()
    )

if __name__ == "__main__":
    STATUS = "running"
    asyncio.run(main())
    STATUS = "stopped"