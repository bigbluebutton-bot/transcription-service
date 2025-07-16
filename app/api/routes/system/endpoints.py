import logging
from fastapi import APIRouter, Depends
from typing import List
from config import Config, STATUS
from .schemas import APIendpointResponse, APIHealthResponse
from fastapi.routing import APIRoute


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = Config()

router = APIRouter(tags=["System"])

@router.get(
    "/api/v1/endpoints",
    response_model=List[APIendpointResponse],
    dependencies=[Depends(CONFIG.API_LVL2_RATE_LIMITER)],
    description="List all available API endpoints"
)
def list_endpoints() -> List[APIendpointResponse]:
    endpoints: List[APIendpointResponse] = []
    for route in router.routes:
        # only include standard HTTP routes (skip websockets, static, etc.)
        if isinstance(route, APIRoute):
            for method in sorted(route.methods):
                # you can also filter by prefix if you only want /api/v1/…
                # if not route.path.startswith("/api/v1"):
                #     continue
                endpoints.append(
                    APIendpointResponse(
                        method=method,
                        path=route.path
                    )
                )
    return endpoints

@router.get(
    "/api/v1/health",
    response_model=APIHealthResponse,
    dependencies=[Depends(CONFIG.API_LVL2_RATE_LIMITER)],
    description="Check if the API is running"
)
def health() -> APIHealthResponse:
    return APIHealthResponse(
        status=STATUS,
        version="1.0.0"
    )