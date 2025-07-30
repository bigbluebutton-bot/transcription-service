from fastapi import APIRouter
from app.api.routes.system.endpoints import router as system_router


api_router = APIRouter()

# include routs from system
api_router.include_router(system_router)
