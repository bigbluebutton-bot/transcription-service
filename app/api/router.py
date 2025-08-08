from fastapi import APIRouter
from app.api.routes.system.endpoints import router as system_router
from app.api.routes.roles.endpoints import router as role_router
from app.api.routes.system.user.endpoints import router as user_router

api_router = APIRouter()

# include routs from system
api_router.include_router(system_router)
api_router.include_router(role_router)
api_router.include_router(user_router)