import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.exceptions import RequestValidationError
from typing import List
from mongoengine.errors import ValidationError, NotUniqueError
from app.core.config import Config
from .schemas import (
    UserResponse,
    UserCreate,
    UserUpdatePassword,
    UserResetPassword,
    UserSetRole,
    OK,
)
from app.db.mongo.role import Role, Endpoint, Method

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = Config()

router = APIRouter(tags=["Roles"])

# ----- GET ----- 
@router.get(
    "/api/v1/users",
    response_model=List[UserResponse],
    tags=["Users"],
    dependencies=[Depends(CONFIG.API_LVL2_RATE_LIMITER)],
    description="List all users in the system."
)
async def api_users() -> List[UserResponse]:
    """List all users in the system."""
    try:
        users = User.objects().order_by('-created_at')  # type: ignore[attr-defined]
        return [UserResponse.user_to_response(user) for user in users]
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch users"
        )
