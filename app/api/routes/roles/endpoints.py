import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.exceptions import RequestValidationError
from typing import List
from mongoengine.errors import ValidationError, NotUniqueError
from app.core.config import Config
from .schemas import (
    RoleResponse, 
    RoleCreateRequest, 
    RoleUpdateRequest, 
    OK, 
)
from app.db.mongo.role import Role, Endpoint, Method

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = Config()

router = APIRouter(tags=["Roles"])

# ----- GET ----- 
@router.get(
    "/roles",
    response_model=List[RoleResponse],
    tags=["Roles"],
    dependencies=[Depends(CONFIG.API_LVL2_RATE_LIMITER)],
    description="List all roles in the system."
)
async def api_roles() -> List[RoleResponse]:
    """List all roles in the system."""
    try:
        roles = Role.objects().order_by('-created_at')  # type: ignore[attr-defined]
        return [RoleResponse.role_to_response(role) for role in roles]
    except Exception as e:
        logger.error(f"Error fetching roles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch roles"
        )
# ----- GET -----




# ----- GET ----- 
@router.get(
    "/roles/{role_id}",
    response_model=RoleResponse,
    tags=["Roles"],
    dependencies=[Depends(CONFIG.API_LVL2_RATE_LIMITER)],
    description="Get a specific role by its ID."
)
def api_role(role_id: str) -> RoleResponse:
    """Get a specific role by its ID."""
    try:
        role = Role.objects(id=role_id).first()  # type: ignore[attr-defined]
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail=f"Role with ID '{role_id}' not found"
            )
        return RoleResponse.role_to_response(role)
    except Exception as e:
        logger.error(f"Error fetching role {role_id}: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch role"
        )
# ----- GET ----- 




# ----- POST ----- 
@router.post(
    "/roles",
    response_model=RoleResponse,
    status_code=201,
    tags=["Roles"],
    dependencies=[Depends(CONFIG.API_LVL2_RATE_LIMITER)],
    description="Create a new role with specified endpoints."
)
async def api_create_role(role_data: RoleCreateRequest) -> RoleResponse:
    """Create a new role with specified endpoints."""
    try:
        # Check if role already exists
        existing_role = Role.objects(rolename=role_data.rolename).first()  # type: ignore[attr-defined]
        if existing_role:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Role '{role_data.rolename}' already exists"
            )
        
        # create endpoints
        endpoints = [
            Endpoint(
                method=Method(e.method.value),
                path_filter=e.path_filter
            )
            for e in role_data.endpoints
        ]
        
        # Create role
        new_role = Role(
            rolename=role_data.rolename,
            api_endpoints=endpoints
        )
        new_role.save()
        
        logger.info(f"Created new role: {new_role.rolename} (ID: {new_role.id})")
        return RoleResponse.role_to_response(new_role)
        
    except HTTPException:
        raise
    except NotUniqueError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Role '{role_data.rolename}' already exists"
        )
    except ValidationError as ve:
        logger.error(f"Validation error creating role: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error creating role: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create role"
        )
# ----- POST ----- 




# ----- PUT ----- 
@router.put(
    "/roles/{role_id}",
    response_model=RoleResponse,
    tags=["Roles"],
    dependencies=[Depends(CONFIG.API_LVL2_RATE_LIMITER)],
    description="Update an existing role."
)
async def api_update_role(role_id: str, role_data: RoleUpdateRequest) -> RoleResponse:
    """Update an existing role."""
    try:
        # Find the role to update
        role = Role.objects(id=role_id).first()  # type: ignore[attr-defined]
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Role with ID '{role_id}' not found"
            )
        
        # Check if new rolename conflicts with existing roles
        if role_data.rolename and role_data.rolename != role.rolename:
            existing_role = Role.objects(rolename=role_data.rolename).first()  # type: ignore[attr-defined]
            if existing_role:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Role '{role_data.rolename}' already exists"
                )
            role.rolename = role_data.rolename
        
        # Update endpoints if provided
        if role_data.endpoints is not None:
            endpoints = [
                Endpoint(
                    method=Method(e.method.value),
                    path_filter=e.path_filter
                )
                for e in role_data.endpoints
            ]
            role.api_endpoints = endpoints
        
        # Save the updated role
        role.save()
        
        logger.info(f"Updated role: {role.rolename} (ID: {role.id})")
        return RoleResponse.role_to_response(role)
        
    except HTTPException:
        raise
    except NotUniqueError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Role '{role_data.rolename}' already exists"
        )
    except ValidationError as ve:
        logger.error(f"Validation error updating role {role_id}: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error updating role {role_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update role"
        )
# ----- PUT ----- 




# ----- DELETE ----- 
@router.delete(
    "/roles/{role_id}",
    response_model=OK,
    tags=["Roles"],
    dependencies=[Depends(CONFIG.API_LVL2_RATE_LIMITER)],
    description="Delete a role by its ID."
)
async def api_delete_role(role_id: str) -> OK:
    """Delete a role by its ID."""
    try:
        # Find the role to delete
        role = Role.objects(id=role_id).first()  # type: ignore[attr-defined]
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Role with ID '{role_id}' not found"
            )
        
        # Delete the role (signals will handle cleanup)
        role_name = role.rolename
        role.delete()
        
        logger.info(f"Deleted role: {role_name} (ID: {role_id})")
        return OK(ok=True)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting role {role_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete role"
        )
# ----- DELETE ----- 