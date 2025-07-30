from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import re
from app.db.mongo.role import Role

class OK(BaseModel):
    """Standard OK response model."""
    ok: bool = Field(..., description="Operation success status")

class MethodResponse(str, Enum):
    """HTTP methods supported by the API endpoints."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    ANY = "ANY"

class EndpointResponse(BaseModel):
    """Response model for API endpoint configuration."""
    method: MethodResponse = Field(..., description="HTTP method for the endpoint")
    path_filter: str = Field(..., description="Regex pattern to match API paths")
    
    @field_validator('path_filter')
    @classmethod
    def validate_path_filter(cls, v: str) -> str:
        """Validate that path_filter is a valid regex pattern."""
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        return v

class RoleResponse(BaseModel):
    """Response model for role data."""
    id: str = Field(..., description="Unique role identifier")
    rolename: str = Field(..., description="Human-readable role name")
    endpoints: List[EndpointResponse] = Field(
        default_factory=list, 
        description="List of API endpoints this role can access"
    )

    @staticmethod
    def role_to_response(role: Role) -> "RoleResponse":
        """Convert Role object to RoleResponse."""
        return RoleResponse(
            id=str(role.id),
            rolename=role.rolename,
            endpoints=[
                EndpointResponse(
                    method=MethodResponse(e.method.value), 
                    path_filter=e.path_filter
                ) for e in role.api_endpoints
            ]
        )

class RoleCreateRequest(BaseModel):
    """Request model for creating a new role."""
    rolename: str = Field(
        ..., 
        min_length=1, 
        max_length=50,
        description="Unique name for the role"
    )
    endpoints: List[EndpointResponse] = Field(
        default_factory=list,
        description="List of API endpoints to assign to this role"
    )
    
    @field_validator('rolename')
    @classmethod
    def validate_rolename(cls, v: str) -> str:
        """Validate and clean the role name."""
        if not v or not v.strip():
            raise ValueError('Role name cannot be empty or whitespace only')
        
        # Check for valid characters (alphanumeric, spaces, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9\s_-]+$', v.strip()):
            raise ValueError('Role name can only contain letters, numbers, spaces, hyphens, and underscores')
        
        return v.strip()

class RoleUpdateRequest(BaseModel):
    """Request model for updating an existing role."""
    rolename: Optional[str] = Field(
        None,
        min_length=1,
        max_length=50,
        description="New name for the role (optional)"
    )
    endpoints: Optional[List[EndpointResponse]] = Field(
        None,
        description="New list of API endpoints for this role (optional)"
    )
    
    @field_validator('rolename')
    @classmethod
    def validate_rolename(cls, v: Optional[str]) -> Optional[str]:
        """Validate and clean the role name if provided."""
        if v is None:
            return v
            
        if not v or not v.strip():
            raise ValueError('Role name cannot be empty or whitespace only')
        
        # Check for valid characters
        if not re.match(r'^[a-zA-Z0-9\s_-]+$', v.strip()):
            raise ValueError('Role name can only contain letters, numbers, spaces, hyphens, and underscores')
        
        return v.strip()