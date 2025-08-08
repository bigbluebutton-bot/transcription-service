from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import re

class OK(BaseModel):
    """Standard OK response model."""
    ok: bool = Field(..., description="Operation success status")

class UserResponse(BaseModel):
    """Response model for user data."""
    id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Username for authentication")
    roles: List[str] = Field(
        default_factory=list, 
        description="List of role names assigned to the user"
    )
    last_login: Optional[datetime] = Field(
        None, 
        description="Timestamp of the user's last login"
    )

class UserCreate(BaseModel):
    """Request model for creating a new user."""
    username: str = Field(
        ..., 
        min_length=3, 
        max_length=50,
        description="Unique username for the new user"
    )
    password: str = Field(
        ..., 
        min_length=12, 
        max_length=256,
        description="Password for the new user (minimum 12 characters)"
    )
    roles: List[str] = Field(
        default_factory=list,
        description="List of role names to assign to the user"
    )
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate and clean the username."""
        if not v or not v.strip():
            raise ValueError('Username cannot be empty or whitespace only')
        
        # Check for valid characters (alphanumeric, hyphens, underscores, dots)
        if not re.match(r'^[a-zA-Z0-9._-]+$', v.strip()):
            raise ValueError('Username can only contain letters, numbers, dots, hyphens, and underscores')
        
        # Username cannot start or end with special characters
        if v.strip()[0] in '._-' or v.strip()[-1] in '._-':
            raise ValueError('Username cannot start or end with dots, hyphens, or underscores')
            
        return v.strip().lower()
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if not v:
            raise ValueError('Password cannot be empty')
        
        # Check for minimum complexity
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[^A-Za-z0-9]', v):
            raise ValueError('Password must contain at least one special character')
        
        return v
    
    @field_validator('roles')
    @classmethod
    def validate_roles(cls, v: List[str]) -> List[str]:
        """Validate role names list."""        
        # Remove duplicates while preserving order
        seen = set()
        unique_roles = []
        for role in v:
            if role.strip() not in seen:
                seen.add(role.strip())
                unique_roles.append(role.strip())
        
        return unique_roles

class UserUpdatePassword(BaseModel):
    """Request model for updating user password."""
    current_password: str = Field(
        ..., 
        min_length=1,
        description="Current password for verification"
    )
    new_password: str = Field(
        ..., 
        min_length=12, 
        max_length=256,
        description="New password (minimum 12 characters)"
    )
    
    @field_validator('current_password')
    @classmethod
    def validate_current_password(cls, v: str) -> str:
        """Validate current password is not empty."""
        if not v or not v.strip():
            raise ValueError('Current password cannot be empty')
        return v
    
    @field_validator('new_password')
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """Validate new password strength."""
        if not v:
            raise ValueError('New password cannot be empty')
        
        # Check for minimum complexity
        if not re.search(r'[A-Z]', v):
            raise ValueError('New password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('New password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('New password must contain at least one digit')
        if not re.search(r'[^A-Za-z0-9]', v):
            raise ValueError('New password must contain at least one special character')
        
        return v

class UserResetPassword(BaseModel):
    """Request model for resetting user password (admin operation)."""
    new_password: str = Field(
        ..., 
        min_length=12, 
        max_length=256,
        description="New password for the user (minimum 12 characters)"
    )
    
    @field_validator('new_password')
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """Validate new password strength."""
        if not v:
            raise ValueError('New password cannot be empty')
        
        # Check for minimum complexity
        if not re.search(r'[A-Z]', v):
            raise ValueError('New password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('New password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('New password must contain at least one digit')
        if not re.search(r'[^A-Za-z0-9]', v):
            raise ValueError('New password must contain at least one special character')
        
        return v

class UserSetRole(BaseModel):
    """Request model for setting user roles."""
    roles: List[str] = Field(
        ...,
        description="List of role names to assign to the user"
    )
    
    @field_validator('roles')
    @classmethod
    def validate_roles(cls, v: List[str]) -> List[str]:
        """Validate and clean role names list."""
        if not v:
            raise ValueError('At least one role must be specified')
        
        # Remove duplicates while preserving order and validate each role
        seen = set()
        unique_roles = []
        for role in v:
            if not role or not role.strip():
                raise ValueError('Role names cannot be empty or whitespace only')
            
            role_clean = role.strip()
            if role_clean not in seen:
                seen.add(role_clean)
                unique_roles.append(role_clean)
        
        if not unique_roles:
            raise ValueError('At least one valid role must be specified')
            
        return unique_roles