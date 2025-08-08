from __future__ import annotations
import uuid
from app.db.mongo.role import Endpoint
"""MongoEngine document model for application users with full static typing support.

This version is mypy-clean when used together with the ``mongo-types`` stub
package for MongoEngine (``pip install mongo-types``) and modern ``bcrypt``
(>=3.2, which ships its own type information).
"""

from datetime import datetime, timezone
from typing import ClassVar

import bcrypt
from mongoengine import Document, signals
from mongoengine.errors import ValidationError
PULL = 4
from mongoengine.fields import (
    StringField,
    BooleanField,
    DateTimeField,
    ListField,
    ReferenceField,
)
from mongoengine.queryset.manager import QuerySetManager  # type: ignore



class User(Document):
    """Application user stored in the ``users`` MongoDB collection."""
    objects: QuerySetManager
    _signals_connected: ClassVar[bool] = False
    
    # Metadata
    meta = {
        'collection': 'users',
        'indexes': [
            'username',  # Index for faster username lookups
            'created_at',  # Index for sorting by creation date
            'last_login',  # Index for last login queries
            ('username', 'created_at'),  # Compound index
            ('disabled', 'created_at'),  # Index for active user queries
        ]
    }
    
    id = StringField(primary_key=True, required=True, default=lambda: f"USER-{uuid.uuid4()}")
    username = StringField(required=True, unique=True, max_length=50)
    password_hash = StringField(required=True)
    password_salt = StringField(required=True)
    created_by = StringField(default="system")  # "system", "ldap", "oidc", "saml", etc.
    disabled = BooleanField(default=False)
    last_login = DateTimeField()
    roles = ListField(ReferenceField("Role"), reverse_delete_rule=PULL) # type: ignore
    api_keys = ListField(ReferenceField("ApiKey"))
    
    # Timestamps
    created_at = DateTimeField(default=lambda: datetime.now(timezone.utc), required=True)
    updated_at = DateTimeField(default=lambda: datetime.now(timezone.utc), required=True)

    def save(self, *args, **kwargs):
        """Override save to update the updated_at timestamp."""
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc)  # type: ignore
        self.updated_at = datetime.now(timezone.utc)  # type: ignore
        return super().save(*args, **kwargs)
    
    def __str__(self):
        return f"User(id={self.id}, username={self.username})"
    
    def __repr__(self):
        return self.__str__()

    def set_password(self, password: str) -> None:
        """Generate a bcrypt salt + hash and store them on the document."""
        salt: bytes = bcrypt.gensalt()
        self.password_salt = salt.decode()  # type: ignore
        self.password_hash = bcrypt.hashpw(password.encode(), salt).decode()  # type: ignore

    def verify_password(self, password: str) -> bool:
        """Return *True* if *password* matches the stored hash."""
        salt_str = str(self.password_salt)
        hashed_bytes: bytes = bcrypt.hashpw(password.encode(), salt_str.encode())
        hashed: str = hashed_bytes.decode()
        return hashed == str(self.password_hash)
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has a specific role."""
        if not self.roles:
            return False
        # Convert ListField to list for iteration
        roles_list = list(self.roles)  # type: ignore
        return any(role.rolename == role_name for role in roles_list if role)
    
    def get_role_names(self) -> list[str]:
        """Get list of role names for this user."""
        if not self.roles:
            return []
        # Convert ListField to list for iteration
        roles_list = list(self.roles)  # type: ignore
        return [role.rolename for role in roles_list if role]
    
    def has_access(self, request_endpoint: Endpoint) -> bool:
        """
        Checks if this user has access to a specific endpoint.

        Args:
            request_endpoint: An Endpoint object representing the incoming request.
                              The `path_filter` field should contain the actual request path string.

        Returns:
            True if access is allowed, False otherwise.
        """
        for role in self.roles:
            if role.has_access(request_endpoint):
                return True
        return False


def create_example_user():
    """Example function to create a user with roles and test functionality."""
    from app.db.mongo.role import Role, Method
    
    print("Creating example user with roles and testing functionality...\n")
    
    # Create example roles first
    admin_role = Role(
        rolename="admin",
        api_endpoints=[
            Endpoint(method=Method.GET, path_filter="/api/.*"),
            Endpoint(method=Method.POST, path_filter="/api/.*"),
            Endpoint(method=Method.PUT, path_filter="/api/.*"),
            Endpoint(method=Method.DELETE, path_filter="/api/.*"),
        ]
    )
    
    user_role = Role(
        rolename="user", 
        api_endpoints=[
            Endpoint(method=Method.GET, path_filter="/api/users/profile"),
            Endpoint(method=Method.PUT, path_filter="/api/users/profile"),
        ]
    )
    
    # Create example user
    test_user = User(
        username="testuser",
        created_by="system",
        disabled=False
    )
    
    # Set password
    test_user.set_password("SecurePassword123!")
    print(f"Created user: {test_user}")
    print(f"Password hash: {str(test_user.password_hash)[:20]}...")
    print(f"Password salt: {str(test_user.password_salt)[:20]}...")
    
    # Test password verification
    print(f"\nPassword verification tests:")
    print(f"Correct password: {test_user.verify_password('SecurePassword123!')}")
    print(f"Wrong password: {test_user.verify_password('wrongpassword')}")
    
    # Assign roles
    test_user.roles.extend([admin_role, user_role])  # type: ignore
    print(f"\nAssigned roles: {test_user.get_role_names()}")
    
    # Test role checking
    print(f"\nRole checking tests:")
    print(f"Has 'admin' role: {test_user.has_role('admin')}")
    print(f"Has 'user' role: {test_user.has_role('user')}")
    print(f"Has 'guest' role: {test_user.has_role('guest')}")
    
    # Test access control
    print(f"\nAccess control tests:")
    test_cases = {
        "ALLOW: GET /api/users/profile (user role)": (
            Endpoint(method=Method.GET, path_filter="/api/users/profile"), True
        ),
        "DENY: DELETE /api/users/profile (user role)": (
            Endpoint(method=Method.DELETE, path_filter="/api/users/profile"), True # from admin_role DELETE /api/*
        ),
        "DENY: DELETE /api2/roles/some-id (admin role)": (
            Endpoint(method=Method.DELETE, path_filter="/api2/roles/some-id"), False
        ),
        "DENY: POST /api2/restricted (no matching role)": (
            Endpoint(method=Method.POST, path_filter="/api2/restricted"), False
        ),
        "DENY: PUT /api2/anything (admin role wildcard)": (
            Endpoint(method=Method.PUT, path_filter="/api2/anything"), False
        ),
    }
    
    all_tests_passed = True
    for description, (test_endpoint, expected) in test_cases.items():
        actual = test_user.has_access(test_endpoint)
        result = "PASSED" if actual is expected else "FAILED"
        if actual is not expected:
            all_tests_passed = False
        print(f"Test: {description}")
        print(f"  -> Result: {actual} (Expected: {expected}) - {result}\n")
    
    if all_tests_passed:
        print("All user access control tests passed successfully!")
    else:
        print("Some user access control tests failed.")
    
    # Test user properties
    print(f"\nUser properties:")
    print(f"ID: {test_user.id}")
    print(f"Username: {test_user.username}")
    print(f"Created by: {test_user.created_by}")
    print(f"Disabled: {test_user.disabled}")
    print(f"Created at: {test_user.created_at}")
    print(f"Updated at: {test_user.updated_at}")
    
    return test_user


if __name__ == "__main__":
    import asyncio
    # Note: The original code had dependencies on local files (e.g., app.db.mongo.connect)
    # I've mocked the database connection to make this script runnable standalone.
    
    print("=== User Model Demo ===")
    create_example_user()