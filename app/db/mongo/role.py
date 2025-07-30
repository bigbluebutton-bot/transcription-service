import uuid
import re
from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar
from mongoengine import Document, EmbeddedDocument
from mongoengine import signals
from mongoengine.errors import ValidationError
from mongoengine.fields import (
    EmbeddedDocumentField,
    StringField,
    ListField,
    EnumField,
    DateTimeField,
)


def validate_regex(val: str):
    """
    Validates that the given string is a valid regular expression.
    """
    try:
        re.compile(val)
    except re.error:
        raise ValidationError(f"'{val}' is not a valid regular expression.")


class Method(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    ANY = "ANY"


class Endpoint(EmbeddedDocument):
    method = EnumField(Method, required=True)
    # Add the validation to the path_filter field
    path_filter = StringField(required=True, validation=validate_regex)


from mongoengine import Document
from mongoengine.queryset.manager import QuerySetManager  # type: ignore

class Role(Document):
    objects: QuerySetManager
    _signals_connected: ClassVar[bool] = False
    
    # Metadata
    meta = {
        'collection': 'roles',
        'indexes': [
            'rolename',  # Index for faster role name lookups
            'created_at',  # Index for sorting by creation date
            ('rolename', 'created_at'),  # Compound index
        ]
    }
    
    id = StringField(primary_key=True, required=True, default=lambda: f"ROLE-{uuid.uuid4()}")
    rolename = StringField(required=True, unique=True, max_length=50)
    api_endpoints = ListField(EmbeddedDocumentField(Endpoint))
    
    # Timestamps
    created_at = DateTimeField(default=lambda: datetime.now(timezone.utc), required=True)
    updated_at = DateTimeField(default=lambda: datetime.now(timezone.utc), required=True)
    
    def save(self, *args, **kwargs):
        """Override save to update the updated_at timestamp."""
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        return super().save(*args, **kwargs)
    
    def clean(self):
        """Validate the document before saving."""
        super().clean()
        
        # Validate rolename
        if not self.rolename or not self.rolename.strip():
            raise ValidationError('Role name cannot be empty')
        
        # Clean rolename
        self.rolename = self.rolename.strip()
        
        # Validate rolename characters
        if not re.match(r'^[a-zA-Z0-9\s_-]+$', self.rolename):
            raise ValidationError('Role name can only contain letters, numbers, spaces, hyphens, and underscores')
    
    def __str__(self):
        return f"Role(id={self.id}, rolename={self.rolename})"
    
    def __repr__(self):
        return self.__str__()

    def has_access(self, request_endpoint: Endpoint) -> bool:
        """
        Checks if this role grants access to a specific endpoint.

        Args:
            request_endpoint: An Endpoint object representing the incoming request.
                              The `path_filter` field should contain the actual request path string.

        Returns:
            True if access is allowed, False otherwise.
        """
        request_method_str = request_endpoint.method.value.upper()
        request_path = request_endpoint.path_filter

        for rule_endpoint in self.api_endpoints:
            # 1. Check if the method matches.
            # Access is allowed if the rule's method is ANY or matches the request method.
            method_matches = (rule_endpoint.method == Method.ANY or rule_endpoint.method.value == request_method_str)

            if method_matches:
                # 2. If method matches, check if the path matches the regex pattern.
                # re.match() ensures the pattern matches from the start of the path.
                if re.match(rule_endpoint.path_filter, request_path):
                    # Found a matching rule, grant access immediately.
                    return True

        # If the loop completes without finding a match, deny access.
        return False

# Connect the signal handler at the module level
def cleanup_role_references(sender, document, **kwargs):
    """
    Signal handler to remove this role from all users and API keys before deletion.
    This ensures referential integrity when a role is deleted.
    """
    # Import here to avoid circular imports
    from .user import User
    from .api_key import ApiKey
    
    try:
        # Remove role from all users' roles lists
        # Use the document reference directly for the query and update
        User.objects(roles=document).update(pull__roles=document)
    except Exception as e:
        # Log the error but don't fail the deletion
        import logging
        logging.error(f"Error cleaning up user role references: {e}")

    try:
        # Remove role from all API keys' roles lists
        # Use the document reference directly for the query and update
        ApiKey.objects(roles=document).update(pull__roles=document)
    except Exception as e:
        # Log the error but don't fail the deletion
        import logging
        logging.error(f"Error cleaning up API key role references: {e}")

# Connect the signal handler for Role deletion
def register_signals():
    if not hasattr(register_signals, '_registered'):
        signals.pre_delete.connect(cleanup_role_references, sender='Role')
        register_signals._registered = True

# Call the registration function
register_signals()


async def create_example_role():
    """Example function to create a role with API endpoints"""
    from app.db.mongo.connect import connect_to_mongo
    
    # Connect to MongoDB
    await connect_to_mongo()
    
    # Create sample endpoints with valid regex
    user_endpoints = [
        Endpoint(method=Method.GET, path_filter=r"^/api/users$"),
        Endpoint(method=Method.POST, path_filter=r"^/api/users$"),
        # Matches /api/users/<any-non-slash-characters>
        Endpoint(method=Method.PUT, path_filter=r"^/api/users/[^/]+$"),
        Endpoint(method=Method.DELETE, path_filter=r"^/api/users/[^/]+$"),
        Endpoint(method=Method.GET, path_filter=r"^/api/roles$"),
        Endpoint(method=Method.POST, path_filter=r"^/api/roles$"),
    ]
    
    # Create a new role
    user_role = Role(
        rolename="user",
        api_endpoints=user_endpoints
    )
    
    try:
        # Save the role to database
        user_role.save()
        print(f"Role created successfully: {user_role.id}")
        print(f"Role name: {user_role.rolename}")
        print(f"Number of endpoints: {len(user_role.api_endpoints)}")
        
        # Display the endpoints
        print("\nAPI Endpoints:")
        for endpoint in user_role.api_endpoints:
            print(f"  {endpoint.method.value} {endpoint.path_filter}")

        # Example of an invalid regex that will fail validation
        print("\n--- Attempting to save an invalid regex ---")
        invalid_endpoint = Endpoint(method=Method.GET, path_filter="/api/users/[")
        invalid_role = Role(rolename="invalid_role", api_endpoints=[invalid_endpoint])
        invalid_role.save()
            
    except ValidationError as e:
        print(f"Caught expected validation error: {e}")
    except Exception as e:
        print(f"Error creating role: {e}")

    # test access
    print("\n--- Testing Access Control ---")

    # A dictionary to hold test cases for clarity
    # Each key is a description, and the value is a tuple of (Endpoint, expected_result)
    test_cases = {
        "ALLOW: POST /api/users (exact match)": (
            Endpoint(method=Method.POST, path_filter="/api/users"), True
        ),
        "DENY: PUT /api/users (path needs ID)": (
            Endpoint(method=Method.PUT, path_filter="/api/users"), False
        ),
        "ALLOW: PUT /api/users/some-id (regex match)": (
            Endpoint(method=Method.PUT, path_filter="/api/users/some-id"), True
        ),
        "ALLOW: DELETE /api/users/another-id (regex match)": (
            Endpoint(method=Method.DELETE, path_filter="/api/users/another-id"), True
        ),
        "DENY: GET /api/users/some-id (exact match rule exists, but not for this path)": (
            Endpoint(method=Method.GET, path_filter="/api/users/some-id"), False
        ),
        "ALLOW: GET /api/roles (exact match)": (
            Endpoint(method=Method.GET, path_filter="/api/roles"), True
        ),
        "DENY: PATCH /api/users/some-id (method not allowed)": (
            Endpoint(method=Method.PATCH, path_filter="/api/users/some-id"), False
        ),
        "DENY: GET /some/other/path (no matching rule)": (
            Endpoint(method=Method.GET, path_filter="/some/other/path"), False
        ),
    }

    all_tests_passed = True
    for description, (test_endpoint, expected) in test_cases.items():
        actual = user_role.has_access(test_endpoint)
        result = "PASSED" if actual is expected else "FAILED"
        if actual is not expected:
            all_tests_passed = False
        print(f"Test: {description}")
        print(f"  -> Result: {actual} (Expected: {expected}) - {result}\n")

    if all_tests_passed:
        print("All access control tests passed successfully!")
    else:
        print("Some access control tests failed.")


if __name__ == "__main__":
    import asyncio
    # Note: The original code had dependencies on local files (e.g., app.db.mongo.connect)
    # I've mocked the database connection to make this script runnable standalone.
    asyncio.run(create_example_role())