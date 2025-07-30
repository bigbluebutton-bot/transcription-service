import uuid
from enum import Enum
from typing import ClassVar
from mongoengine import Document, EmbeddedDocument
from mongoengine import signals
from mongoengine.fields import (
    EmbeddedDocumentField,
    StringField,
    BooleanField,
    DateTimeField,
    ListField,
    ReferenceField,
    EnumField,
)


class Method(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    ANY = "ANY"


class Endpoint(EmbeddedDocument):
    method = EnumField(Method, required=True)
    path_filter = StringField(required=True)


from mongoengine import Document
from mongoengine.queryset.manager import QuerySetManager  # type: ignore

class Role(Document):
    objects: QuerySetManager
    _signals_connected: ClassVar[bool] = False
    
    id = StringField(primary_key=True, required=True, default=lambda: f"ROLE-{uuid.uuid4()}")
    rolename = StringField(required=True, unique=True)
    api_endpoints = ListField(EmbeddedDocumentField(Endpoint))

    meta = {
        'collection': 'roles',
        'indexes': [
            'rolename',
        ]
    }

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
