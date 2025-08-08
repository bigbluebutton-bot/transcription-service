from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, ClassVar
import bcrypt
import base64
from mongoengine import Document
from mongoengine.fields import (
    EmbeddedDocumentField,
    StringField,
    ListField,
    EnumField,
    DateTimeField,
    BooleanField
)

if TYPE_CHECKING:
    from app.db.mongo.user import User


class ApiKey(Document):
    """
    API Key model for authenticating users.

    Attributes:
        key (str): The unique API key, which is the primary key.
        name (str): A user-defined name for the key for easy identification.
        user (User): A reference to the user who owns the key.
        is_active (bool): A flag to enable or disable the key.
        created_at (datetime): The timestamp when the key was created.
        updated_at (datetime): The timestamp when the key was last updated.
        last_used (datetime): The timestamp when the key was last used.
    """

    meta: ClassVar[dict[str, str | list[str | tuple[str, str]]]] = {
        'collection': 'api_keys',
        'indexes': [
            'user',
            'created_at',
            ('user', 'name'),  # Compound index for user-specific key names
        ],
    }

    key = StringField(primary_key=True, default=lambda: str(uuid.uuid4()))
    name = StringField(required=True, min_length=3, max_length=50)
    user: User = ReferenceField('User', required=True, reverse_delete_rule=Document.CASCADE)  # type: ignore[assignment]
    is_active = BooleanField(default=True)
    disabled = BooleanField(default=False)

    created_at = DateTimeField(default=lambda: datetime.now(timezone.utc))
    updated_at = DateTimeField(default=lambda: datetime.now(timezone.utc))
    last_used = DateTimeField(null=True)

    def save(self, *args, **kwargs) -> ApiKey:
        """Override save to update timestamps and run validation."""
        self.updated_at = datetime.now(timezone.utc)
        return super().save(*args, **kwargs)

    def clean(self) -> None:
        """Validate and clean the data before saving."""
        if self.name:
            self.name = self.nastrip()

    def __str__(self) -> str:
        return f'ApiKey(name={self.name}, user={self.user.username if self.user else "N/A"})'

    def __repr__(self) -> str:
        return f'<ApiKey key={self.key} name={self.name}>'

    @staticmethod
    def hash_key(key: str) -> str:
        """
        Deterministically hash the key using PBKDF2 with a salt derived from the key.
        """
        base64_salt = base64.b64encode(key.encode('utf-8')).decode('utf-8')
        salt = ("key_salt" + base64_salt + "key_salt").encode()
        hashed_bytes: bytes = bcrypt.hashpw(key.encode(), salt)
        hashed: str = hashed_bytes.decode()
        return hashed

    def verify_key(self, key: str) -> bool:
        """
        Verify the key against the stored hash.
        """
        return self.hash_key(key) == self.key_hash


    def is_valid(self) -> bool:
        """
        Check if the API key is expired or disabled. Returns true if the key is valid.
        """
        if self.disabled:
            return False
        if self.expiration is None:
            return True
        return datetime.now(timezone.utc) < self.expiration

# Connect the signal handler at the module level
def cleanup_apikey_references(sender, document, **kwargs):
    """
    Signal handler to remove this API key from the user's api_keys list before deletion.
    This ensures referential integrity when an API key is deleted.
    """
    if document.user and document in document.user.api_keys:
        # Use atomic operation to remove the reference
        from .user import User
        User.objects(id=document.user.id).update_one(pull__api_keys=document)

# Connect the signal handler for ApiKey deletion
def register_signals():
    if not hasattr(register_signals, '_registered'):
        signals.pre_delete.connect(cleanup_apikey_references, sender='ApiKey')
        register_signals._registered = True

# Call the registration function
register_signals()