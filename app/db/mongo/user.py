from __future__ import annotations
import uuid
"""MongoEngine document model for application users with full static typing support.

This version is mypy-clean when used together with the ``mongo-types`` stub
package for MongoEngine (``pip install mongo-types``) and modern ``bcrypt``
(>=3.2, which ships its own type information).
"""

from datetime import datetime

import bcrypt
from mongoengine import Document
PULL = 4
from mongoengine.fields import (
    StringField,
    BooleanField,
    DateTimeField,
    ListField,
    ReferenceField,
)



class User(Document):
    """Application user stored in the ``users`` MongoDB collection."""
    id = StringField(primary_key=True, required=True, default=lambda: f"USER-{uuid.uuid4()}")
    username = StringField(required=True, unique=True)
    password_hash = StringField(required=True)
    password_salt = StringField(required=True)
    created_by = StringField(default="system")  # "system", "ldap", "oidc", "saml", etc.
    disabled = BooleanField(default=False)
    last_login = DateTimeField()
    roles = ListField(ReferenceField("Role"), reverse_delete_rule=PULL) # type: ignore
    api_keys = ListField(ReferenceField("ApiKey"))

    meta = {
        "collection": "users",
        "indexes": ["username"],
    }

    # ---------------------------------------------------------------------
    # Public helpers -------------------------------------------------------
    # ---------------------------------------------------------------------

    def set_password(self, password: str) -> None:
        """Generate a bcrypt salt + hash and store them on the document."""
        salt: bytes = bcrypt.gensalt()
        self.password_salt = salt.decode()
        self.password_hash = bcrypt.hashpw(password.encode(), salt).decode()

    def verify_password(self, password: str) -> bool:
        """Return *True* if *password* matches the stored hash."""
        hashed_bytes: bytes = bcrypt.hashpw(password.encode(), self.password_salt.encode())
        hashed: str = hashed_bytes.decode()
        return hashed == self.password_hash