from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import json

# For Redis Cluster, you need the cluster-specific client
from redis.asyncio.cluster import RedisCluster, ClusterNode

# --- Mock objects for demonstration ---
# In your actual application, you would import these from your models/config files.
class MockMongoObject:
    _id_counter = 0
    def __init__(self, **kwargs):
        self.id = f"mock_id_{MockMongoObject._id_counter}"
        MockMongoObject._id_counter += 1
        self.username = "test_user"
        self.roles = []
        self.last_login = None
        self.key_hash = "mock_hash"
        self.expiration = None
        self.user = self
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def save(self): pass
    def verify_password(self, password): return password == "password123"
    @classmethod
    def hash_key(cls, key): return "mock_hash"
    @classmethod
    def objects(cls, **kwargs):
        class MockQuerySet:
            def first(self): return MockMongoObject(**kwargs)
            def delete(self): pass
        return MockQuerySet()

class ApiKey(MockMongoObject): pass
class User(MockMongoObject): pass
class Role(MockMongoObject): pass

@dataclass
class Config:
    NODE_ID: str = "node-1"
    WEBRTC_TIMEOUT: int = 3600
# --- End Mock objects ---


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Constants & helpers with HASH TAGS for Redis Cluster
# --------------------------------------------------------------------------- #
SESSION_KEY_PREFIX = "session"
SESSION_MAP_PREFIX = "session_map"

# Redis key format templates with hash tags {session_id}
# This ensures all keys related to a single session are on the same node.
USER_SESSION_KEY_TPL = f"{SESSION_KEY_PREFIX}:{{user_id}}:sessions:{{{{session_id}}}}:data"
APIKEY_SESSION_KEY_TPL = f"{SESSION_KEY_PREFIX}:{{user_id}}:api_keys:{{apikey_id}}:sessions:{{{{session_id}}}}:data"
WEBRTC_SESSION_KEY_TPL = f"{SESSION_KEY_PREFIX}:{{user_id}}:{{parent_type}}:{{{{session_id}}}}:node:{{node_id}}:webrtc:{{webrtc_id}}:data"
WS_SESSION_KEY_TPL = f"{SESSION_KEY_PREFIX}:{{user_id}}:{{parent_type}}:{{{{session_id}}}}:node:{{node_id}}:ws:{{ws_id}}:data"
SESSION_MAP_KEY_TPL = f"{SESSION_MAP_PREFIX}:{{{{session_id}}}}"

CONFIG = Config()

def _map_session_type_to_key_segment(session_type: str) -> str:
    """Maps a session type ('user', 'api_key') to its Redis key segment."""
    if session_type == "user":
        return "sessions"
    if session_type == "api_key":
        return "api_keys"
    if session_type == "*":
        return "*"
    raise ValueError("session_type must be 'user', 'api_key', or '*'")

def build_user_session_key(user_id: str, session_id: str) -> str:
    return USER_SESSION_KEY_TPL.format(user_id=user_id, session_id=session_id)

def build_apikey_session_key(user_id: str, apikey_id: str, session_id: str) -> str:
    return APIKEY_SESSION_KEY_TPL.format(user_id=user_id, apikey_id=apikey_id, session_id=session_id)

def build_webrtc_session_key(user_id: str, parent_session_type: str, session_id: str, webrtc_id: str, node_id: str = "*") -> str:
    return WEBRTC_SESSION_KEY_TPL.format(
        user_id=user_id,
        parent_type=_map_session_type_to_key_segment(parent_session_type),
        session_id=session_id,
        node_id=node_id,
        webrtc_id=webrtc_id
    )

def build_ws_session_key(user_id: str, parent_session_type: str, session_id: str, ws_id: str, node_id: str = "*") -> str:
    return WS_SESSION_KEY_TPL.format(
        user_id=user_id,
        parent_type=_map_session_type_to_key_segment(parent_session_type),
        session_id=session_id,
        node_id=node_id,
        ws_id=ws_id
    )

def build_logout_scan_pattern(user_id: str = "*", session_type: str = "*", session_id: str = "*") -> str:
    """Builds a pattern for scanning and deleting multiple sessions."""
    return f"{SESSION_KEY_PREFIX}:{user_id}:{_map_session_type_to_key_segment(session_type)}:{session_id}:*"

# Helper converters
def datetime_to_str(dt: datetime) -> str:
    return dt.isoformat()

def str_to_datetime(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str)

# --------------------------------------------------------------------------- #
#  Session dataclasses
# --------------------------------------------------------------------------- #
class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

@dataclass
class Session(ABC):
    expiration_date: datetime
    creation_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str = ""
    _id: str = field(default_factory=lambda: f"SESSION-{uuid.uuid4()}")

    @property
    def id(self) -> str:
        return self._id

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_json(cls, json_str: str) -> "Session":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        # Common deserialization logic is handled here.
        data['creation_date'] = str_to_datetime(data['creation_date'])
        data['expiration_date'] = str_to_datetime(data['expiration_date'])
        
        # Filter data to only include fields that exist in the specific subclass
        cls_fields = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in cls_fields}
        return cls(**filtered_data)

    async def logout(self) -> None:
        await SessionManager().logout(self._id)

@dataclass
class SessionUser(Session):
    _id: str = field(default_factory=lambda: f"SESSION-USER-{uuid.uuid4()}")

    def to_dict(self) -> dict:
        return {
            "_id": self._id,
            "creation_date": datetime_to_str(self.creation_date),
            "expiration_date": datetime_to_str(self.expiration_date),
            "user_id": self.user_id,
        }

@dataclass
class SessionAPIKey(Session):
    _id: str = field(default_factory=lambda: f"SESSION-API-{uuid.uuid4()}")
    apikey_id: str = ""

    def to_dict(self) -> dict:
        return {
            "_id": self._id,
            "creation_date": datetime_to_str(self.creation_date),
            "expiration_date": datetime_to_str(self.expiration_date),
            "apikey_id": self.apikey_id,
            "user_id": self.user_id,
        }

@dataclass
class SessionWebRTC(Session):
    _id: str = field(default_factory=lambda: f"SESSION-WEBRTC-{uuid.uuid4()}")
    parent_session_id: str = ""

    def to_dict(self) -> dict:
        return {
            "_id": self._id,
            "creation_date": datetime_to_str(self.creation_date),
            "expiration_date": datetime_to_str(self.expiration_date),
            "user_id": self.user_id,
            "parent_session_id": self.parent_session_id,
        }

@dataclass
class SessionWS(Session):
    _id: str = field(default_factory=lambda: f"SESSION-WS-{uuid.uuid4()}")
    parent_session_id: str = ""

    def to_dict(self) -> dict:
        return {
            "_id": self._id,
            "creation_date": datetime_to_str(self.creation_date),
            "expiration_date": datetime_to_str(self.expiration_date),
            "user_id": self.user_id,
            "parent_session_id": self.parent_session_id,
        }

# --------------------------------------------------------------------------- #
#  SessionManager for Redis Cluster
# --------------------------------------------------------------------------- #
class SessionManager:
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "SessionManager":
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, startup_nodes: Optional[List[ClusterNode]] = None, session_duration: int = 86400) -> None:
        if not hasattr(self, 'redis'):
            if startup_nodes is None:
                # Provide a default for local testing
                startup_nodes = [ClusterNode("localhost", 7000), ClusterNode("localhost", 7001)]
            
            self.redis = RedisCluster(startup_nodes=startup_nodes, decode_responses=True)
            self.session_duration = session_duration

    async def _store_session(self, session: Session, key: str, ttl: int):
        """Stores session data and the session_id->key mapping. This now works on a cluster."""
        map_key = SESSION_MAP_KEY_TPL.format(session_id=session.id)
        # This pipeline is now CLUSTER SAFE because both keys share the same hash tag.
        async with self.redis.pipeline() as pipe:
            pipe.set(key, session.to_json(), ex=ttl)
            pipe.set(map_key, key, ex=ttl)
            await pipe.execute()

    async def login(self, username: str, password: str) -> Tuple[SessionUser, User]:
        user = User.objects(username=username).first()
        if not user or not user.verify_password(password):
            raise ValueError("Invalid username or password")

        user.last_login = datetime.now(timezone.utc)
        user.save()

        session = SessionUser(
            user_id=str(user.id),
            expiration_date=datetime.now(timezone.utc) + timedelta(seconds=self.session_duration),
        )
        key = build_user_session_key(str(user.id), session.id)
        await self._store_session(session, key, self.session_duration)
        return session, user

    async def login_apikey(self, apikey: str) -> Tuple[SessionAPIKey, ApiKey]:
        key_hash = ApiKey.hash_key(apikey)
        apikey_obj = ApiKey.objects(key_hash=key_hash).first()
        if not apikey_obj or (apikey_obj.expiration and apikey_obj.expiration < datetime.now(timezone.utc)):
            raise ValueError("API key not found or expired")
        
        user = apikey_obj.user
        if not user:
            raise ValueError("Owner of API key not found")

        session = SessionAPIKey(
            apikey_id=str(apikey_obj.id),
            user_id=str(user.id),
            expiration_date=datetime.now(timezone.utc) + timedelta(seconds=self.session_duration),
        )
        key = build_apikey_session_key(str(user.id), str(apikey_obj.id), session.id)
        await self._store_session(session, key, self.session_duration)
        return session, apikey_obj

    async def login_webrtc(self, session: Union[SessionUser, SessionAPIKey]) -> SessionWebRTC:
        webrtc_session = SessionWebRTC(
            user_id=session.user_id,
            parent_session_id=session.id,
            expiration_date=session.expiration_date,
        )
        key = build_webrtc_session_key(
            user_id=session.user_id,
            parent_session_type="user" if isinstance(session, SessionUser) else "api_key",
            session_id=session.id,
            webrtc_id=webrtc_session.id,
            node_id=CONFIG.NODE_ID
        )
        await self._store_session(webrtc_session, key, CONFIG.WEBRTC_TIMEOUT * 2)
        return webrtc_session

    async def login_ws(self, session: Union[SessionUser, SessionAPIKey]) -> SessionWS:
        ws_session = SessionWS(
            user_id=session.user_id,
            parent_session_id=session.id,
            expiration_date=session.expiration_date,
        )
        key = build_ws_session_key(
            user_id=session.user_id,
            parent_session_type="user" if isinstance(session, SessionUser) else "api_key",
            session_id=session.id,
            ws_id=ws_session.id,
            node_id=CONFIG.NODE_ID
        )
        await self._store_session(ws_session, key, CONFIG.WEBRTC_TIMEOUT * 2)
        return ws_session

    async def _delete_keys_by_pattern(self, pattern: str):
        """Scans for keys by pattern and deletes them, including their map entries."""
        keys_to_delete = []
        map_keys_to_delete = []
        async for key in self.redis.scan_iter(match=pattern):
            keys_to_delete.append(key)
            # Extract session ID from key to delete corresponding map entry
            try:
                # For keys with hash tags, the session ID is within the curly braces
                if "{" in key and "}" in key:
                    start = key.find("{") + 1
                    end = key.find("}", start)
                    session_id = key[start:end]
                    if session_id.startswith("SESSION-"):
                        map_keys_to_delete.append(SESSION_MAP_KEY_TPL.format(session_id=session_id))
            except (IndexError, ValueError):
                logger.warning(f"Could not parse session_id from key: {key}")

        if keys_to_delete:
            # Note: This may fail in cluster mode if keys are on different nodes
            # For production, you'd want to batch delete by node
            try:
                await self.redis.delete(*keys_to_delete)
            except Exception as e:
                logger.warning(f"Batch delete failed, falling back to individual deletes: {e}")
                for key in keys_to_delete:
                    try:
                        await self.redis.delete(key)
                    except Exception as delete_error:
                        logger.error(f"Failed to delete key {key}: {delete_error}")
        
        if map_keys_to_delete:
            try:
                await self.redis.delete(*map_keys_to_delete)
            except Exception as e:
                logger.warning(f"Batch delete of map keys failed, falling back to individual deletes: {e}")
                for map_key in map_keys_to_delete:
                    try:
                        await self.redis.delete(map_key)
                    except Exception as delete_error:
                        logger.error(f"Failed to delete map key {map_key}: {delete_error}")
        
        return len(keys_to_delete)

    async def logout(self, session_id: str) -> None:
        map_key = SESSION_MAP_KEY_TPL.format(session_id=session_id)
        full_key = await self.redis.get(map_key)
        if full_key:
            # This multi-key delete is CLUSTER SAFE because both keys share the same hash tag.
            await self.redis.delete(full_key, map_key)

    async def logout_user(self, user_id: str) -> None:
        pattern = build_logout_scan_pattern(user_id=user_id, session_type="user")
        await self._delete_keys_by_pattern(pattern)

    async def logout_apikey(self, user_id: str) -> None:
        pattern = build_logout_scan_pattern(user_id=user_id, session_type="api_key")
        await self._delete_keys_by_pattern(pattern)

    async def logout_all(self) -> None:
        # This is a destructive operation, be careful.
        # Deletes all session data and all map keys.
        await self._delete_keys_by_pattern(f"{SESSION_KEY_PREFIX}:*")
        await self._delete_keys_by_pattern(f"{SESSION_MAP_PREFIX}:*")

    async def set_ttl(self, session_id: str, new_ttl: int) -> None:
        map_key = SESSION_MAP_KEY_TPL.format(session_id=session_id)
        full_key = await self.redis.get(map_key)
        if not full_key:
            logger.error(f"Unable to find session key for session_id: {session_id}")
            return
        
        # This pipeline is CLUSTER SAFE because both keys share the same hash tag.
        async with self.redis.pipeline() as pipe:
            pipe.expire(full_key, new_ttl)
            pipe.expire(map_key, new_ttl)
            await pipe.execute()
        logger.debug(f"Set ttl of redis key {full_key} to {new_ttl}")

    async def exists(self, session_id: str) -> bool:
        map_key = SESSION_MAP_KEY_TPL.format(session_id=session_id)
        return await self.redis.exists(map_key) > 0

    async def get_session(self, session_id: str) -> Optional[Session]:
        map_key = SESSION_MAP_KEY_TPL.format(session_id=session_id)
        full_key = await self.redis.get(map_key)
        if not full_key:
            return None

        value = await self.redis.get(full_key)
        if not value:
            return None

        # The key string determines which Session subclass to return.
        if ":webrtc:" in full_key:
            return SessionWebRTC.from_json(value)
        if ":ws:" in full_key:
            return SessionWS.from_json(value)
        if ":api_keys:" in full_key:
            return SessionAPIKey.from_json(value)
        return SessionUser.from_json(value)

    async def get_all_sessions(self) -> Dict[str, Session]:
        """
        Return **all** sessions in Redis.
        NOTE: This uses SCAN and can be slow on databases with many keys.
        """
        sessions: Dict[str, Session] = {}
        async for key in self.redis.scan_iter(match=f"{SESSION_KEY_PREFIX}:*:*:*:data"):
            value = await self.redis.get(key)
            if not value:
                continue
            
            session_obj = None
            if ":webrtc:" in key: 
                session_obj = SessionWebRTC.from_json(value)
            elif ":ws:" in key: 
                session_obj = SessionWS.from_json(value)
            elif ":api_keys:" in key: 
                session_obj = SessionAPIKey.from_json(value)
            elif ":sessions:" in key: 
                session_obj = SessionUser.from_json(value)

            if session_obj:
                sessions[session_obj.id] = session_obj
        return sessions

    async def get_sessions_by_user(self, user_id: str) -> List[SessionUser]:
        pattern = build_user_session_key(user_id=user_id, session_id="*")
        sessions: List[SessionUser] = []
        async for key in self.redis.scan_iter(match=pattern):
            raw = await self.redis.get(key)
            if raw:
                sessions.append(SessionUser.from_json(raw))
        return sessions

    async def get_sessions_by_apikey(self, apikey_id: str) -> List[SessionAPIKey]:
        pattern = build_apikey_session_key(user_id="*", apikey_id=apikey_id, session_id="*")
        sessions: List[SessionAPIKey] = []
        async for key in self.redis.scan_iter(match=pattern):
            raw = await self.redis.get(key)
            if raw:
                sessions.append(SessionAPIKey.from_json(raw))
        return sessions

    async def get_webrtc_sessions(self, user_id: str = "*", user_or_api_session_type: str = "*", session_id: str = "*", webrtc_id: str = "*", node_id: str = "*") -> List[SessionWebRTC]:
        sessions: List[SessionWebRTC] = []
        pattern = build_webrtc_session_key(
            user_id=user_id,
            parent_session_type=user_or_api_session_type,
            session_id=session_id,
            node_id=node_id,
            webrtc_id=webrtc_id
        )

        async for key in self.redis.scan_iter(match=pattern):
            raw = await self.redis.get(key)
            if raw:
                sessions.append(SessionWebRTC.from_json(raw))
        return sessions

    async def get_webrtc_sessions_from_session(self, session: Union[SessionUser, SessionAPIKey]) -> List[SessionWebRTC]:
        """
        Get all WebRTC sessions associated with a user or API key session.
        """
        if not isinstance(session, (SessionUser, SessionAPIKey)):
            raise TypeError(
                "get_webrtc_sessions_from_session expects a SessionUser or SessionAPIKey, "
                f"got {type(session).__name__}"
            )

        pattern = build_webrtc_session_key(
            user_id=session.user_id,
            parent_session_type="user" if isinstance(session, SessionUser) else "api_key",
            session_id=session.id,
            node_id="*",
            webrtc_id="*"
        )
        webrtc_sessions: List[SessionWebRTC] = []
        async for key in self.redis.scan_iter(match=pattern):
            raw = await self.redis.get(key)
            if raw:
                webrtc_sessions.append(SessionWebRTC.from_json(raw))
        return webrtc_sessions
        
    async def get_ws_session(self, session_id: str) -> Optional[SessionWS]:
        """
        Get a specific WebSocket session by ID.
        """
        session = await self.get_session(session_id)
        if isinstance(session, SessionWS):
            return session
        return None
        
    async def get_ws_sessions(self, user_id: str = "*", user_or_api_session_type: str = "*", session_id: str = "*", ws_id: str = "*", node_id: str = "*") -> List[SessionWS]:
        """
        Get WebSocket sessions matching the specified criteria.
        """
        sessions: List[SessionWS] = []
        pattern = build_ws_session_key(
            user_id=user_id,
            parent_session_type=user_or_api_session_type,
            session_id=session_id,
            node_id=node_id,
            ws_id=ws_id
        )

        async for key in self.redis.scan_iter(match=pattern):
            raw = await self.redis.get(key)
            if raw:
                sessions.append(SessionWS.from_json(raw))
        return sessions
        
    async def get_ws_sessions_from_session(self, session: Union[SessionUser, SessionAPIKey]) -> List[SessionWS]:
        """
        Get all WebSocket sessions associated with a user or API key session.
        """
        if not isinstance(session, (SessionUser, SessionAPIKey)):
            raise TypeError(
                "get_ws_sessions_from_session expects a SessionUser or SessionAPIKey, "
                f"got {type(session).__name__}"
            )

        pattern = build_ws_session_key(
            user_id=session.user_id,
            parent_session_type="user" if isinstance(session, SessionUser) else "api_key",
            session_id=session.id,
            node_id="*",
            ws_id="*"
        )
        ws_sessions: List[SessionWS] = []
        async for key in self.redis.scan_iter(match=pattern):
            raw = await self.redis.get(key)
            if raw:
                ws_sessions.append(SessionWS.from_json(raw))
        return ws_sessions

    async def close_connection(self):
        """Close the Redis cluster connection."""
        await self.redis.close()


# --------------------------------------------------------------------------- #
#  Example Usage
# --------------------------------------------------------------------------- #
async def main():
    """Demonstrates the usage of the optimized SessionManager with Redis Cluster."""
    print("--- Initializing Session Manager for Redis Cluster ---")
    
    # IMPORTANT: Replace these with the actual host/port of your cluster nodes
    startup_nodes = [
        ClusterNode("127.0.0.1", 7000),
        ClusterNode("127.0.0.1", 7001),
        ClusterNode("127.0.0.1", 7002),
    ]
    
    try:
        # Pass the startup nodes to the constructor
        session_manager = SessionManager(startup_nodes=startup_nodes)
        await session_manager.redis.ping()
        print("Successfully connected to Redis Cluster.")
    except Exception as e:
        print(f"Could not connect to Redis Cluster. Please ensure it's running. Error: {e}")
        print("Falling back to single Redis instance for demo...")
        # Fallback to single Redis instance for demo
        try:
            from redis.asyncio import Redis
            session_manager.redis = Redis.from_url("redis://localhost:6379/10", decode_responses=True)
            await session_manager.redis.ping()
            print("Connected to single Redis instance.")
        except Exception as fallback_error:
            print(f"Could not connect to Redis at all: {fallback_error}")
            return

    # Clean slate for the demo
    try:
        await session_manager.redis.flushdb()
        print("Cleaned Redis database for demo.")
    except Exception as e:
        print(f"Could not flush database: {e}")

    # --- 1. User Login ---
    print("\n--- 1. User Login ---")
    try:
        session_user, user_obj = await session_manager.login("test_user", "password123")
        print(f"Successfully logged in user '{user_obj.username}'.")
        print(f"Created User Session ID: {session_user.id}")
    except ValueError as e:
        print(f"Login failed: {e}")
        await session_manager.close_connection()
        return

    # --- 2. Verify Session Existence and Retrieve it ---
    print("\n--- 2. Verify and Retrieve Session ---")
    exists = await session_manager.exists(session_user.id)
    print(f"Does session {session_user.id} exist? {exists}")

    retrieved_session = await session_manager.get_session(session_user.id)
    if retrieved_session:
        print(f"Successfully retrieved session. Type: {type(retrieved_session).__name__}")
        print(f"Session User ID: {retrieved_session.user_id}")
    else:
        print("Failed to retrieve session.")

    # --- 3. API Key Login ---
    print("\n--- 3. API Key Login ---")
    try:
        # In a real app, the API key would be created and stored for a user.
        # We are mocking this process.
        session_api, apikey_obj = await session_manager.login_apikey("my-secret-api-key")
        print(f"Successfully logged in with API key for user '{apikey_obj.user.username}'.")
        print(f"Created API Key Session ID: {session_api.id}")
    except ValueError as e:
        print(f"API key login failed: {e}")
        session_api = None

    # --- 4. Create Child Sessions (WebRTC and WebSocket) ---
    print("\n--- 4. Create Child Sessions ---")
    webrtc_session = await session_manager.login_webrtc(session_user)
    ws_session = await session_manager.login_ws(session_user)
    print(f"Created WebRTC Session ID: {webrtc_session.id} (child of {session_user.id})")
    print(f"Created WebSocket Session ID: {ws_session.id} (child of {session_user.id})")

    # --- 5. List various sessions ---
    print("\n--- 5. List Sessions ---")
    all_sessions = await session_manager.get_all_sessions()
    print(f"Total active sessions found: {len(all_sessions)}")
    for sid, s_obj in all_sessions.items():
        print(f"  - ID: {sid}, Type: {type(s_obj).__name__}")

    user_sessions = await session_manager.get_sessions_by_user(user_obj.id)
    print(f"\nFound {len(user_sessions)} sessions for user_id '{user_obj.id}'.")

    child_webrtc_sessions = await session_manager.get_webrtc_sessions_from_session(session_user)
    print(f"Found {len(child_webrtc_sessions)} WebRTC child sessions for session '{session_user.id}'.")

    child_ws_sessions = await session_manager.get_ws_sessions_from_session(session_user)
    print(f"Found {len(child_ws_sessions)} WebSocket child sessions for session '{session_user.id}'.")

    # --- 6. Logout a single session ---
    print("\n--- 6. Logout a Single Session ---")
    if session_api:
        print(f"Logging out API key session: {session_api.id}")
        await session_manager.logout(session_api.id)
        exists = await session_manager.exists(session_api.id)
        print(f"Does API key session {session_api.id} still exist? {exists}")

    # --- 7. Set TTL for a session ---
    print("\n--- 7. Set TTL for a Session ---")
    print(f"Setting TTL of 60 seconds for WebRTC session: {webrtc_session.id}")
    await session_manager.set_ttl(webrtc_session.id, 60)
    print("TTL set successfully.")

    # --- 8. Logout all sessions for a user ---
    print("\n--- 8. Logout All Sessions for a User ---")
    print(f"Logging out all sessions for user_id: {user_obj.id}")
    await session_manager.logout_user(user_obj.id)
    
    user_sessions_after_logout = await session_manager.get_sessions_by_user(user_obj.id)
    print(f"Sessions found for user after mass logout: {len(user_sessions_after_logout)}")
    
    exists = await session_manager.exists(session_user.id)
    print(f"Does original user session {session_user.id} still exist? {exists}")

    # --- 9. Final Cleanup ---
    print("\n--- 9. Final Cleanup ---")
    try:
        await session_manager.redis.flushdb()
        print("Test Redis database flushed.")
    except Exception as e:
        print(f"Could not flush database: {e}")
    
    await session_manager.close_connection()
    print("Redis connection closed.")


if __name__ == "__main__":
    # To run this with a Redis Cluster, you need a Redis Cluster running locally.
    # A simple way is using Docker: https://redis.io/docs/getting-started/cluster/
    # 
    # For testing without a cluster, the code will fall back to a single Redis instance.
    asyncio.run(main())