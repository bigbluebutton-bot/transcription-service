#!/usr/bin/env python3
"""
Simplified Redis async/await test script without complex dependencies.
This demonstrates the proper way to handle async Redis operations.
"""

import asyncio
import uuid
import time
from typing import List

# Mock Redis cluster for demonstration purposes
class MockRedisCluster:
    """Mock Redis cluster that simulates async operations"""
    
    def __init__(self):
        self.data = {}
        self.sets = {}
    
    async def set(self, key: str, value: str, ex: int = None):
        """Mock SET operation"""
        await asyncio.sleep(0.001)  # Simulate network delay
        self.data[key] = value
        return True
    
    async def get(self, key: str):
        """Mock GET operation"""
        await asyncio.sleep(0.001)
        return self.data.get(key)
    
    async def delete(self, key: str):
        """Mock DELETE operation"""
        await asyncio.sleep(0.001)
        if key in self.data:
            del self.data[key]
            return 1
        return 0
    
    async def sadd(self, key: str, *values):
        """Mock SADD operation"""
        await asyncio.sleep(0.001)
        if key not in self.sets:
            self.sets[key] = set()
        added = 0
        for value in values:
            if value not in self.sets[key]:
                self.sets[key].add(value)
                added += 1
        return added
    
    async def smembers(self, key: str):
        """Mock SMEMBERS operation"""
        await asyncio.sleep(0.001)
        return list(self.sets.get(key, set()))
    
    async def srem(self, key: str, *values):
        """Mock SREM operation"""
        await asyncio.sleep(0.001)
        if key not in self.sets:
            return 0
        removed = 0
        for value in values:
            if value in self.sets[key]:
                self.sets[key].remove(value)
                removed += 1
        return removed
    
    async def exists(self, key: str):
        """Mock EXISTS operation"""
        await asyncio.sleep(0.001)
        return 1 if key in self.data else 0
    
    async def ttl(self, key: str):
        """Mock TTL operation"""
        await asyncio.sleep(0.001)
        # For demo purposes, return a mock TTL
        return 3600 if key in self.data else -1


async def login(user_id: int, user_data: dict, redis_client: MockRedisCluster):
    """
    Simuliert den Login und misst die Zeit für die Redis-Operationen.
    """
    session_id = f"session:{uuid.uuid4()}"
    user_sessions_set_key = f"user:{user_id}:sessions"
    
    start_time = time.time()
    
    # Session speichern (SET + SADD)
    await redis_client.set(session_id, str(user_data), ex=3600)
    await redis_client.sadd(user_sessions_set_key, session_id)
    
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    
    print(f"✅ User {user_id} angemeldet. Session-ID: {session_id}")
    print(f"   ⏱️  Dauer für SET+SADD: {duration_ms:.4f} ms")
    
    return session_id


async def get_all_sessions_for_user(user_id: int, redis_client: MockRedisCluster):
    """
    Holt alle Sessions und misst die Zeit für die SMEMBERS-Operation.
    """
    user_sessions_set_key = f"user:{user_id}:sessions"
    
    start_time = time.time()
    session_ids = await redis_client.smembers(user_sessions_set_key)
    end_time = time.time()
    
    duration_ms = (end_time - start_time) * 1000
    
    print(f"\nℹ️  Aktive Sessions für User {user_id} (Abfrage dauerte {duration_ms:.4f} ms):")
    if not session_ids:
        print("   -> Keine.")
    else:
        for sid in session_ids:
            # Diese Logik ist für die Anzeige und wird nicht extra gemessen
            # FIXED: Properly await all Redis operations
            if await redis_client.exists(sid):
                ttl = await redis_client.ttl(sid)
                print(f"   -> {sid} (Gültig für {ttl}s)")
            else:
                print(f"   -> {sid} (Abgelaufen, wird aus Set entfernt)")
                await redis_client.srem(user_sessions_set_key, sid)
    return session_ids


async def logout(user_id: int, session_id: str, redis_client: MockRedisCluster):
    """
    Meldet eine einzelne Session ab und misst die Zeit.
    """
    user_sessions_set_key = f"user:{user_id}:sessions"
    
    start_time = time.time()
    
    # Session löschen (DEL + SREM)
    await redis_client.delete(session_id)
    await redis_client.srem(user_sessions_set_key, session_id)
    
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    
    print(f"❌ User {user_id} von Session {session_id} abgemeldet.")
    print(f"   ⏱️  Dauer für DEL+SREM: {duration_ms:.4f} ms")


async def logout_from_all_devices(user_id: int, redis_client: MockRedisCluster):
    """
    Meldet alle Sessions ab und misst die Zeit.
    """
    user_sessions_set_key = f"user:{user_id}:sessions"
    
    # Alle Sessions holen
    session_ids = await redis_client.smembers(user_sessions_set_key)
    
    if not session_ids:
        print(f"ℹ️  User {user_id} hat keine aktiven Sessions.")
        return
    
    start_time = time.time()
    
    # Alle Sessions löschen
    for session_id in session_ids:
        await redis_client.delete(session_id)
    
    # Set leeren
    for session_id in session_ids:
        await redis_client.srem(user_sessions_set_key, session_id)
    
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    
    print(f"❌ User {user_id} von allen {len(session_ids)} Sessions abgemeldet.")
    print(f"   ⏱️  Dauer für Bulk-Logout: {duration_ms:.4f} ms")


async def main():
    """
    Hauptfunktion zum Testen der Session-Verwaltung.
    """
    USER_ID = 123
    USER_DATA = {"username": "testuser", "role": "admin"}
    
    # Mock Redis client erstellen
    redis_client = MockRedisCluster()
    
    print("🚀 Starte Session-Management Test mit korrektem async/await...")
    
    # 1. Mehrere Sessions erstellen
    session1 = await login(USER_ID, USER_DATA, redis_client)
    session2 = await login(USER_ID, USER_DATA, redis_client)
    session3 = await login(USER_ID, USER_DATA, redis_client)
    
    # 2. Alle Sessions anzeigen
    await get_all_sessions_for_user(USER_ID, redis_client)
    
    # 3. Eine Session abmelden
    await logout(USER_ID, session1, redis_client)
    
    # 4. Verbleibende Sessions anzeigen
    await get_all_sessions_for_user(USER_ID, redis_client)
    
    # 5. Neue Session erstellen
    session4 = await login(USER_ID, USER_DATA, redis_client)
    
    # 6. Von allen Geräten abmelden
    await logout_from_all_devices(USER_ID, redis_client)
    
    # 7. Finale Überprüfung
    await get_all_sessions_for_user(USER_ID, redis_client)
    
    print("\n--- ENDE ---")


if __name__ == "__main__":
    asyncio.run(main())
