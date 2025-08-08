from app.db.redis.redis import get_redis_cluster
import uuid
import json
import time
import asyncio

# --- Konfiguration ---
r = get_redis_cluster()

# --- Kernfunktionen mit Zeitmessung ---

async def login(user_id: int, user_data: dict):
    """
    Simuliert den Login und misst die Zeit für die Redis-Operationen.
    """
    session_id = f"session:{uuid.uuid4()}"
    user_sessions_set_key = f"user:{user_id}:sessions"
    
    # Zeitmessung für die Schreiboperationen (SET + SADD)
    start_time = time.perf_counter()
    
    # Eine Pipeline verwenden, um beide Befehle in einem einzigen Round-Trip zu senden
    pipe = r.pipeline()
    pipe.set(session_id, json.dumps(user_data), ex=60)
    pipe.sadd(user_sessions_set_key, session_id)
    await pipe.execute()
    
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    
    print(f"✅ User {user_id} angemeldet. Session-ID: {session_id}")
    print(f"   ⏱️  Dauer für SET+SADD: {duration_ms:.4f} ms")
    return session_id

async def get_all_sessions_for_user(user_id: int):
    """
    Holt alle Sessions und misst die Zeit für die SMEMBERS-Operation.
    """
    user_sessions_set_key = f"user:{user_id}:sessions"
    
    # Zeitmessung für die Leseoperation (SMEMBERS)
    start_time = time.perf_counter()
    session_ids = await r.smembers(user_sessions_set_key)
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    
    print(f"\nℹ️  Aktive Sessions für User {user_id} (Abfrage dauerte {duration_ms:.4f} ms):")
    if not session_ids:
        print("   -> Keine.")
    else:
        for sid in session_ids:
            # Diese Logik ist für die Anzeige und wird nicht extra gemessen
            if await r.exists(sid):
                ttl = await r.ttl(sid)
                # print(f"   -> {sid} (Gültig für {ttl}s)")
            else:
                # print(f"   -> {sid} (Abgelaufen, wird aus Set entfernt)")
                await r.srem(user_sessions_set_key, sid)
    return session_ids

async def logout(user_id: int, session_id: str):
    """
    Meldet eine einzelne Session ab und misst die Zeit.
    """
    user_sessions_set_key = f"user:{user_id}:sessions"
    
    start_time = time.perf_counter()
    
    pipe = r.pipeline()
    pipe.delete(session_id)
    pipe.srem(user_sessions_set_key, session_id)
    await pipe.execute()
    
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    
    print(f"\n❌ User {user_id} von Session {session_id} abgemeldet.")
    print(f"   ⏱️  Dauer für DEL+SREM: {duration_ms:.4f} ms")

async def logout_from_all_devices(user_id: int):
    """
    Meldet alle Sessions ab und misst die Zeit.
    """
    user_sessions_set_key = f"user:{user_id}:sessions"
    
    # Schritt 1: Alle Session-IDs holen
    start_get_time = time.perf_counter()
    session_ids = await r.smembers(user_sessions_set_key)
    end_get_time = time.perf_counter()
    get_duration_ms = (end_get_time - start_get_time) * 1000
    
    if not session_ids:
        print(f"\n🤷 User {user_id} war nirgends angemeldet.")
        return

    # Schritt 2: Alle Sessions und das Set löschen
    start_del_time = time.perf_counter()
    
    pipe = r.pipeline()
    # Die Liste der Schlüssel für den DEL-Befehl vorbereiten
    keys_to_delete = list(session_ids)
    keys_to_delete.append(user_sessions_set_key)
    pipe.delete(*keys_to_delete)
    pipe.execute()
    
    end_del_time = time.perf_counter()
    del_duration_ms = (end_del_time - start_del_time) * 1000
    
    print(f"\n💥 User {user_id} von allen {len(session_ids)} Geräten abgemeldet.")
    print(f"   ⏱️  Dauer für SMEMBERS: {get_duration_ms:.4f} ms")
    print(f"   ⏱️  Dauer für das Löschen von {len(keys_to_delete)} Schlüsseln: {del_duration_ms:.4f} ms")


async def main():
    # Vor dem Test aufräumen, um saubere Ergebnisse zu gewährleisten
    await r.flushdb()
    
    USER_ID = 123
    USER_DATA = {"username": "Max", "roles": ["editor"]}

    print("--- START: Demo für sicheres Session-Management mit Zeitmessung ---")

    # Simuliere 5 Logins für den Benutzer
    for i in range(100000):
        await login(USER_ID, {'device': 'Desktop', 'ip': '192.168.1.100'})
    await get_all_sessions_for_user(USER_ID) 
    
    await get_all_sessions_for_user(USER_ID)
    
    session_to_logout = await login(USER_ID, {'device': 'Desktop', 'ip': '192.168.1.100'})
    await logout(USER_ID, session_to_logout)
    
    await get_all_sessions_for_user(USER_ID)
    
    print("\n--- ENDE ---")

# --- Beispiel-Ablauf ---
if __name__ == "__main__":
    asyncio.run(main())