from mongoengine import connect
from mongoengine import disconnect # type: ignore
from app.db.mongo.role import Role, Endpoint, Method
from app.db.mongo.user import User
from app.core.config import CONFIG

async def connect_to_mongo(
        db=CONFIG.MONGO_DB_NAME,
        host=CONFIG.MONGO_HOST,
        port=CONFIG.MONGO_PORT,
        username=CONFIG.MONGO_USER,
        password=CONFIG.MONGO_PASSWORD,
        authentication_source=CONFIG.MONGO_AUTH_SOURCE
    ):
    # connect to db
    connect(
        db=db,
        host=host,
        port=port,
        username=username,
        password=password,
        authentication_source=authentication_source
    )

async def disconnect_from_mongo():
    disconnect()

async def create_default_boss(
        admin_role_name: str = "boss",
        admin_user_name: str = "boss",
        admin_password: str = "boss"
    ):
    # Create default boss role if it doesn't exist
    if not Role.objects(rolename=admin_role_name).first(): # type: ignore[attr-defined]
        boss_role = Role(
            rolename=admin_role_name,
            api_endpoints=[
                Endpoint(
                    method=Method.ANY,
                    path_filter="/*"
                )
            ]
        )
        print(boss_role.to_mongo())
        boss_role.save()

    BOSS_ROLE = Role.objects(rolename=admin_role_name).first() # type: ignore[attr-defined]
    if BOSS_ROLE is None:
        raise Exception("Boss role not found. Please reinitialize the database.")

    # create a boss user if it doesn't exist
    if not User.objects(username=admin_user_name).first(): # type: ignore[attr-defined]
        boss_user = User(
            username=admin_user_name,
            roles=[BOSS_ROLE]
        )
        boss_user.set_password(admin_password)
        boss_user.save()

    # check if the boss user has the boss role, and add it if not
    BOSS_USER = User.objects(username=admin_user_name).first() # type: ignore[attr-defined]
    if not BOSS_ROLE in BOSS_USER.roles: # type: ignore[attr-defined]
        BOSS_USER.roles.append(BOSS_ROLE) # type: ignore[attr-defined]
        BOSS_USER.save()

async def main():
    await connect_to_mongo()
    await create_default_boss(
        admin_role_name="admin",
        admin_user_name="admin",
        admin_password="admin"
    )
    await disconnect_from_mongo()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())