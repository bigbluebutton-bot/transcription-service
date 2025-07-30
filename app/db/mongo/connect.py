from mongoengine import connect
from mongoengine import disconnect # type: ignore
from app.db.mongo.role import Role, Endpoint, Method
from app.db.mongo.user import User
from app.core.config import CONFIG

async def connect_to_mongo():
    # connect to db
    connect(
        db=CONFIG.MONGO_DB_NAME,
        host=CONFIG.MONGO_HOST,
        port=CONFIG.MONGO_PORT,
        username=CONFIG.MONGO_USER,
        password=CONFIG.MONGO_PASSWORD,
        authentication_source=CONFIG.MONGO_AUTH_SOURCE
    )

async def disconnect_from_mongo():
    disconnect()

async def create_default_boss():
    # Create default boss role if it doesn't exist
    if not Role.objects(rolename="boss").first(): # type: ignore[attr-defined]
        boss_role = Role(
            rolename="boss",
            api_endpoints=[
                Endpoint(
                    method=Method.ANY,
                    path_filter="/*"
                )
            ]
        )
        print(boss_role.to_mongo())
        boss_role.save()

    BOSS_ROLE = Role.objects(rolename="boss").first() # type: ignore[attr-defined]
    if BOSS_ROLE is None:
        raise Exception("Boss role not found. Please reinitialize the database.")

    # create a boss user if it doesn't exist
    if not User.objects(username="boss").first(): # type: ignore[attr-defined]
        boss_user = User(
            username="boss",
            roles=[BOSS_ROLE]
        )
        boss_user.set_password("boss")
        boss_user.save()

    # check if the boss user has the boss role, and add it if not
    BOSS_USER = User.objects(username="boss").first() # type: ignore[attr-defined]
    if not BOSS_ROLE in BOSS_USER.roles: # type: ignore[attr-defined]
        BOSS_USER.roles.append(BOSS_ROLE) # type: ignore[attr-defined]
        BOSS_USER.save()

if __name__ == "__main__":
    connect_to_mongo()
    create_default_boss()
    disconnect_from_mongo()