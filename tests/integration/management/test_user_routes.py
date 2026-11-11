import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Final

from tests.integration._support.client import Gateway, object_value
from tests.integration._support.database import read_rows


def test_concurrent_user_creation_persists_models_and_aliases(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        alias: Final = f"alias-{uuid.uuid4().hex}"
        with ThreadPoolExecutor(max_workers=10) as pool:
            users: Final = list(pool.map(lambda _: scenario.user(models=[model], aliases={alias: model}), range(10)))
        assert len(set(users)) == 10
        rows: Final = [
            read_rows('SELECT models FROM "LiteLLM_UserTable" WHERE user_id = %s', (user,)) for user in users
        ]
        assert rows == [[{"models": [model]}]] * len(users)


def test_user_info_serves_admin_and_self_and_denies_other_users(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        user: Final = scenario.user(user_role="internal_user")
        own_key: Final = scenario.key(user_id=user)
        other: Final = scenario.user(user_role="internal_user")
        other_key: Final = scenario.key(user_id=other)
        admin: Final = object_value(gateway.get("/user/info", {"user_id": user})["user_info"])
        assert admin["user_id"] == user
        own: Final = gateway.request("GET", "/user/info", params={"user_id": user}, key=own_key)
        assert own.status_code == 200, own.text
        assert object_value(object_value(own.json())["user_info"])["user_id"] == user
        denied: Final = gateway.request("GET", "/user/info", params={"user_id": user}, key=other_key)
        assert denied.status_code == 403, denied.text
