import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Final

from tests.integration._support.client import Gateway, eventually, object_value
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


def test_temporary_user_budget_allows_requests_without_changing_base_budget(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.001)
        user: Final = scenario.user(user_role="internal_user", max_budget=1)
        key: Final = scenario.key(user_id=user, models=[model])
        gateway.post("/user/update", {"user_id": user, "spend": 2})
        chat: Final = {"model": model, "messages": [{"role": "user", "content": "temporary user budget"}]}
        blocked: Final = gateway.request("POST", "/v1/chat/completions", chat, key=key)
        assert blocked.status_code == 422, blocked.text
        assert "budget" in blocked.text.lower(), blocked.text
        peer_blocked: Final = peer.request("POST", "/v1/chat/completions", chat, key=key)
        assert peer_blocked.status_code == 422, peer_blocked.text

        expiry: Final = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=12)
        update: Final = gateway.request(
            "POST",
            "/user/update",
            {"user_id": user, "temp_budget_increase": 9, "temp_budget_expiry": expiry.isoformat()},
        )
        assert update.status_code == 200, update.text
        info: Final = gateway.get("/v2/user/info", {"user_id": user})
        assert (info.get("max_budget"), info.get("temp_budget_increase"), info.get("effective_max_budget")) == (
            1,
            9,
            10,
        )
        expiry_value: Final = info.get("temp_budget_expiry")
        assert isinstance(expiry_value, str) and datetime.fromisoformat(expiry_value) == expiry
        assert info.get("metadata") == {}

        legacy_info: Final = gateway.get("/user/info", {"user_id": user})["user_info"]
        assert (
            legacy_info["max_budget"],
            legacy_info["temp_budget_increase"],
            legacy_info["effective_max_budget"],
        ) == (
            1,
            9,
            10,
        )

        allowed: Final = gateway.request("POST", "/v1/chat/completions", chat, key=key)
        assert allowed.status_code == 200, allowed.text
        peer_allowed: Final = eventually(
            lambda: peer.request("POST", "/v1/chat/completions", chat, key=key),
            lambda result: result.status_code == 200,
            seconds=8,
        )
        assert peer_allowed.status_code == 200, peer_allowed.text

        eventually(lambda: datetime.now(timezone.utc), lambda now: now >= expiry, seconds=15)
        expired: Final = peer.request("POST", "/v1/chat/completions", chat, key=key)
        assert expired.status_code == 422, expired.text
        assert "budget" in expired.text.lower(), expired.text
        expired_info: Final = gateway.get("/v2/user/info", {"user_id": user})
        assert (
            expired_info.get("max_budget"),
            expired_info.get("temp_budget_increase"),
            expired_info.get("effective_max_budget"),
        ) == (1, 9, 1)
        assert expired_info.get("temp_budget_expiry") == expiry_value

        gateway.post(
            "/user/update",
            {
                "user_id": user,
                "temp_budget_increase": 9,
                "temp_budget_expiry": (expiry + timedelta(hours=1)).isoformat(),
            },
        )
        eventually(
            lambda: peer.request("POST", "/v1/chat/completions", chat, key=key),
            lambda result: result.status_code == 200,
            seconds=8,
        )
        gateway.post("/user/update", {"user_id": user, "temp_budget_increase": None, "temp_budget_expiry": None})
        cleared: Final = gateway.get("/v2/user/info", {"user_id": user})
        assert (cleared.get("max_budget"), cleared.get("temp_budget_increase"), cleared.get("temp_budget_expiry")) == (
            1,
            None,
            None,
        )
        cleared_request: Final = gateway.request("POST", "/v1/chat/completions", chat, key=key)
        assert cleared_request.status_code == 422, cleared_request.text
        eventually(
            lambda: peer.request("POST", "/v1/chat/completions", chat, key=key),
            lambda result: result.status_code == 422,
            seconds=8,
        )


def test_temporary_user_budget_requires_a_pair_and_admin(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        user: Final = scenario.user(user_role="internal_user", max_budget=1)
        key: Final = scenario.key(user_id=user)
        expiry: Final = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        incomplete: Final = gateway.request("POST", "/user/update", {"user_id": user, "temp_budget_increase": 9})
        assert incomplete.status_code == 422, incomplete.text
        assert "temp_budget_increase and temp_budget_expiry must be set together" in incomplete.text

        escalation: Final = gateway.request(
            "POST",
            "/user/update",
            {"user_id": user, "temp_budget_increase": 9, "temp_budget_expiry": expiry},
            key=key,
        )
        assert escalation.status_code == 401, escalation.text
