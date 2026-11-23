from hashlib import sha256
from typing import Final

import httpx
from integration._support.client import Gateway, Scenario, object_value
from integration._support.database import read_rows
from pydantic import JsonValue

_SELF_SERVICE_ROUTES: Final = ("/user/update", "/user/bulk_update", "/user/info")


def _chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "user block probe"}]},
        key=key,
    )


def _user_metadata(user_id: str) -> dict[str, JsonValue]:
    rows: Final = read_rows('SELECT metadata FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,))
    assert len(rows) == 1, rows
    return object_value(rows[0]["metadata"] or {})


def _key_row(key: str) -> dict[str, JsonValue]:
    rows: Final = read_rows(
        'SELECT blocked, metadata FROM "LiteLLM_VerificationToken" WHERE token = %s',
        (sha256(key.encode()).hexdigest(),),
    )
    assert len(rows) == 1, rows
    return rows[0]


def _update(gateway: Gateway, body: dict[str, JsonValue], key: str | None = None) -> httpx.Response:
    return gateway.request("POST", "/user/update", body, key=key)


def _user_with_key(scenario: Scenario, model: str) -> tuple[str, str]:
    user_id: Final = scenario.user(user_role="internal_user")
    return user_id, scenario.key(user_id=user_id, models=[model])


def test_disabling_a_user_denies_their_keys_and_reenabling_restores_only_the_keys_it_blocked(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id, key = _user_with_key(scenario, model)
        hand_blocked_key: Final = scenario.key(user_id=user_id, models=[model])
        assert gateway.post("/key/block", {"key": hand_blocked_key})["blocked"] is True
        assert _chat(gateway, model, key).status_code == 200

        disabled: Final = _update(gateway, {"user_id": user_id, "blocked": True})
        assert disabled.status_code == 200, disabled.text
        assert _user_metadata(user_id)["scim_active"] is False
        assert _key_row(key)["blocked"] is True
        assert _chat(gateway, model, key).status_code == 401
        listed: Final = object_value(gateway.get("/user/info", {"user_id": user_id})["user_info"])
        assert listed["user_id"] == user_id

        metadata_edit: Final = _update(gateway, {"user_id": user_id, "metadata": {"team": "core-infra"}})
        assert metadata_edit.status_code == 200, metadata_edit.text
        assert _user_metadata(user_id) == {"team": "core-infra", "scim_active": False}
        assert _chat(gateway, model, key).status_code == 401

        enabled: Final = _update(gateway, {"user_id": user_id, "blocked": False})
        assert enabled.status_code == 200, enabled.text
        assert _user_metadata(user_id) == {"team": "core-infra", "scim_active": True}
        assert _chat(gateway, model, key).status_code == 200
        assert _key_row(key) == {"blocked": False, "metadata": {}}
        assert _key_row(hand_blocked_key)["blocked"] is True
        assert _chat(gateway, model, hand_blocked_key).status_code == 401


def test_blocked_false_on_an_enabled_user_changes_nothing(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id, key = _user_with_key(scenario, model)
        hand_blocked_key: Final = scenario.key(user_id=user_id, models=[model])
        assert gateway.post("/key/block", {"key": hand_blocked_key})["blocked"] is True
        metadata_before: Final = _user_metadata(user_id)

        updated: Final = _update(gateway, {"user_id": user_id, "blocked": False})

        assert updated.status_code == 200, updated.text
        assert _user_metadata(user_id) == metadata_before
        assert "scim_active" not in metadata_before
        assert _key_row(hand_blocked_key)["blocked"] is True
        assert _chat(gateway, model, key).status_code == 200


def test_bulk_update_users_list_disables_and_reenables_each_user(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        first_user, first_key = _user_with_key(scenario, model)
        second_user, second_key = _user_with_key(scenario, model)

        disabled: Final = gateway.post(
            "/user/bulk_update",
            {"users": [{"user_id": first_user, "blocked": True}, {"user_id": second_user, "blocked": True}]},
        )
        assert disabled["successful_updates"] == 2, disabled
        assert _chat(gateway, model, first_key).status_code == 401
        assert _chat(gateway, model, second_key).status_code == 401

        enabled: Final = gateway.post(
            "/user/bulk_update",
            {"users": [{"user_id": first_user, "blocked": False}, {"user_id": second_user, "blocked": False}]},
        )
        assert enabled["successful_updates"] == 2, enabled
        assert _chat(gateway, model, first_key).status_code == 200
        assert _chat(gateway, model, second_key).status_code == 200


def test_bulk_update_all_users_refuses_blocked(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id, key = _user_with_key(scenario, model)

        refused: Final = gateway.request(
            "POST", "/user/bulk_update", {"all_users": True, "user_updates": {"blocked": True}}
        )

        assert refused.status_code == 400, refused.text
        assert "'users' list" in refused.text, refused.text
        assert "scim_active" not in _user_metadata(user_id)
        assert _chat(gateway, model, key).status_code == 200


def test_a_non_admin_cannot_block_a_user(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = scenario.user(user_role="internal_user")
        self_key: Final = scenario.key(
            user_id=user_id, models=[model], allowed_routes=[*_SELF_SERVICE_ROUTES, "/v1/chat/completions"]
        )

        single: Final = _update(gateway, {"user_id": user_id, "blocked": True}, key=self_key)
        bulk: Final = gateway.request(
            "POST", "/user/bulk_update", {"users": [{"user_id": user_id, "blocked": True}]}, key=self_key
        )

        assert single.status_code == 403, single.text
        assert bulk.status_code == 403, bulk.text
        assert "scim_active" not in _user_metadata(user_id)
        assert _chat(gateway, model, self_key).status_code == 200
