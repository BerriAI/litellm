import os
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import httpx
import psycopg
import pytest
from integration._support.budget import case_timeout_seconds
from integration._support.client import Gateway, Scenario, eventually, held, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.redis_process import owned_redis
from psycopg import sql
from pydantic import JsonValue

_LLM_ROUTES: Final[Mapping[str, Mapping[str, JsonValue]]] = {
    "/v1/chat/completions": {"messages": [{"role": "user", "content": "user block probe"}]},
    "/v1/messages": {"max_tokens": 16, "messages": [{"role": "user", "content": "user block probe"}]},
    "/v1/responses": {"input": "user block probe"},
}
_BURST: Final = 30

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


def _llm(gateway: Gateway, path: str, model: str, key: str) -> httpx.Response:
    return gateway.request("POST", path, {"model": model, **_LLM_ROUTES[path]}, key=key)


def _statuses(gateway: Gateway, model: str, key: str) -> dict[str, int]:
    return {path: _llm(gateway, path, model, key).status_code for path in _LLM_ROUTES}


def test_disabling_a_user_denies_every_llm_route_and_reenabling_serves_them_again(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id, key = _user_with_key(scenario, model)
        served: Final = _statuses(gateway, model, key)
        assert 401 not in served.values(), served
        assert served["/v1/chat/completions"] == 200, served

        assert _update(gateway, {"user_id": user_id, "blocked": True}).status_code == 200
        assert _statuses(gateway, model, key) == dict.fromkeys(_LLM_ROUTES, 401)

        assert _update(gateway, {"user_id": user_id, "blocked": False}).status_code == 200
        assert _statuses(gateway, model, key) == served


def test_disabling_on_one_warmed_worker_denies_on_both_and_reenabling_on_the_peer_restores_both(
    gateway: Gateway, peer: Gateway
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id, key = _user_with_key(scenario, model)
        for worker in (gateway, peer):
            eventually(lambda worker=worker: _chat(worker, model, key).status_code, lambda status: status == 200)

        disabled: Final = _update(gateway, {"user_id": user_id, "blocked": True})
        assert disabled.status_code == 200, disabled.text
        assert _chat(gateway, model, key).status_code == 401
        eventually(lambda: _chat(peer, model, key).status_code, lambda status: status == 401, seconds=3)

        enabled: Final = _update(peer, {"user_id": user_id, "blocked": False})
        assert enabled.status_code == 200, enabled.text
        assert _chat(peer, model, key).status_code == 200
        eventually(lambda: _chat(gateway, model, key).status_code, lambda status: status == 200, seconds=3)
        assert _key_row(key) == {"blocked": False, "metadata": {}}


def test_a_key_minted_for_a_disabled_user_is_denied_until_the_user_is_reenabled(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id, key = _user_with_key(scenario, model)
        assert _update(gateway, {"user_id": user_id, "blocked": True}).status_code == 200

        late_key: Final = scenario.key(user_id=user_id, models=[model])
        refused: Final = _chat(gateway, model, late_key)
        assert refused.status_code == 401, refused.text

        assert _update(gateway, {"user_id": user_id, "blocked": False}).status_code == 200
        assert _chat(gateway, model, late_key).status_code == 200
        assert _chat(gateway, model, key).status_code == 200


def test_disabling_and_reenabling_by_email_reaches_the_users_keys(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        email: Final = f"block-{uuid.uuid4().hex}@example.com"
        user_id: Final = scenario.user(user_role="internal_user", user_email=email)
        key: Final = scenario.key(user_id=user_id, models=[model])

        disabled: Final = _update(gateway, {"user_email": email, "blocked": True})
        assert disabled.status_code == 200, disabled.text
        assert _user_metadata(user_id)["scim_active"] is False
        assert _chat(gateway, model, key).status_code == 401

        enabled: Final = _update(gateway, {"user_email": email, "blocked": False})
        assert enabled.status_code == 200, enabled.text
        assert _chat(gateway, model, key).status_code == 200
        assert _key_row(key) == {"blocked": False, "metadata": {}}


def test_disabling_twice_is_undone_by_one_reenable(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id, key = _user_with_key(scenario, model)

        for _ in range(2):
            repeated = _update(gateway, {"user_id": user_id, "blocked": True})
            assert repeated.status_code == 200, repeated.text
            assert _key_row(key) == {"blocked": True, "metadata": {"scim_blocked": True}}
            assert _chat(gateway, model, key).status_code == 401

        for _ in range(2):
            enabled = _update(gateway, {"user_id": user_id, "blocked": False})
            assert enabled.status_code == 200, enabled.text
            assert _user_metadata(user_id) == {"scim_active": True}
            assert _key_row(key) == {"blocked": False, "metadata": {}}
            assert _chat(gateway, model, key).status_code == 200


@pytest.mark.parametrize("blocked", ["maybe", "", 2, [True], {"value": True}, "y" * 5120])
def test_a_malformed_blocked_value_is_rejected_and_changes_nothing(gateway: Gateway, blocked: JsonValue) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id, key = _user_with_key(scenario, model)

        single: Final = _update(gateway, {"user_id": user_id, "blocked": blocked})
        bulk: Final = gateway.request(
            "POST", "/user/bulk_update", {"users": [{"user_id": user_id, "blocked": blocked}]}
        )

        assert single.status_code in (400, 422), single.text
        assert bulk.status_code in (400, 422), bulk.text
        assert "scim_active" not in _user_metadata(user_id)
        assert _key_row(key) == {"blocked": None, "metadata": {}}
        assert _chat(gateway, model, key).status_code == 200


def test_an_unauthenticated_or_unknown_caller_cannot_block_a_user(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id, key = _user_with_key(scenario, model)

        anonymous: Final = gateway.client.post("/user/update", json={"user_id": user_id, "blocked": True})
        forged: Final = _update(gateway, {"user_id": user_id, "blocked": True}, key="sk-" + uuid.uuid4().hex)

        assert anonymous.status_code == 401, anonymous.text
        assert forged.status_code == 401, forged.text
        assert "scim_active" not in _user_metadata(user_id)
        assert _chat(gateway, model, key).status_code == 200


def test_edits_without_blocked_keep_a_disabled_user_disabled(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id, key = _user_with_key(scenario, model)
        assert _update(gateway, {"user_id": user_id, "blocked": True}).status_code == 200

        edits: Final[tuple[tuple[dict[str, JsonValue], dict[str, JsonValue]], ...]] = (
            ({"max_budget": 12.5}, {"scim_active": False}),
            ({"metadata": {}}, {"scim_active": False}),
            ({"metadata": {"team": "core-infra"}}, {"team": "core-infra", "scim_active": False}),
        )
        for fields, stored in edits:
            single = _update(gateway, {"user_id": user_id, **fields})
            assert single.status_code == 200, single.text
            assert _user_metadata(user_id) == stored, fields
            assert _chat(gateway, model, key).status_code == 401, fields
        bulk: Final = gateway.post(
            "/user/bulk_update", {"users": [{"user_id": user_id, "metadata": {"cost_center": "42"}}]}
        )
        assert bulk["successful_updates"] == 1, bulk
        assert _user_metadata(user_id) == {"cost_center": "42", "scim_active": False}
        assert _key_row(key) == {"blocked": True, "metadata": {"scim_blocked": True}}
        assert _chat(gateway, model, key).status_code == 401


def test_bulk_update_disables_only_the_entries_that_ask_for_it(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id, key = _user_with_key(scenario, model)
        bystander_id, bystander_key = _user_with_key(scenario, model)

        mixed: Final = gateway.post(
            "/user/bulk_update",
            {"users": [{"user_id": bystander_id, "max_budget": 12.5}, {"user_id": user_id, "blocked": True}]},
        )

        assert mixed["successful_updates"] == 2, mixed
        assert _user_metadata(user_id)["scim_active"] is False
        assert _chat(gateway, model, key).status_code == 401
        assert "scim_active" not in _user_metadata(bystander_id)
        assert _key_row(bystander_key) == {"blocked": None, "metadata": {}}
        assert _chat(gateway, model, bystander_key).status_code == 200


@contextmanager
def _owned_two_worker_proxy(gateway: Gateway, directory: Path) -> Iterator[tuple[OwnedProxy, str]]:
    original: Final = os.environ["DATABASE_URL"]
    identity: Final = "integration_user_block_" + uuid.uuid4().hex
    parsed: Final = urlsplit(original)
    database_url: Final = urlunsplit((parsed.scheme, parsed.netloc, "/" + identity, "", ""))
    with psycopg.connect(original, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(identity)))
        try:
            with (
                owned_redis(directory) as coordination,
                owned_proxy_process(
                    gateway,
                    directory,
                    {
                        "DATABASE_URL": database_url,
                        "REDIS_HOST": coordination.host,
                        "REDIS_PORT": str(coordination.port),
                        "COORDINATION_REDIS_SOCKET_TIMEOUT": "5.0",
                    },
                    config=Path("tests/integration/coordination_redis_proxy_config.yaml"),
                    remove_environment=("DATABASE_URL_READ_REPLICA",),
                    workers=2,
                ) as owned,
            ):
                yield owned, database_url
        finally:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(identity)))


def _owned_metadata(database_url: str, user_id: str) -> dict[str, JsonValue]:
    rows: Final = read_rows(
        'SELECT metadata FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,), database_url=database_url
    )
    assert len(rows) == 1, rows
    return object_value(rows[0]["metadata"] or {})


def _owned_user_with_key(candidate: Gateway, model: str) -> tuple[str, str]:
    user_id: Final = "integration-" + uuid.uuid4().hex
    candidate.post("/user/new", {"user_id": user_id, "auto_create_key": False, "user_role": "internal_user"})
    return user_id, string_value(candidate.post("/key/generate", {"user_id": user_id, "models": [model]})["key"])


@pytest.mark.timeout(case_timeout_seconds())
def test_disabling_users_during_a_burst_with_a_worker_kill_denies_them_on_every_worker(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _owned_two_worker_proxy(gateway, tmp_path) as (owned, database_url), owned.gateway.scenario() as scenario:
        candidate: Final = owned.gateway
        model: Final = scenario.model()
        disabled_users: Final = tuple(_owned_user_with_key(candidate, model) for _ in range(3))
        control_user, control_key = _owned_user_with_key(candidate, model)
        keys: Final = (*(key for _user, key in disabled_users), control_key)
        paths: Final = ("/v1/chat/completions", "/v1/messages")
        for key in keys:
            held(lambda key=key: _chat(candidate, model, key).status_code, lambda status: status == 200, holding=1)

        baseline: Final = {path: _llm(candidate, path, model, control_key).status_code for path in paths}
        assert baseline["/v1/chat/completions"] == 200 and 401 not in baseline.values(), baseline

        def burst_request(index: int) -> tuple[str, str, int]:
            key: Final = keys[index % len(keys)]
            path: Final = paths[index % len(paths)]
            return key, path, _llm(candidate, path, model, key).status_code

        block: Final[dict[str, JsonValue]] = {
            "users": [{"user_id": user_id, "blocked": True} for user_id, _key in disabled_users]
        }
        with ThreadPoolExecutor(_BURST + 1) as pool, owned.replacing_worker(min(owned.worker_pids())) as kill:
            kill()
            racing: Final = [pool.submit(burst_request, index) for index in range(_BURST)]
            disabling: Final = pool.submit(candidate.post, "/user/bulk_update", block)
            raced: Final = [future.result() for future in racing]
            disabled: Final = disabling.result()
            settled: Final = list(pool.map(burst_request, range(_BURST)))

        assert disabled["successful_updates"] == 3, disabled
        for key, path, status in raced:
            assert status == baseline[path] or (key != control_key and status == 401), raced
        for key, path, status in settled:
            assert status == (baseline[path] if key == control_key else 401), settled
        assert len(owned.worker_pids()) == 2
        for _user, key in disabled_users:
            held(lambda key=key: _chat(candidate, model, key).status_code, lambda status: status == 401, holding=1)
        assert "scim_active" not in _owned_metadata(database_url, control_user)

        everyone: Final = candidate.request(
            "POST", "/user/bulk_update", {"all_users": True, "user_updates": {"max_budget": 12.5}}
        )
        assert everyone.status_code == 200, everyone.text
        for user_id, key in disabled_users:
            assert _owned_metadata(database_url, user_id) == {"scim_active": False}
            held(lambda key=key: _chat(candidate, model, key).status_code, lambda status: status == 401, holding=1)
        assert _chat(candidate, model, control_key).status_code == 200

        enabled: Final = candidate.post(
            "/user/bulk_update", {"users": [{"user_id": user_id, "blocked": False} for user_id, _key in disabled_users]}
        )
        assert enabled["successful_updates"] == 3, enabled
        for key in keys:
            held(lambda key=key: _chat(candidate, model, key).status_code, lambda status: status == 200, holding=1)
