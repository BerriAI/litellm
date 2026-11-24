import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import psycopg

from tests.integration._support.client import Gateway, eventually, object_value
from tests.integration._support.database import read_rows, scratch_database, write_rows
from tests.integration._support.process import owned_proxy, owned_proxy_process
from tests.integration._support.wire import Reply, Request, wire_server


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


def test_user_info_v2_remains_available_when_the_writer_user_table_is_locked(gateway: Gateway, tmp_path: Path) -> None:
    user: Final = f"replica-user-{uuid.uuid4().hex}"
    params: Final = {"user_id": user}
    with scratch_database("user_info_writer") as writer, scratch_database("user_info_replica") as reader:
        with owned_proxy(gateway, tmp_path / "seed-replica", {"DATABASE_URL": reader}) as replica_seed:
            replica_seed.post("/user/new", {"user_id": user, "user_role": "internal_user", "max_budget": 10})
            key: Final = replica_seed.post("/key/generate", {"user_id": user})["key"]
            assert isinstance(key, str)
        with owned_proxy_process(
            gateway,
            tmp_path / "replica-proxy",
            {
                "DATABASE_URL": writer,
                "DATABASE_URL_READ_REPLICA": reader,
                "PROXY_DB_LOOKUP_DEADLINE_SECONDS": "10",
            },
            client_timeout=16,
        ) as proxy:
            write_rows(
                """INSERT INTO "LiteLLM_UserTable" (user_id, user_role, max_budget)
                   VALUES (%s, 'internal_user', 10)""",
                (user,),
                database_url=writer,
            )
            expected: Final = proxy.gateway.get("/v2/user/info", params)
            assert expected["user_id"] == user and expected["max_budget"] == 10
            write_rows(
                """UPDATE "LiteLLM_UserTable"
                   SET spend = 12, temp_budget_increase = 3, temp_budget_expiry = now() + interval '1 hour'
                   WHERE user_id = %s""",
                (user,),
                database_url=reader,
            )
            with psycopg.connect(writer) as locked:
                locked.execute('LOCK TABLE "LiteLLM_UserTable" IN ACCESS EXCLUSIVE MODE')
                result: Final = proxy.gateway.get("/v2/user/info", params)
                assert result["user_id"] == user and result["max_budget"] == 10
                own: Final = proxy.gateway.request("GET", "/v2/user/info", params=params, key=key)
                assert own.status_code == 200, own.text
                assert own.json()["max_budget"] == 10
                chat: Final = proxy.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": "openai/gpt-5-mini", "messages": [{"role": "user", "content": "budget check"}]},
                    key=key,
                )
                assert chat.status_code == 422 and "budget" in chat.text.lower(), chat.text
                readiness: Final = proxy.gateway.request("GET", "/health/readiness")
                assert readiness.status_code == 200, readiness.text
                assert readiness.json()["db"] == "connected"
            write_rows(
                """UPDATE "LiteLLM_UserTable"
                   SET spend = 0, temp_budget_increase = NULL, temp_budget_expiry = NULL WHERE user_id = %s""",
                (user,),
                database_url=reader,
            )
            assert proxy.gateway.get("/v2/user/info", params) == expected
            expiry: Final = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            proxy.gateway.post("/user/update", {**params, "temp_budget_increase": 3, "temp_budget_expiry": expiry})
            assert proxy.gateway.get("/v2/user/info", params).get("temp_budget_increase") is None
            current: Final = proxy.gateway.get("/v2/user/info", {**params, "read_after_write": "true"})
            assert (current.get("temp_budget_increase"), current.get("effective_max_budget")) == (3, 13)
            proxy.gateway.post("/user/update", {**params, "temp_budget_increase": None, "temp_budget_expiry": None})
            cleared: Final = proxy.gateway.get("/v2/user/info", {**params, "read_after_write": "true"})
            assert (cleared.get("temp_budget_increase"), cleared.get("effective_max_budget")) == (None, None)


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


def test_named_proxy_admin_temporary_budget_preserves_the_existing_base_cap(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(
            user_role="proxy_admin", max_budget=10, budget_duration="30d", rollover_max_budget=15
        )
        write_rows('UPDATE "LiteLLM_UserTable" SET rollover_credit = 5 WHERE user_id = %s', (user,))
        gateway.post("/user/update", {"user_id": user, "spend": 12})
        key: Final = scenario.key(user_id=user, models=[model])
        chat: Final = {"model": model, "messages": [{"role": "user", "content": "named admin budget"}]}
        blocked: Final = gateway.request("POST", "/v1/chat/completions", chat, key=key)
        assert blocked.status_code == 422, blocked.text
        assert "budget" in blocked.text.lower(), blocked.text

        expiry: Final = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        gateway.post(
            "/user/update", {"user_id": user, "spend": 16, "temp_budget_increase": 3, "temp_budget_expiry": expiry}
        )
        info: Final = gateway.get("/v2/user/info", {"user_id": user})
        assert (info["rollover_credit"], info["temp_budget_increase"], info["effective_max_budget"]) == (5, 3, 18)
        allowed: Final = gateway.request("POST", "/v1/chat/completions", chat, key=key)
        assert allowed.status_code == 200, allowed.text

        write_rows(
            'UPDATE "LiteLLM_UserTable" SET temp_budget_expiry = %s WHERE user_id = %s',
            (datetime(2000, 1, 1, tzinfo=timezone.utc), user),
        )
        expired: Final = gateway.get("/v2/user/info", {"user_id": user})
        assert expired["effective_max_budget"] == 15
        gateway.post("/user/update", {"user_id": user, "temp_budget_increase": None, "temp_budget_expiry": None})
        cleared_info: Final = gateway.get("/v2/user/info", {"user_id": user})
        assert cleared_info["effective_max_budget"] == expired["effective_max_budget"]
        cleared: Final = gateway.request("POST", "/v1/chat/completions", chat, key=key)
        assert cleared.status_code == 422, cleared.text


def test_paid_fallback_obeys_the_named_admin_temporary_cap(gateway: Gateway, tmp_path: Path) -> None:
    primary_error: Final = "integration free primary failed"
    hold: Final = threading.Event()
    entered: Final = threading.Event()
    release: Final = threading.Event()

    def respond(_request: Request) -> Reply:
        if hold.is_set():
            entered.set()
        return Reply(
            status=500,
            body=json.dumps({"error": {"message": primary_error}}).encode(),
            gate_before_first=release if hold.is_set() else None,
        )

    with (
        wire_server(respond) as upstream,
        scratch_database("named_admin_fallback") as database,
        owned_proxy(gateway, tmp_path, {"DATABASE_URL": database}) as proxy,
        proxy.scenario() as scenario,
    ):
        free: Final = scenario.model(api_base=f"{upstream.url}/v1", input_cost_per_token=0.0, output_cost_per_token=0.0)
        paid: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.001)
        proxy.post(
            "/config/update",
            {
                "general_settings": {"enforce_fallback_budget": True},
                "router_settings": {"num_retries": 0, "fallbacks": [{free: [paid]}]},
            },
        )
        user: Final = scenario.user(
            user_role="proxy_admin", max_budget=10, budget_duration="30d", rollover_max_budget=15
        )
        write_rows(
            'UPDATE "LiteLLM_UserTable" SET rollover_credit = 5 WHERE user_id = %s',
            (user,),
            database_url=database,
        )
        key: Final = scenario.key(user_id=user, models=[free, paid])
        expiry: Final = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        proxy.post(
            "/user/update", {"user_id": user, "spend": 19, "temp_budget_increase": 3, "temp_budget_expiry": expiry}
        )
        chat: Final = {"model": free, "messages": [{"role": "user", "content": "paid fallback budget"}]}
        blocked: Final = proxy.request("POST", "/v1/chat/completions", chat, key=key)
        assert blocked.status_code == 500 and primary_error in blocked.text, blocked.text

        proxy.post("/user/update", {"user_id": user, "spend": 16})
        allowed: Final = proxy.request("POST", "/v1/chat/completions", chat, key=key)
        assert allowed.status_code == 200, allowed.text

        hold.set()
        proxy.post("/user/update", {"user_id": user, "spend": 2})
        with ThreadPoolExecutor(max_workers=1) as pool:
            under_cap: Final = pool.submit(proxy.request, "POST", "/v1/chat/completions", chat, key=key)
            try:
                assert entered.wait(timeout=15), "The upstream did not receive the admitted request"
                with psycopg.connect(database) as locked:
                    locked.execute('LOCK TABLE "LiteLLM_UserTable" IN ACCESS EXCLUSIVE MODE')
                    release.set()
                    while_locked: Final = under_cap.result(timeout=8)
                    assert while_locked.status_code == 200, while_locked.text
            finally:
                release.set()

        entered.clear()
        release.clear()
        proxy.post("/user/update", {"user_id": user, "spend": 16})
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending: Final = pool.submit(proxy.request, "POST", "/v1/chat/completions", chat, key=key)
            try:
                assert entered.wait(timeout=15), "The upstream did not receive the admitted request"
                proxy.post("/user/update", {"user_id": user, "temp_budget_increase": None, "temp_budget_expiry": None})
            finally:
                release.set()
            after_clear: Final = pending.result(timeout=15)
        assert after_clear.status_code == 500 and primary_error in after_clear.text, after_clear.text

        cleared: Final = proxy.request("POST", "/v1/chat/completions", chat, key=key)
        assert cleared.status_code == 500 and primary_error in cleared.text, cleared.text


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
