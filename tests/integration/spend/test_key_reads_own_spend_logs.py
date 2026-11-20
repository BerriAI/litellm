import os
import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Final

import psycopg
import pytest
from integration._support.client import Gateway, string_value


def _seed_spend_row(
    request_id: str, key: str, user: str = "", organization_id: str | None = None, spend: float = 0.0
) -> None:
    started: Final = datetime.now(timezone.utc) - timedelta(minutes=5)
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute(
            'INSERT INTO "LiteLLM_SpendLogs" '
            '(request_id, call_type, api_key, "user", "startTime", "endTime", status, organization_id, spend) '
            "VALUES (%s, 'acompletion', %s, %s, %s, %s, 'success', %s, %s)",
            (request_id, sha256(key.encode()).hexdigest(), user, started, started, organization_id, spend),
        )


def _delete_spend_rows(request_ids: tuple[str, ...]) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s)', (list(request_ids),))


def _window() -> dict[str, str]:
    today: Final = datetime.now(timezone.utc).date()
    return {"start_date": str(today - timedelta(days=1)), "end_date": str(today + timedelta(days=1))}


def _request_ids(payload: object) -> tuple[str, ...]:
    rows: Final = payload["data"] if isinstance(payload, dict) else payload
    return tuple(sorted(row["request_id"] for row in rows))


@pytest.mark.parametrize(
    ("route", "params"),
    [
        ("/spend/logs", {}),
        ("/spend/logs", {"summarize": "false"}),
        ("/spend/logs/v2", {}),
    ],
)
def test_key_without_user_reads_only_its_own_spend_logs(gateway: Gateway, route: str, params: dict[str, str]) -> None:
    own_request: Final = f"own-{uuid.uuid4().hex}"
    other_request: Final = f"other-{uuid.uuid4().hex}"
    dated: Final = {**params, **_window()} if route.endswith("/v2") or params else params
    with gateway.scenario() as scenario:
        own_key: Final = scenario.key()
        other_key: Final = scenario.key()
        scenario.cleanups.callback(_delete_spend_rows, (own_request, other_request))
        _seed_spend_row(own_request, own_key)
        _seed_spend_row(other_request, other_key)
        own_hash: Final = sha256(own_key.encode()).hexdigest()
        other_hash: Final = sha256(other_key.encode()).hexdigest()

        unfiltered: Final = gateway.request("GET", route, params=dated, key=own_key)
        assert unfiltered.status_code == 200, unfiltered.text
        assert _request_ids(unfiltered.json()) == (own_request,), unfiltered.text[:500]

        by_own_hash: Final = gateway.request("GET", route, params={**dated, "api_key": own_hash}, key=own_key)
        assert by_own_hash.status_code == 200, by_own_hash.text
        assert _request_ids(by_own_hash.json()) == (own_request,), by_own_hash.text[:500]

        for foreign_filter in ({"api_key": other_hash}, {"user_id": "someone-else"}):
            foreign = gateway.request("GET", route, params={**dated, **foreign_filter}, key=own_key)
            assert foreign.status_code == 403, f"{foreign_filter}: {foreign.status_code} {foreign.text[:300]}"


@pytest.mark.parametrize("route", ["/spend/logs", "/spend/logs/v2"])
@pytest.mark.parametrize("administers_org", [True, False])
def test_org_admin_reads_stored_organization_rows(gateway: Gateway, route: str, administers_org: bool) -> None:
    own_request: Final = f"own-{uuid.uuid4().hex}"
    other_request: Final = f"other-{uuid.uuid4().hex}"
    null_request: Final = f"null-{uuid.uuid4().hex}"
    window: Final = _window()
    with gateway.scenario() as scenario:
        own_org: Final = scenario.organization()
        other_org: Final = scenario.organization()
        admin: Final = scenario.user(user_role="internal_user")
        gateway.post(
            "/organization/member_add",
            {
                "organization_id": own_org,
                "member": {"role": "org_admin" if administers_org else "internal_user", "user_id": admin},
            },
        )
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
            connection.execute('UPDATE "LiteLLM_UserTable" SET user_role = %s WHERE user_id = %s', ("org_admin", admin))
        admin_key: Final = scenario.key(user_id=admin)
        other_user: Final = scenario.user(user_role="internal_user")
        other_key: Final = scenario.key(user_id=other_user)
        scenario.cleanups.callback(_delete_spend_rows, (own_request, other_request, null_request))
        _seed_spend_row(own_request, other_key, other_user, own_org, 1.0)
        _seed_spend_row(other_request, other_key, other_user, other_org, 2.0)
        _seed_spend_row(null_request, admin_key, admin, spend=4.0)
        for filters in ({}, {"request_id": other_request}, {"user_id": admin}):
            response = gateway.request("GET", route, params={**window, "summarize": "false", **filters}, key=admin_key)
            assert response.status_code == 200, response.text
            expected = (own_request,) if administers_org and not filters else ()
            assert _request_ids(response.json()) == expected, response.text[:1000]
            if route != "/spend/logs":
                assert response.json()["total"] == len(expected), response.text
        if route == "/spend/logs":
            summary = gateway.request("GET", route, params=window, key=admin_key)
            assert summary.status_code == 200, summary.text
            assert sum(row["spend"] for row in summary.json()) == (1.0 if administers_org else 0.0), summary.text


@pytest.mark.parametrize("route", ["/spend/logs", "/spend/logs/v2"])
def test_key_restricted_to_other_routes_still_reads_only_its_own_spend_logs(gateway: Gateway, route: str) -> None:
    own_request: Final = f"own-{uuid.uuid4().hex}"
    other_request: Final = f"other-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        restricted_key: Final = scenario.key(allowed_routes=["llm_api_routes"])
        other_key: Final = scenario.key()
        scenario.cleanups.callback(_delete_spend_rows, (own_request, other_request))
        _seed_spend_row(own_request, restricted_key)
        _seed_spend_row(other_request, other_key)

        own: Final = gateway.request("GET", route, params={"summarize": "false", **_window()}, key=restricted_key)
        assert own.status_code == 200, own.text
        assert _request_ids(own.json()) == (own_request,), own.text[:500]

        foreign: Final = gateway.request(
            "GET",
            route,
            params={**_window(), "api_key": sha256(other_key.encode()).hexdigest()},
            key=restricted_key,
        )
        assert foreign.status_code == 403, f"{foreign.status_code} {foreign.text[:300]}"

        unrelated: Final = gateway.request("GET", "/key/list", key=restricted_key)
        assert unrelated.status_code in (401, 403), f"{unrelated.status_code} {unrelated.text[:300]}"


@pytest.mark.parametrize("role", ["internal_user", "internal_user_viewer"])
@pytest.mark.parametrize("route", ["/spend/logs", "/spend/logs/v2"])
def test_internal_user_is_refused_filters_outside_their_own_spend_logs(gateway: Gateway, route: str, role: str) -> None:
    own_request: Final = f"own-{uuid.uuid4().hex}"
    other_request: Final = f"other-{uuid.uuid4().hex}"
    params: Final = {"summarize": "false", **_window()}
    with gateway.scenario() as scenario:
        own_user: Final = scenario.user(user_role=role)
        other_user: Final = scenario.user(user_role="internal_user")
        own_key: Final = scenario.key(user_id=own_user)
        other_key: Final = scenario.key(user_id=other_user)
        scenario.cleanups.callback(_delete_spend_rows, (own_request, other_request))
        _seed_spend_row(own_request, own_key, own_user)
        _seed_spend_row(other_request, other_key, other_user)

        for readable_filter in (
            {},
            {"user_id": own_user},
            {"api_key": sha256(own_key.encode()).hexdigest()},
        ):
            readable = gateway.request("GET", route, params={**params, **readable_filter}, key=own_key)
            assert readable.status_code == 200, f"{readable_filter}: {readable.text[:300]}"
            assert _request_ids(readable.json()) == (own_request,), f"{readable_filter}: {readable.text[:500]}"

        deleted_key: Final = string_value(gateway.post("/key/generate", {"user_id": own_user})["key"])
        deleted_request: Final = f"deleted-{uuid.uuid4().hex}"
        scenario.cleanups.callback(_delete_spend_rows, (deleted_request,))
        _seed_spend_row(deleted_request, deleted_key, own_user)
        gateway.post("/key/delete", {"keys": [deleted_key]})
        historical: Final = gateway.request(
            "GET", route, params={**params, "api_key": sha256(deleted_key.encode()).hexdigest()}, key=own_key
        )
        assert historical.status_code == 200, historical.text
        assert _request_ids(historical.json()) == (deleted_request,), historical.text[:500]

        for foreign_filter in (
            {"user_id": other_user},
            {"api_key": sha256(other_key.encode()).hexdigest()},
        ):
            foreign = gateway.request("GET", route, params={**params, **foreign_filter}, key=own_key)
            assert foreign.status_code == 403, f"{foreign_filter}: {foreign.status_code} {foreign.text[:300]}"
