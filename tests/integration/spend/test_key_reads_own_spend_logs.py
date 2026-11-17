import os
import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Final

import psycopg
import pytest
from integration._support.client import Gateway


def _seed_spend_row(request_id: str, key: str) -> None:
    started: Final = datetime.now(timezone.utc) - timedelta(minutes=5)
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute(
            'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, api_key, "startTime", "endTime", status) '
            "VALUES (%s, 'acompletion', %s, %s, %s, 'success')",
            (request_id, sha256(key.encode()).hexdigest(), started, started),
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
def test_key_restricted_to_other_routes_cannot_read_spend_logs(gateway: Gateway, route: str) -> None:
    with gateway.scenario() as scenario:
        restricted_key: Final = scenario.key(allowed_routes=["llm_api_routes"])
        response: Final = gateway.request("GET", route, params=_window(), key=restricted_key)
        assert response.status_code in (401, 403), f"{response.status_code} {response.text[:300]}"
