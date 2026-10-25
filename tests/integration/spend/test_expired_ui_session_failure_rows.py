import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import jwt
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment, held, string_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process

from litellm.proxy._types import hash_token

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 240), pytest.mark.shares_worker_state]

SESSION_SECONDS: Final = 3
DASHBOARD_INFO_ROUTES: Final = (
    "/key/list",
    "/v2/user/info",
    "/team/list",
    "/organization/list",
    "/v2/model/info",
    "/get/user_banner",
)
SPEND_LOG_FLUSH_SECONDS: Final = 25


@dataclass(frozen=True, slots=True)
class ExpiredKeys:
    session: str
    virtual: str


@pytest.fixture(scope="module")
def owned(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    with (
        gateway_from_environment() as base,
        owned_proxy_process(
            base,
            tmp_path_factory.mktemp("expired-ui-session"),
            {"LITELLM_UI_SESSION_DURATION": f"{SESSION_SECONDS}s", "LITELLM_DISABLE_NO_REDIS_WARNING": "true"},
            workers=2,
        ) as proxy,
    ):
        yield proxy


def _session_key(gateway: Gateway) -> str:
    response: Final = gateway.client.post("/v2/login", json={"username": "admin", "password": gateway.key})
    assert response.status_code == 200, response.text
    claims: Final = jwt.decode(string_value(response.json()["token"]), options={"verify_signature": False})
    key: Final = string_value(claims["key"])
    rows: Final = read_rows('SELECT team_id FROM "LiteLLM_VerificationToken" WHERE token = %s', (hash_token(key),))
    assert [row["team_id"] for row in rows] == ["litellm-dashboard"], rows
    return key


def _rejected_as_expired(gateway: Gateway, key: str, route: str) -> None:
    response: Final = eventually(
        lambda: gateway.request("GET", route, key=key),
        lambda reply: reply.status_code == 401,
        seconds=30,
    )
    assert "Expired Key" in response.text, response.text


def _failure_rows(key: str) -> list[str]:
    rows: Final = read_rows(
        'SELECT call_type FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND status = %s',
        (hash_token(key), "failure"),
    )
    return sorted(string_value(row["call_type"]) for row in rows)


@pytest.fixture
def expired(owned: OwnedProxy) -> Iterator[ExpiredKeys]:
    with owned.gateway.scenario() as scenario:
        keys: Final = ExpiredKeys(
            session=_session_key(owned.gateway),
            virtual=scenario.key(duration="1s", key_alias=f"integration-expired-{uuid.uuid4().hex[:12]}"),
        )
        _rejected_as_expired(owned.gateway, keys.session, "/key/info")
        _rejected_as_expired(owned.gateway, keys.virtual, "/key/info")
        yield keys


def test_an_expired_dashboard_session_key_writes_no_failure_rows_for_dashboard_info_calls(
    owned: OwnedProxy, expired: ExpiredKeys
) -> None:
    for route in DASHBOARD_INFO_ROUTES:
        _rejected_as_expired(owned.gateway, expired.session, route)
        _rejected_as_expired(owned.gateway, expired.virtual, route)

    virtual_rows: Final = eventually(
        lambda: _failure_rows(expired.virtual),
        lambda rows: len(rows) == len(DASHBOARD_INFO_ROUTES) + 1,
        seconds=70,
    )
    assert virtual_rows == sorted(("/key/info", *DASHBOARD_INFO_ROUTES))
    assert (
        held(
            lambda: _failure_rows(expired.session),
            lambda rows: rows == [],
            holding=SPEND_LOG_FLUSH_SECONDS,
            seconds=SPEND_LOG_FLUSH_SECONDS + 5,
        )
        == []
    )


def test_an_expired_dashboard_session_key_still_logs_a_failed_llm_call(owned: OwnedProxy, expired: ExpiredKeys) -> None:
    response: Final = owned.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": "integration-unrouted", "messages": [{"role": "user", "content": "expired session"}]},
        key=expired.session,
    )
    assert response.status_code == 401, response.text
    assert "Expired Key" in response.text, response.text
    assert (
        eventually(
            lambda: _failure_rows(expired.session),
            lambda rows: "acompletion" in rows,
            seconds=70,
        ).count("acompletion")
        == 1
    )


@pytest.mark.parametrize("key_prefix", ["sk-", "wordlist-"])
def test_unrecognized_keys_do_not_fill_spend_tables(gateway: Gateway, key_prefix: str) -> None:
    attempts: Final = tuple(
        (route, key_prefix + uuid.uuid4().hex)
        for route in ("/v1/models", "/v1/chat/completions", "/v1/responses", "/v1/messages")
    )
    for route, key in attempts:
        response: Final = gateway.request(
            "GET" if route == "/v1/models" else "POST",
            route,
            {
                "model": "integration-unrouted",
                "messages": [{"role": "user", "content": "invalid key"}],
                "input": "invalid key",
                "max_tokens": 10,
            },
            key=key,
        )
        assert response.status_code == 401, response.text

    with gateway.scenario() as scenario:
        known_key: Final = scenario.key(duration="1s")
        _rejected_as_expired(gateway, known_key, "/v1/models")
        assert len(eventually(lambda: _failure_rows(known_key), lambda rows: len(rows) == 1, seconds=70)) == 1

        keys: Final = [key for _, key in attempts] + [hash_token(key) for _, key in attempts]
        empty: Final = [{"spend_logs": 0, "daily_user": 0, "daily_team": 0}]
        assert (
            held(
                lambda: read_rows(
                    'SELECT (SELECT count(*)::int FROM "LiteLLM_SpendLogs" WHERE api_key = ANY(%s)) AS spend_logs, '
                    '(SELECT count(*)::int FROM "LiteLLM_DailyUserSpend" WHERE api_key = ANY(%s)) AS daily_user, '
                    '(SELECT count(*)::int FROM "LiteLLM_DailyTeamSpend" WHERE api_key = ANY(%s)) AS daily_team',
                    (keys, keys, keys),
                ),
                lambda rows: rows == empty,
                holding=SPEND_LOG_FLUSH_SECONDS,
                seconds=SPEND_LOG_FLUSH_SECONDS + 5,
            )
            == empty
        )
