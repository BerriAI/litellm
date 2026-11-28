import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import jwt
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment, string_value
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


def _model_refusal_row_landed(owned: OwnedProxy) -> None:
    with owned.gateway.scenario() as scenario:
        refused_key: Final = scenario.key(models=[f"not-{uuid.uuid4().hex}"])
        response: Final = owned.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": "integration-unrouted", "messages": [{"role": "user", "content": "expired session"}]},
            key=refused_key,
        )
        assert response.status_code == 403, response.text
        assert eventually(lambda: _failure_rows(refused_key), lambda rows: rows != [], seconds=70) == ["acompletion"]


def test_expired_dashboard_session_and_virtual_keys_write_no_failure_rows_for_dashboard_info_calls(
    owned: OwnedProxy, expired: ExpiredKeys
) -> None:
    for route in DASHBOARD_INFO_ROUTES:
        _rejected_as_expired(owned.gateway, expired.session, route)
        _rejected_as_expired(owned.gateway, expired.virtual, route)

    _model_refusal_row_landed(owned)
    assert (_failure_rows(expired.virtual), _failure_rows(expired.session)) == ([], [])


def test_an_expired_dashboard_session_key_writes_no_failure_row_for_an_llm_call(
    owned: OwnedProxy, expired: ExpiredKeys
) -> None:
    response: Final = owned.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": "integration-unrouted", "messages": [{"role": "user", "content": "expired session"}]},
        key=expired.session,
    )
    assert response.status_code == 401, response.text
    assert "Expired Key" in response.text, response.text
    _model_refusal_row_landed(owned)
    assert _failure_rows(expired.session) == []
