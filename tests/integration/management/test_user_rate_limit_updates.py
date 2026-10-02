from typing import Final
from uuid import uuid4

import httpx
import pytest
from pydantic import TypeAdapter

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from tests.integration._support.database import read_rows

_HEADER_VALUE_ADAPTER: Final[TypeAdapter[str | None]] = TypeAdapter(str | None)


def _chat(proxy: Gateway, model: str, key: str) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"user rate limit probe {uuid4().hex}"}]},
        key=key,
    )


def _assert_user_rate_limit_error(response: httpx.Response, user: str, limit_type: str) -> None:
    request_limit: Final[str | None] = _HEADER_VALUE_ADAPTER.validate_python(
        response.headers.get("x-ratelimit-user-limit-requests")
    )
    token_limit: Final[str | None] = _HEADER_VALUE_ADAPTER.validate_python(
        response.headers.get("x-ratelimit-user-limit-tokens")
    )
    context: Final = (
        f"Expected a user {limit_type} limit error for {user}, received HTTP {response.status_code} with "
        f"user limits requests={request_limit}, tokens={token_limit}: {response.text}"
    )
    assert response.status_code == 429, context
    body: Final = JSON_OBJECT.validate_json(response.content)
    error: Final = object_value(body["error"])
    message: Final = string_value(error["message"])
    assert error.get("type") == "throttling_error", context
    assert message.startswith(f"Rate limit exceeded for user: {user}. Limit type: {limit_type}. Current limit: 1,"), (
        context
    )


@pytest.mark.parametrize("field", ("tpm_limit", "rpm_limit"))
def test_user_rate_limit_lowered_on_gateway_is_enforced_by_peer(gateway: Gateway, peer: Gateway, field: str) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(tpm_limit=100000, rpm_limit=1000)
        key: Final = scenario.key(user_id=user, models=[model])

        gateway_warm: Final = _chat(gateway, model, key)
        peer_warm: Final = _chat(peer, model, key)
        assert gateway_warm.status_code == 200, (
            f"Gateway rejected the initial user-limited request: {gateway_warm.text}"
        )
        assert peer_warm.status_code == 200, f"Peer rejected the initial user-limited request: {peer_warm.text}"
        assert peer_warm.headers.get("x-ratelimit-user-limit-requests") == "1000", peer_warm.headers
        assert peer_warm.headers.get("x-ratelimit-user-limit-tokens") == "100000", peer_warm.headers

        gateway.post("/user/update", {"user_id": user, field: 1})

        expected_tpm: Final = 1 if field == "tpm_limit" else 100000
        expected_rpm: Final = 1 if field == "rpm_limit" else 1000
        rows: Final = read_rows(
            'SELECT tpm_limit, rpm_limit FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user,),
        )
        assert rows == [{"tpm_limit": expected_tpm, "rpm_limit": expected_rpm}], (
            f"User {field} update did not persist without changing the other limit: {rows!r}"
        )

        limit_type: Final = "tokens" if field == "tpm_limit" else "requests"
        peer_limited: Final = eventually(
            lambda: _chat(peer, model, key),
            lambda response: response.status_code == 429,
            seconds=10,
            return_last_on_timeout=True,
        )
        _assert_user_rate_limit_error(peer_limited, user, limit_type)


def test_user_rate_limit_explicit_null_clears_and_omitted_limit_is_untouched(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(tpm_limit=100000, rpm_limit=1)
        key: Final = scenario.key(user_id=user, models=[model])

        gateway_warm: Final = _chat(gateway, model, key)
        assert gateway_warm.status_code == 200, f"Gateway rejected the initial request under RPM 1: {gateway_warm.text}"
        peer_limited: Final = _chat(peer, model, key)
        _assert_user_rate_limit_error(peer_limited, user, "requests")

        gateway.post("/user/update", {"user_id": user, "rpm_limit": None})

        cleared_rows: Final = read_rows(
            'SELECT tpm_limit, rpm_limit FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user,),
        )
        assert cleared_rows == [{"tpm_limit": 100000, "rpm_limit": None}], (
            f"Clearing RPM changed the wrong user limits: {cleared_rows!r}"
        )
        info: Final = gateway.get("/v2/user/info", {"user_id": user})
        assert info["tpm_limit"] == 100000, f"User info omitted or changed TPM after RPM clear: {info!r}"
        assert info["rpm_limit"] is None, f"User info did not report the cleared RPM limit: {info!r}"

        gateway_after_clear: Final = _chat(gateway, model, key)
        assert gateway_after_clear.status_code == 200, (
            f"Gateway still enforced RPM after it was cleared: {gateway_after_clear.text}"
        )
        peer_after_clear: Final = eventually(
            lambda: _chat(peer, model, key),
            lambda response: response.status_code == 200,
            seconds=10,
        )
        assert peer_after_clear.status_code == 200, (
            f"Peer did not stop enforcing RPM after it was cleared: {peer_after_clear.text}"
        )

        gateway.post("/user/update", {"user_id": user, "tpm_limit": 50000})
        omitted_rows: Final = read_rows(
            'SELECT tpm_limit, rpm_limit FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user,),
        )
        assert omitted_rows == [{"tpm_limit": 50000, "rpm_limit": None}], (
            f"Omitting RPM during the TPM update changed it: {omitted_rows!r}"
        )
