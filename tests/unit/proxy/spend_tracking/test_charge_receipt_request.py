import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, tzinfo
from typing import Final

import jwt
import pytest
from fastapi import HTTPException, Request
from starlette.types import Message

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.spend_tracking.charge_receipt_capture import (
    AuthenticatedBillingContext,
    BillingContext,
    ChargeReceiptRuntime,
)
from litellm.proxy.spend_tracking.charge_receipt_request import (
    ChargeReceiptConfigurationError,
    authorize_charge_receipt,
    await_charge_receipt,
    bind_charge_receipt,
    configured_charge_receipts,
    request_charge_capture,
    required_receipt_key,
)
from litellm.proxy.spend_tracking.charge_receipts import ChargeReceiptSettings, ChargeReceiptStore
from tests.unit.proxy.spend_tracking._charge_receipt_cases import (
    RECEIPT,
    SETTINGS,
    STAMP,
    ScriptedQuery,
    acknowledged,
    priced_event,
)


async def no_database(query: str, *parameters: object) -> Sequence[Mapping[str, object]]:
    pytest.fail("Receipt key selection must not access a database")


@pytest.mark.parametrize("representation", ("raw", "bearer", "lowercase-bearer", "hashed", "other"))
def test_receipt_policy_uses_the_authenticated_virtual_key_identity(representation: str) -> None:
    raw: Final = "sk-receipt-policy-unit-only"
    original: Final = UserAPIKeyAuth(api_key=raw)
    assert original.api_key is not None
    assert original.token == original.api_key
    inputs: Final = {
        "raw": raw,
        "bearer": "Bearer " + raw,
        "lowercase-bearer": "bearer " + raw,
        "hashed": original.api_key,
        "other": "sk-another-policy-unit-only",
    }
    runtime: Final = ChargeReceiptRuntime(
        ChargeReceiptSettings(key_hashes=frozenset((original.api_key,))),
        ChargeReceiptStore(no_database),
    )
    authenticated: Final = UserAPIKeyAuth(api_key=inputs[representation])
    assert required_receipt_key(authenticated, runtime) == (None if representation == "other" else original.api_key)
    assert required_receipt_key(authenticated, None) is None


def request_for(path: str = "/v1/responses", *, method: str = "POST", token: str = "") -> Request:
    return Request(
        {
            "type": "http",
            "method": method,
            "scheme": "https",
            "path": path,
            "query_string": b"",
            "server": ("gateway.example.test", 443),
            "headers": [(b"x-litellm-billing-context", token.encode())] if token else [],
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,method,token",
    (
        ("/v1/embeddings", "POST", ""),
        ("/v1/responses", "GET", ""),
        ("/v1/responses", "POST", ""),
        ("/v1/responses", "POST", "x" * 4097),
        ("/v1/responses", "POST", "not-a-jwt"),
    ),
)
async def test_unqualified_requests_fail_before_any_database_or_provider_work(
    path: str,
    method: str,
    token: str,
) -> None:
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(ScriptedQuery()))
    with pytest.raises(HTTPException) as error:
        await authorize_charge_receipt(
            request_for(path, method=method, token=token),
            UserAPIKeyAuth(api_key=RECEIPT.key_hash),
            runtime,
            {"model": "owned-model"},
        )
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_read_only_metadata_and_token_counting_do_not_require_charge_context() -> None:
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(ScriptedQuery()))
    auth: Final = UserAPIKeyAuth(api_key=RECEIPT.key_hash)
    for path, method in (("/billing/receipts/capabilities", "GET"), ("/utils/token_counter", "POST")):
        request: Final = request_for(path, method=method)
        await authorize_charge_receipt(request, auth, runtime, {})
        assert request_charge_capture(request) is None


@pytest.mark.asyncio
async def test_websocket_requests_cannot_escape_the_required_receipt_policy() -> None:
    request: Final = request_for()
    request.scope["litellm_websocket_auth"] = True
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(ScriptedQuery()))
    with pytest.raises(HTTPException) as error:
        await authorize_charge_receipt(request, UserAPIKeyAuth(api_key=RECEIPT.key_hash), runtime, {})
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_unbound_authenticated_context_cannot_release_a_successful_response() -> None:
    request: Final = request_for()
    request.state.charge_receipt_context = AuthenticatedBillingContext(
        BillingContext(request_id=RECEIPT.request_id),
        RECEIPT.key_hash,
    )
    with pytest.raises(HTTPException) as error:
        await await_charge_receipt(request)
    assert error.value.status_code == 503 and error.value.headers == {"x-should-retry": "false"}


@pytest.mark.asyncio
async def test_capture_cannot_be_rebound_to_another_admission() -> None:
    query: Final = ScriptedQuery()
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    logging: Final = Logging(
        model=RECEIPT.model,
        messages=[],
        stream=False,
        call_type="acompletion",
        start_time=STAMP,
        litellm_call_id="owned-call",
        function_id="owned-function",
    )
    request: Final = request_for()
    request.state.charge_receipt_context = AuthenticatedBillingContext(
        BillingContext(request_id=RECEIPT.request_id),
        RECEIPT.key_hash,
    )
    bind_charge_receipt(request, logging, runtime)
    capture: Final = request_charge_capture(request)
    assert capture is not None
    query.replies = (acknowledged(RECEIPT.model_copy(update={"event_id": capture.event_id})),)
    await capture.capture_event(priced_event())
    await await_charge_receipt(request)
    other: Final = request_for()
    other.state.charge_receipt_context = AuthenticatedBillingContext(
        BillingContext(request_id="another-request"),
        RECEIPT.key_hash,
    )
    with pytest.raises(HTTPException) as error:
        bind_charge_receipt(other, logging, runtime)
    assert error.value.status_code == 503 and request_charge_capture(other) is None
    assert len(query.calls) == 1


@pytest.mark.parametrize("runtime_present", (False, True))
def test_authenticated_requests_need_both_the_capture_runtime_and_logging_object(runtime_present: bool) -> None:
    request: Final = request_for()
    request.state.charge_receipt_context = AuthenticatedBillingContext(
        BillingContext(request_id=RECEIPT.request_id),
        RECEIPT.key_hash,
    )
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(ScriptedQuery())) if runtime_present else None
    with pytest.raises(HTTPException) as error:
        bind_charge_receipt(request, object(), runtime)
    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_receipt_startup_refuses_an_enabled_policy_without_a_writer_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in (
        ("ENABLED", "true"),
        ("GATEWAY_SCOPE", SETTINGS.gateway_scope),
        ("KEY_HASHES", '["' + RECEIPT.key_hash + '"]'),
        ("DESTINATION_URL", SETTINGS.destination_url),
        ("DESTINATION_SECRET", SETTINGS.destination_secret.get_secret_value()),
    ):
        monkeypatch.setenv("LITELLM_CHARGE_RECEIPTS_" + name, value)
    assert isinstance(await configured_charge_receipts(None), ChargeReceiptConfigurationError)
    monkeypatch.setenv("LITELLM_CHARGE_RECEIPTS_ENABLED", "false")
    assert await configured_charge_receipts(None) is None


class FrozenJWTClock(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        return STAMP.astimezone(tz) if tz is not None else STAMP.replace(tzinfo=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ("none", "body", "scope", "key", "path", "method", "window", "background"))
async def test_signed_context_binds_the_original_scope_and_exact_request(
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    monkeypatch.setattr(jwt.api_jwt, "datetime", FrozenJWTClock)
    body: Final = json.dumps({"model": RECEIPT.model, "background": changed == "background"}).encode()
    issued: Final = int(STAMP.timestamp())
    token: Final = jwt.encode(
        {
            "version": 1,
            "sub": RECEIPT.request_id,
            "aud": "litellm-charge-receipts",
            "iat": issued,
            "exp": issued + (301 if changed == "window" else 300),
            "gateway_scope": "c" * 64 if changed == "scope" else RECEIPT.gateway_scope,
            "key_hash": "c" * 64 if changed == "key" else RECEIPT.key_hash,
            "body_sha256": hashlib.sha256(b"other" if changed == "body" else body).hexdigest(),
            "method": "GET" if changed == "method" else "POST",
            "route": "/v1/chat/completions" if changed == "path" else "/v1/responses",
        },
        SETTINGS.destination_secret.get_secret_value(),
        algorithm="HS256",
    )

    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    request: Final = Request(request_for(token=token).scope, receive)
    query: Final = ScriptedQuery((({"installed": True, "durable": True},),))
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    if changed == "none":
        await authorize_charge_receipt(request, UserAPIKeyAuth(api_key=RECEIPT.key_hash), runtime, json.loads(body))
        context: Final = request.state.charge_receipt_context
        assert context.context.request_id == RECEIPT.request_id and context.key_hash == RECEIPT.key_hash
        assert len(query.calls) == 1
    else:
        with pytest.raises(HTTPException) as error:
            await authorize_charge_receipt(request, UserAPIKeyAuth(api_key=RECEIPT.key_hash), runtime, json.loads(body))
        assert error.value.status_code == 400 and not query.calls
