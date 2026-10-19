import hashlib
import json
from datetime import datetime, tzinfo
from typing import Final

import jwt
import pytest
from fastapi import HTTPException, Request
from starlette.types import Message

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.spend_tracking.charge_receipt_capture import (
    BillingContext,
    ChargeReceiptRuntime,
)
from litellm.proxy.spend_tracking.charge_receipt_request import (
    authorize_charge_receipt,
    await_charge_receipt,
)
from litellm.proxy.spend_tracking.charge_receipts import ChargeReceiptStore
from tests.unit.proxy.spend_tracking._charge_receipt_cases import (
    RECEIPT,
    SETTINGS,
    STAMP,
    ScriptedQuery,
)


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
async def test_unbound_authenticated_context_cannot_release_a_successful_response() -> None:
    request: Final = request_for()
    request.state.charge_receipt_context = BillingContext(request_id=RECEIPT.request_id, key_hash=RECEIPT.key_hash)
    with pytest.raises(HTTPException) as error:
        await await_charge_receipt(request)
    assert error.value.status_code == 503 and error.value.headers == {"x-should-retry": "false"}


class FrozenJWTClock(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        return STAMP.astimezone(tz) if tz is not None else STAMP.replace(tzinfo=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    (
        "none",
        "body",
        "scope",
        "key",
        "path",
        "method",
        "window",
        "background",
        "signature",
        "expired",
        "unsupported",
        "websocket",
    ),
)
async def test_signed_context_binds_the_original_scope_and_exact_request(
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    monkeypatch.setattr(jwt.api_jwt, "datetime", FrozenJWTClock)
    body: Final = json.dumps({"model": RECEIPT.model, "background": changed == "background"}).encode()
    issued: Final = int(STAMP.timestamp()) - (600 if changed == "expired" else 0)
    path: Final = "/v1/embeddings" if changed == "unsupported" else "/v1/responses"
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
            "route": "/v1/chat/completions" if changed == "path" else path,
        },
        "wrong-signing-secret" if changed == "signature" else SETTINGS.destination_secret.get_secret_value(),
        algorithm="HS256",
    )

    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    request: Final = Request(request_for(path, token=token).scope, receive)
    request.scope["litellm_websocket_auth"] = changed == "websocket"
    query: Final = ScriptedQuery((({"installed": True, "durable": True},),))
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    if changed == "none":
        await authorize_charge_receipt(request, UserAPIKeyAuth(api_key=RECEIPT.key_hash), runtime, json.loads(body))
        context: Final = request.state.charge_receipt_context
        assert context.request_id == RECEIPT.request_id and context.key_hash == RECEIPT.key_hash
        assert len(query.calls) == 1
    else:
        with pytest.raises(HTTPException) as error:
            await authorize_charge_receipt(request, UserAPIKeyAuth(api_key=RECEIPT.key_hash), runtime, json.loads(body))
        assert error.value.status_code == 400 and not query.calls
