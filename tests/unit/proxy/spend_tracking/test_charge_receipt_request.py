import hashlib
import json
from datetime import datetime, tzinfo
from typing import Final

import jwt
import pytest
from fastapi import HTTPException, Request
from starlette.types import Message

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.spend_tracking.charge_receipt_capture import (
    ChargeReceiptRuntime,
)
from litellm.proxy.spend_tracking.charge_receipt_request import (
    authorize_charge_receipt,
    await_charge_receipt,
    bind_charge_receipt,
)
from litellm.proxy.spend_tracking.charge_receipts import ChargeReceiptStore
from tests.unit.proxy.spend_tracking._charge_receipt_cases import (
    RECEIPT,
    SETTINGS,
    STAMP,
    ScriptedQuery,
    acknowledged,
    priced_event,
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


class FrozenJWTClock(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        return STAMP.astimezone(tz) if tz is not None else STAMP.replace(tzinfo=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    (
        "none",
        "prefix",
        "stripped",
        "mounted_prefix",
        "mounted_stripped",
        "wrong_prefix",
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
        "missing",
        "oversized",
        "malformed",
    ),
)
async def test_signed_context_binds_the_original_scope_and_exact_request(
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    monkeypatch.setattr(jwt.api_jwt, "datetime", FrozenJWTClock)
    body: Final = json.dumps({"model": RECEIPT.model, "background": changed == "background"}).encode()
    issued: Final = int(STAMP.timestamp()) - (600 if changed == "expired" else 0)
    root: Final = (
        "/genai" if changed in ("prefix", "stripped", "mounted_prefix", "mounted_stripped", "wrong_prefix") else ""
    )
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
            "route": (
                "/v1/chat/completions"
                if changed == "path"
                else "/other" + path
                if changed == "wrong_prefix"
                else root + path
            ),
        },
        "wrong-signing-secret" if changed == "signature" else SETTINGS.destination_secret.get_secret_value(),
        algorithm="HS256",
    )

    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    supplied: Final = {"missing": "", "oversized": "x" * 4097, "malformed": "not-a-jwt"}.get(changed, token)
    request: Final = Request(
        request_for(path if changed.endswith("stripped") else root + path, token=supplied).scope, receive
    )
    request.scope["root_path"] = root
    request.scope["app_root_path"] = "" if changed.startswith("mounted") else root
    request.scope["headers"].append((b"host", b"gateway.example.test/v1/chat/completions?"))
    request.scope["litellm_websocket_auth"] = changed == "websocket"
    query: Final = ScriptedQuery((({"installed": True, "durable": True},),))
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    if changed in ("none", "prefix", "stripped", "mounted_prefix", "mounted_stripped"):
        await authorize_charge_receipt(request, UserAPIKeyAuth(api_key=RECEIPT.key_hash), runtime, json.loads(body))
        with pytest.raises(HTTPException) as missing:
            await await_charge_receipt(request)
        assert missing.value.status_code == 503 and missing.value.headers == {"x-should-retry": "false"}
        call: Final = Logging(
            model=RECEIPT.model,
            messages=[],
            stream=False,
            call_type="acompletion",
            start_time=STAMP,
            litellm_call_id="owned-call",
            function_id="owned-call",
        )
        bind_charge_receipt(request, call, runtime)
        capture: Final = request.state.required_charge_capture
        expected: Final = RECEIPT.model_copy(update={"event_id": capture.event_id})
        query.replies = (*query.replies, acknowledged(expected))
        await capture.capture_event(priced_event())
        await await_charge_receipt(request)
        assert json.loads(str(query.calls[1][2])) == expected.model_dump()
    else:
        with pytest.raises(HTTPException) as error:
            await authorize_charge_receipt(request, UserAPIKeyAuth(api_key=RECEIPT.key_hash), runtime, json.loads(body))
        assert error.value.status_code == 400 and not query.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path", (("GET", "/billing/receipts/capabilities"), ("HEAD", "/key/info"), ("POST", "/utils/token_counter"))
)
async def test_prefixed_non_inference_routes_do_not_require_signed_context(method: str, path: str) -> None:
    request: Final = request_for("/genai" + path, method=method)
    request.scope["root_path"] = "/genai"
    request.scope["app_root_path"] = ""
    query: Final = ScriptedQuery()
    await authorize_charge_receipt(
        request, UserAPIKeyAuth(api_key=RECEIPT.key_hash), ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query)), {}
    )
    assert not query.calls and not hasattr(request.state, "charge_receipt_context")
