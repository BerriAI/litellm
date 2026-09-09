import base64
import hashlib
import json
import time
from types import MappingProxyType
from typing import Final

import httpx
from fastapi import HTTPException, Request, Response, WebSocket

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.chatgpt.codex import (
    CodexRealtimeCall,
    CodexRealtimeOffer,
    build_call_request,
    build_sideband_request,
    parse_call_response,
)
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_resolved_model
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.spend_tracking.budget_reservation import release_or_invalidate_budget_reservation


def encode_call(call: CodexRealtimeCall) -> str:
    encrypted: Final = encrypt_value_helper(call.model_dump_json())
    return "rtc_litellm_" + base64.urlsafe_b64encode(encrypted.encode()).decode().rstrip("=")


def decode_call(token: str, authorization: str) -> CodexRealtimeCall:
    try:
        if not token.startswith("rtc_litellm_"):
            raise ValueError("Invalid call prefix")
        encoded: Final = token.removeprefix("rtc_litellm_")
        encrypted: Final = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        plaintext: Final = decrypt_value_helper(encrypted.decode(), key="codex_realtime_call")
        call: Final = CodexRealtimeCall.model_validate_json(plaintext or "")
    except (ValueError, TypeError, UnicodeError) as exc:
        raise HTTPException(403, "Invalid realtime call") from exc
    if call.expires_at < time.time() or call.owner != hashlib.sha256(authorization.encode()).hexdigest():
        raise HTTPException(403, "Invalid or expired realtime call")
    return call


async def read_codex_offer(request: Request) -> CodexRealtimeOffer:
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        form: Final = await request.form()
        return CodexRealtimeOffer.model_validate(
            MappingProxyType({"sdp": form.get("sdp"), "session": json.loads(str(form.get("session", "{}")))})
        )
    return CodexRealtimeOffer.model_validate(await request.json())


async def create_codex_realtime_call(request: Request) -> Response:
    from litellm.proxy import proxy_server as server
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    try:
        offer: Final = await read_codex_offer(request)
    except ValueError as exc:
        raise HTTPException(400, "Invalid realtime offer: expected sdp and session") from exc
    model: Final = offer.session.model
    if not model:
        raise HTTPException(400, "session.model is required")
    auth: Final = await user_api_key_auth(
        request=request,
        api_key=request.headers.get("authorization", ""),
        azure_api_key_header="",
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        custom_litellm_key_header=None,
    )
    try:
        await can_key_call_resolved_model(
            model=model,
            llm_model_list=server.llm_model_list,
            valid_token=auth,
            llm_router=server.llm_router,
        )
        data: Final = build_call_request(offer, request.query_params, request.headers)
        processor: Final = ProxyBaseLLMRequestProcessing(data=data)
        processed, _ = await processor.common_processing_pre_call_logic(
            request=request,
            general_settings=server.general_settings,
            user_api_key_dict=auth,
            version=server.version,
            proxy_logging_obj=server.proxy_logging_obj,
            proxy_config=server.proxy_config,
            user_model=server.user_model,
            user_temperature=server.user_temperature,
            user_request_timeout=server.user_request_timeout,
            user_max_tokens=server.user_max_tokens,
            user_api_base=server.user_api_base,
            model=model,
            route_type="arealtime_calls",
        )
        result: Final = await server.route_request(
            data=processed,
            route_type="arealtime_calls",
            llm_router=server.llm_router,
            user_model=server.user_model,
        )
        try:
            response: Final = await result
        except BaseLLMException as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        if not isinstance(response, httpx.Response):
            raise HTTPException(502, "Invalid realtime signaling response")
        if response.is_error:
            return Response(response.content, status_code=response.status_code, media_type="application/json")
        try:
            call: Final = parse_call_response(
                response,
                alias=model,
                owner=hashlib.sha256(request.headers.get("authorization", "").encode()).hexdigest(),
                expires_at=time.time() + 3600,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        token: Final = encode_call(call)
        return Response(
            response.content,
            status_code=response.status_code,
            media_type="application/sdp",
            headers=MappingProxyType({"Location": f"/v1/realtime/calls/{token}"}),
        )
    finally:
        await release_or_invalidate_budget_reservation(budget_reservation=auth.budget_reservation)


async def codex_realtime_sideband(websocket: WebSocket, token: str, auth: UserAPIKeyAuth) -> None:
    import litellm
    from litellm.proxy import proxy_server as server

    try:
        try:
            call: Final = decode_call(token, websocket.headers.get("authorization", ""))
            await can_key_call_resolved_model(
                model=call.alias,
                llm_model_list=server.llm_model_list,
                valid_token=auth,
                llm_router=server.llm_router,
            )
        except (HTTPException, ProxyException):
            await websocket.close(code=1008, reason="Invalid realtime call")
            return
        await websocket.accept()
        await litellm._arealtime(  # pyright: ignore[reportPrivateUsage]  # dispatch for an already authorized call
            **build_sideband_request(call),
            websocket=websocket,
            user_api_key_dict=auth,
        )
    finally:
        await release_or_invalidate_budget_reservation(budget_reservation=auth.budget_reservation)
