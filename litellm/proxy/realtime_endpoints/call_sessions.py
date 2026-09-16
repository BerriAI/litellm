import asyncio
import base64
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, nullcontext
from contextvars import Token
from types import MappingProxyType
from typing import Final, Literal

import httpx
from fastapi import HTTPException, Request, Response, WebSocket
from pydantic import TypeAdapter
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.types import Message, Scope

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.realtime_streaming import (
    REALTIME_SESSION_SUCCESS_LOGGED_KEY,
    RealTimeStreaming,
    realtime_attachment_cleanup,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.chatgpt.codex import (
    CodexRealtimeCall,
    CodexRealtimeOffer,
    build_call_request,
    build_sideband_request,
    parse_call_response,
)
from litellm.llms.chatgpt.realtime import (
    CallAccounting,
    ChatGPTRealtime,
    configured_realtime_headers,
    realtime_endpoint,
)
from litellm.proxy._types import InternalRequestOrigin, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_resolved_model
from litellm.proxy.auth.user_api_key_auth import (
    get_api_key,
    get_api_key_from_custom_header,
    get_websocket_api_key,
    user_api_key_auth,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.common_utils.http_parsing_utils import (
    _normalize_media_type,  # pyright: ignore[reportPrivateUsage]  # reuse the shared HTTP media-type normalization contract
)
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,  # pyright: ignore[reportPrivateUsage]  # existing built-in limiter has no public alias
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage]  # existing built-in limiter has no public alias
    isolated_request_stash,
)
from litellm.proxy.hooks.realtime_call_lease import RealtimeCallLease, realtime_call_attachment
from litellm.proxy.spend_tracking.budget_reservation import (
    invalidate_budget_reservation_counters,
    release_or_invalidate_budget_reservation,
)
from litellm.types.realtime import RealtimeQueryParams
from litellm.types.router import GenericLiteLLMParams


async def supervise_codex_call(
    request: Request, call: CodexRealtimeCall, auth: UserAPIKeyAuth, lease: RealtimeCallLease | None = None
) -> None:
    with isolated_request_stash():
        await _start_codex_supervisor(request, call, auth, lease)


async def _start_codex_supervisor(
    request: Request, call: CodexRealtimeCall, auth: UserAPIKeyAuth, lease: RealtimeCallLease | None
) -> None:
    import litellm
    from litellm.proxy.realtime_endpoints.call_supervision import CALL_SUPERVISORS, CallSupervisor

    async def receive() -> Message:
        body: Final[RealtimeQueryParams] = {"model": call.alias}
        message: Final[Message] = {
            "type": "http.request",
            "body": json.dumps(body).encode(),
            "more_body": False,
        }
        return message

    async def send(_message: Message) -> None:
        return None

    supervision_owned = False  # rebind-ok: supervisor owns cleanup after construction
    effective_handler: ChatGPTRealtime | None = None  # rebind-ok: reuse hook-enriched credentials for cleanup
    sockets: Final = AsyncExitStack()
    try:
        observer_scope: Final[Scope] = {**request.scope}
        observer_request: Final = Request(observer_scope, receive=receive)
        processed, logger = await process_codex_request(
            observer_request,
            {  # mutable-ok: common request processing enriches metadata
                **build_sideband_request(call),
                "model": call.alias,
            },
            auth,
            call.alias,
            "_arealtime",
            internal_realtime_observer=True,
        )
        pinned: Final = {  # mutable-ok: logging and provider parameter contract
            **processed,
            **build_sideband_request(call),
            "extra_headers": MappingProxyType(
                {
                    **configured_realtime_headers(
                        TypeAdapter(Mapping[str, object] | None).validate_python(processed.get("extra_headers"))
                    ),
                    **configured_realtime_headers(call.extra_headers),
                }
            ),
            "litellm_metadata": {  # mutable-ok: Logging.update_from_kwargs requires a dict to retain ownership metadata
                **TypeAdapter(Mapping[str, object]).validate_python(
                    processed.get("litellm_metadata") or MappingProxyType({})
                ),
                **(
                    MappingProxyType(
                        {
                            "model_info": {  # mutable-ok: logging and cost callbacks require a concrete model-info dict
                                **litellm.get_model_info(model=call.model_id),
                                "id": call.model_id,
                            }
                        }
                    )
                    if call.model_id is not None
                    else MappingProxyType({})
                ),
            },
        }
        logger.update_from_kwargs(
            kwargs=pinned,
            model=call.model,
            user=None,
            optional_params={},  # mutable-ok: logging contract
            litellm_params={  # mutable-ok: Logging.update_from_kwargs pops metadata from its argument
                **logger.litellm_params,
                "litellm_metadata": pinned["litellm_metadata"],
                "arealtime": True,
            },
            custom_llm_provider="chatgpt",
        )
        params: Final = GenericLiteLLMParams.model_validate(pinned)
        handler: Final = ChatGPTRealtime(
            params, request.headers, TypeAdapter(Mapping[str, object]).validate_python(pinned["extra_headers"])
        )
        effective_handler = handler
        api_base: Final = ChatGPTRealtime.get_api_base(call.api_base)
        connection: Final = await handler.open_call_connection(call.model, api_base)
        sockets.push_async_callback(connection.close)

        async def close_call() -> None:
            await handler.close_call(connection, call.model, api_base)

        async def force_close_call() -> None:
            await handler.hangup_call(api_base)

        frontend_scope: Final[Scope] = {**request.scope, "type": "websocket"}
        frontend: Final = WebSocket(frontend_scope, receive=receive, send=send)
        stream: Final = RealTimeStreaming(frontend, connection, logger, model=call.model, user_api_key_dict=auth)
        supervisor: Final = CallSupervisor(
            connection,
            stream,
            logger,
            auth,
            close_call,
            force_close_call=force_close_call,
            terminal_usage_required=realtime_endpoint(call.model) == "live",
            lease=lease,
        )
        supervision_owned = True
        sockets.pop_all()
        await CALL_SUPERVISORS.start(supervisor)
    except BaseException:
        if not supervision_owned:
            try:
                fallback_handler: Final = effective_handler or ChatGPTRealtime(
                    GenericLiteLLMParams.model_validate(build_sideband_request(call)),
                    request.headers,
                    call.extra_headers,
                )
                await fallback_handler.hangup_call(ChatGPTRealtime.get_api_base(call.api_base))
            except Exception:  # noqa: BLE001  # preserve original failure without logging provider credentials
                verbose_proxy_logger.error("Realtime startup cleanup could not confirm upstream termination")
                try:
                    await invalidate_budget_reservation_counters(budget_reservation=auth.budget_reservation)
                except Exception:  # noqa: BLE001  # cleanup errors must not replace the original startup failure
                    verbose_proxy_logger.error("Realtime startup cleanup could not invalidate budget counters")
            else:
                await release_or_invalidate_budget_reservation(budget_reservation=auth.budget_reservation)
            finally:
                try:
                    await sockets.aclose()
                except Exception:  # noqa: BLE001  # socket cleanup must preserve the original startup failure
                    verbose_proxy_logger.error("Realtime startup cleanup could not close observer socket")
        raise


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


MAX_REALTIME_OFFER_BYTES: Final = 8 * 1024 * 1024


async def _cache_bounded_offer_body(request: Request) -> None:
    try:
        if int(request.headers.get("content-length", "")) > MAX_REALTIME_OFFER_BYTES:
            raise HTTPException(413, "Realtime offer exceeds the 8 MiB limit")
    except ValueError:
        pass
    if hasattr(request, "_body"):
        if len(request._body) > MAX_REALTIME_OFFER_BYTES:  # pyright: ignore[reportPrivateUsage]  # validate Starlette's cached body without consuming it again
            raise HTTPException(413, "Realtime offer exceeds the 8 MiB limit")
        return
    if request._form is not None and request._stream_consumed:  # pyright: ignore[reportPrivateUsage]  # a mixed-case empty form cache may leave the stream unread
        return
    body: Final = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_REALTIME_OFFER_BYTES:
            raise HTTPException(413, "Realtime offer exceeds the 8 MiB limit")
        body.extend(chunk)
    request._body = bytes(body)  # pyright: ignore[reportPrivateUsage]  # Starlette has no public setter for its shared body cache


async def read_codex_offer(request: Request) -> CodexRealtimeOffer:
    await _cache_bounded_offer_body(request)
    content_type: Final = request.headers.get("content-type", "")
    if _normalize_media_type(content_type) == "multipart/form-data":
        if content_type.split(";", 1)[0] != "multipart/form-data" and not await request.form():
            try:
                request._form = await MultiPartParser(request.headers, request.stream()).parse()  # pyright: ignore[reportPrivateUsage]  # Starlette exposes no setter for its shared form cache; # rebind-ok: Request.close must own and close uploaded files
                request.scope.pop("parsed_body", None)
            except MultiPartException as exc:
                raise HTTPException(400, "Invalid realtime multipart offer") from exc
        form: Final = await request.form()
        return CodexRealtimeOffer.model_validate(
            MappingProxyType({"sdp": form.get("sdp"), "session": json.loads(str(form.get("session", "{}")))})
        )
    return CodexRealtimeOffer.model_validate(await request.json())


async def process_codex_request(
    request: Request,
    data: dict[str, object],  # mutable-ok: common request processor enriches this dictionary
    auth: UserAPIKeyAuth,
    model: str,
    route_type: Literal["arealtime_calls", "_arealtime"],
    *,
    internal_realtime_observer: bool = False,
) -> tuple[dict[str, object], Logging]:  # mutable-ok: common request processor returns enriched routing arguments
    from litellm.proxy import proxy_server as server
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    processed, logging_obj = await processor.common_processing_pre_call_logic(
        request=request,
        general_settings=server.general_settings,
        user_api_key_dict=auth,
        version=server.version,
        proxy_logging_obj=server.proxy_logging_obj,
        proxy_config=server.proxy_config,
        llm_router=server.llm_router,
        user_model=server.user_model,
        user_temperature=server.user_temperature,
        user_request_timeout=server.user_request_timeout,
        user_max_tokens=server.user_max_tokens,
        user_api_base=server.user_api_base,
        model=model,
        route_type=route_type,
        **(
            MappingProxyType({"internal_realtime_observer": True})
            if internal_realtime_observer
            else MappingProxyType({})
        ),
    )
    if internal_realtime_observer:
        logging_obj.model_call_details["internal_request_origin"] = InternalRequestOrigin.REALTIME_OBSERVER
    return processed, logging_obj


async def create_codex_realtime_call(request: Request) -> Response:
    try:
        with isolated_request_stash():
            return await _create_codex_realtime_call(request)
    finally:
        await request.close()


async def _create_codex_realtime_call(request: Request) -> Response:
    from litellm.proxy import proxy_server as server

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
        azure_api_key_header=request.headers.get("api-key", ""),
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        custom_litellm_key_header=request.headers.get("x-litellm-api-key"),
    )
    selected_key, _ = get_api_key(
        request=request,
        api_key=request.headers.get("authorization", ""),
        azure_api_key_header=request.headers.get("api-key", ""),
        custom_litellm_key_header=request.headers.get("x-litellm-api-key"),
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        pass_through_endpoints=None,
        route="/v1/realtime/calls",
    )
    custom_header: Final = server.general_settings.get("litellm_key_header_name")
    owner_key: Final = (
        get_api_key_from_custom_header(request, custom_header) if isinstance(custom_header, str) else selected_key
    )
    supervision_started = False  # rebind-ok: transfer reservation ownership only after supervision is established
    call_lease: RealtimeCallLease | None = None
    lease_transferred = False  # rebind-ok: failed startup leaves the signaling task responsible for its lease
    preprocessing_started = False  # rebind-ok: only refund reservations belonging to this signaling request
    limiter: Final = server.proxy_logging_obj.get_proxy_hook("parallel_request_limiter")
    try:
        await can_key_call_resolved_model(
            model=model,
            llm_model_list=server.llm_model_list,
            valid_token=auth,
            llm_router=server.llm_router,
        )
        data: Final = build_call_request(offer, request.query_params, request.headers)
        signaling_auth: Final = auth.model_copy(update=MappingProxyType({"budget_reservation": None}))
        if isinstance(limiter, _PROXY_MaxParallelRequestsHandler) and (
            auth.max_parallel_requests is not None
            or server.general_settings.get("global_max_parallel_requests") is not None
        ):
            raise HTTPException(400, "Realtime calls with parallel limits require the V3 rate limiter")
        preprocessing_started = True
        processed, _ = await process_codex_request(request, data, signaling_auth, model, "arealtime_calls")
        if isinstance(limiter, _PROXY_MaxParallelRequestsHandler_v3):
            call_lease = limiter.transfer_realtime_call_slot(processed)
            if call_lease is not None:
                call_lease.start()
                if not await call_lease.renew():
                    raise HTTPException(503, "Realtime call quota reservation was lost")
        with isolated_request_stash():
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
                owner=hashlib.sha256(f"Bearer {owner_key}".encode()).hexdigest(),
                expires_at=time.time() + 3600,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        supervised_call: Final = call.model_copy(
            update=MappingProxyType({"usage_supervised": True, "parallel_reserved": call_lease is not None})
        )
        token: Final = encode_call(supervised_call)
        supervision_started = True
        if call_lease is None:
            await supervise_codex_call(request, supervised_call, auth)
        else:
            await supervise_codex_call(request, supervised_call, auth, call_lease)
            lease_transferred = True
        return Response(
            response.content,
            status_code=response.status_code,
            media_type="application/sdp",
            headers=MappingProxyType({"Location": f"/v1/realtime/calls/{token}"}),
        )
    finally:
        try:
            if call_lease is not None and not lease_transferred:
                await call_lease.close()
        finally:
            try:
                if preprocessing_started and isinstance(limiter, _PROXY_MaxParallelRequestsHandler_v3):
                    await asyncio.shield(
                        limiter.async_post_call_failure_hook(
                            request_data={},  # mutable-ok: existing failure-hook contract
                            original_exception=Exception("Realtime signaling completed without token usage"),
                            user_api_key_dict=auth,
                        )
                    )
            finally:
                if not supervision_started:
                    await release_or_invalidate_budget_reservation(budget_reservation=auth.budget_reservation)


async def codex_realtime_sideband(websocket: WebSocket, token: str, auth: UserAPIKeyAuth) -> None:
    import litellm
    from litellm.proxy import proxy_server as server

    protocols: Final = tuple(
        p.strip() for p in websocket.headers.get("sec-websocket-protocol", "").split(",") if p.strip()
    )
    logging_obj: Logging | None = None  # rebind-ok: cleanup needs the logger only after pre-call succeeds
    attachment_limiter: _PROXY_MaxParallelRequestsHandler | _PROXY_MaxParallelRequestsHandler_v3 | None = None
    cleanup_token: Token[Callable[[], Awaitable[None]] | None] | None = None
    try:
        try:
            api_key: Final = get_websocket_api_key(websocket)
            if not api_key:
                raise HTTPException(403, "No API key provided")
            call: Final = decode_call(token, f"Bearer {api_key}")
            await can_key_call_resolved_model(
                model=call.alias,
                llm_model_list=server.llm_model_list,
                valid_token=auth,
                llm_router=server.llm_router,
            )
        except (HTTPException, ProxyException):
            await websocket.close(code=1008, reason="Invalid realtime call")
            return

        async def receive() -> Message:
            return {  # mutable-ok: ASGI receive message
                "type": "http.request",
                "body": json.dumps({"model": call.alias}).encode(),  # mutable-ok: JSON request serialization
                "more_body": False,
            }

        request: Final = Request(
            {  # mutable-ok: Starlette stores request state in the ASGI scope
                **websocket.scope,
                "type": "http",
                "method": "POST",
                "path": websocket.scope.get("path", "/v1/realtime"),
            },
            receive=receive,
        )
        data: Final = {  # mutable-ok: common request processor enriches routing arguments
            **build_sideband_request(call),
            "model": call.alias,
            "websocket": websocket,
            "guardrails": [  # mutable-ok: guardrail processing expects a list
                name.strip() for name in websocket.query_params.get("guardrails", "").split(",") if name.strip()
            ],
        }
        limiter: Final = server.proxy_logging_obj.get_proxy_hook("parallel_request_limiter")
        if call.usage_supervised and isinstance(
            limiter, (_PROXY_MaxParallelRequestsHandler, _PROXY_MaxParallelRequestsHandler_v3)
        ):
            attachment_limiter = limiter
            if isinstance(limiter, _PROXY_MaxParallelRequestsHandler):
                limiter.begin_realtime_attachment(data)
        try:
            with realtime_call_attachment(websocket) if call.parallel_reserved else nullcontext():
                processed, logging_obj = await process_codex_request(request, data, auth, call.alias, "_arealtime")
        except Exception:  # noqa: BLE001  # custom hook exceptions must reject the connection
            verbose_proxy_logger.exception("Realtime sideband pre-call rejected")
            await websocket.close(code=1008, reason="Realtime pre-call rejected")
            return
        await websocket.accept(
            subprotocol=next((p for p in protocols if not p.startswith("openai-insecure-api-key.")), None)
        )
        if attachment_limiter is not None:
            selected_limiter: Final = attachment_limiter

            async def release_attachment() -> None:
                await selected_limiter.async_release_realtime_attachment(data, auth)

            cleanup_token = realtime_attachment_cleanup.set(release_attachment)
        await litellm._arealtime(  # pyright: ignore[reportPrivateUsage]  # dispatch for an already authorized call
            model=f"chatgpt/{call.model}",
            websocket=websocket,
            **MappingProxyType(
                {
                    key: value
                    for key, value in {  # mutable-ok: retain processed metadata with pinned routing
                        **processed,
                        **build_sideband_request(call),
                        "extra_headers": MappingProxyType(
                            {
                                **configured_realtime_headers(
                                    TypeAdapter(Mapping[str, object] | None).validate_python(
                                        processed.get("extra_headers")
                                    )
                                ),
                                **configured_realtime_headers(call.extra_headers),
                            }
                        ),
                        "websocket": websocket,
                        "user_api_key_dict": auth,
                        "chatgpt_call_accounting": CallAccounting.SUPERVISED if call.usage_supervised else None,
                    }.items()
                    if key not in ("model", "websocket")
                }
            ),
        )
    finally:
        try:
            if attachment_limiter is not None:
                await attachment_limiter.async_release_realtime_attachment(data, auth)
        finally:
            if cleanup_token is not None:
                realtime_attachment_cleanup.reset(cleanup_token)
            if logging_obj is None or not logging_obj.model_call_details.get(REALTIME_SESSION_SUCCESS_LOGGED_KEY):
                await release_or_invalidate_budget_reservation(budget_reservation=auth.budget_reservation)
