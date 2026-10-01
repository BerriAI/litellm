from collections.abc import Awaitable, Callable, Mapping
from typing import Final

import orjson
from fastapi import HTTPException, Request, Response
from pydantic import TypeAdapter
from starlette.types import Message

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.ip_address_utils import IPAddressUtils
from litellm.proxy.auth.resolvers.store import IdentityStore
from litellm.proxy.auth.user_api_key_auth import authorize_internal_virtual_key
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.spend_tracking.budget_reservation import release_unbound_budget_reservation
from litellm.types.utils import ModelResponse


async def validate_key(key_id: str | None) -> UserAPIKeyAuth | None:
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    if key_id is None:
        return None
    key: Final = IdentityStore.key_from_principal(
        await IdentityStore(prisma_client, user_api_key_cache, proxy_logging_obj=proxy_logging_obj).resolve(
            hashed_token=key_id
        )
    )
    if key.blocked or key.is_session_token:
        raise HTTPException(400, "Choose an active virtual key for Lens analysis")
    return key


async def complete(
    key_id: str, data: dict[str, object], reserve: Callable[[], Awaitable[None]], incoming: Request
) -> tuple[ModelResponse, float | None]:
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import llm_router, proxy_config, proxy_logging_obj, version

    payload: Final = orjson.dumps(data)
    client_ip: Final = IPAddressUtils.get_mcp_client_ip(incoming)

    body: Final[Message] = {
        "type": "http.request",
        "body": payload,
        "more_body": False,
    }
    messages: Final = iter((body,))

    async def receive() -> Message:
        message: Final = next(messages, None)
        return message if message is not None else await incoming.receive()

    request: Final = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "scheme": incoming.url.scheme or "http",
            "client": (client_ip, incoming.client.port if incoming.client else 0) if client_ip else None,
            "server": ("litellm.internal", 80),
        },
        receive=receive,
    )
    try:
        auth: Final = await authorize_internal_virtual_key(key_id, request, data)
        await reserve()
        processor: Final = ProxyBaseLLMRequestProcessing(data=data)
        fastapi_response: Final = Response()
        try:
            response: Final = TypeAdapter(ModelResponse).validate_python(
                await processor.base_process_llm_request(
                    request=request,
                    fastapi_response=fastapi_response,
                    user_api_key_dict=auth,
                    route_type="acompletion",
                    proxy_logging_obj=proxy_logging_obj,
                    general_settings=TypeAdapter(dict[str, object]).validate_python(proxy_server.general_settings),  # pyright: ignore[reportUnknownMemberType]  # Validate the legacy untyped config at the request boundary
                    proxy_config=proxy_config,
                    llm_router=llm_router,
                    version=version,
                )
            )
            billed: Final = fastapi_response.headers.get("x-litellm-response-cost")
            return response, float(billed) if billed not in (None, "", "None") else None
        except Exception as exc:
            raise await processor._handle_llm_api_exception(  # pyright: ignore[reportPrivateUsage]  # Standard proxy endpoint failure hook releases limits and records failures
                e=exc, user_api_key_dict=auth, proxy_logging_obj=proxy_logging_obj, version=version
            )
    except litellm.BudgetExceededError:
        raise HTTPException(402, "The analysis key or its owner has reached a budget limit")
    finally:
        reservation: Final = getattr(request.state, "budget_reservation", None)
        if isinstance(reservation, Mapping):
            await release_unbound_budget_reservation(TypeAdapter(dict[str, object]).validate_python(reservation))
