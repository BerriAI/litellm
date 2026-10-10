"""
USAGE AI CHAT ENDPOINTS

/usage/ai/chat - Stream AI chat responses about usage data
"""

import asyncio
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Final, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from openai._streaming import SSEDecoder
from pydantic import ConfigDict, Field, TypeAdapter

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.sse_keepalive import (
    SSE_COMMENT_PING,
    wrap_sse_stream_with_keepalive_pings,
)
from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
    USAGE_AI_TEMPERATURE,
    _ToolDef,  # pyright: ignore[reportPrivateUsage]  # Shared internal usage-chat tool schema
    get_tools_for_role,
)
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.utils import ModelResponse, ModelResponseStream

router: Final = APIRouter()


class ChatMessage(LiteLLMBaseModel):
    role: Literal["user", "assistant"]
    content: str


class UsageAIChatRequest(LiteLLMBaseModel):
    messages: list[ChatMessage] = Field(..., description="Chat messages (user/assistant history)")
    model: str | None = Field(default=None, description="Model to use for AI chat")


@dataclass(frozen=True)
class ProxyUsageChatCompletion:
    request: Request
    auth: UserAPIKeyAuth
    is_admin: bool = False

    async def _process(self, payload: Mapping[str, object]) -> object:
        from litellm.proxy.auth.user_api_key_auth import (
            enforce_key_and_fallback_model_access,  # pyright: ignore[reportUnknownVariableType]  # Shared auth helper has legacy collection annotations
            run_centralized_common_checks,
        )
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
        from litellm.proxy.common_utils.http_parsing_utils import rewrite_request_model
        from litellm.proxy.proxy_server import (
            general_settings,  # pyright: ignore[reportUnknownVariableType]  # Proxy singleton has legacy untyped settings
            llm_model_list,  # pyright: ignore[reportUnknownVariableType]  # Proxy deployment list has legacy annotations
            llm_router,
            proxy_config,
            proxy_logging_obj,
            select_data_generator,  # pyright: ignore[reportUnknownVariableType]  # Existing proxy serializer has untyped parameters
        )

        data: Final = dict(payload)
        request: Final = Request(
            {
                **self.request.scope,
                "path": "/chat/completions",
                "raw_path": b"/chat/completions",
                "state": {},
                "parsed_body": (tuple(data), data),
            },
            receive=self.request.receive,
        )
        rewrite_request_model(data, request, str(data["model"]))
        auth: Final = self.auth.model_copy(deep=True)
        processor: Final = ProxyBaseLLMRequestProcessing(data=data)
        try:
            await enforce_key_and_fallback_model_access(
                valid_token=auth,
                request_data=data,
                request=request,
                route="/chat/completions",
                llm_model_list=llm_model_list,
                llm_router=llm_router,
            )
            await run_centralized_common_checks(
                user_api_key_auth_obj=auth,
                request=request,
                request_data=data,
                route="/chat/completions",
            )
            result: Final = TypeAdapter[ModelResponse | StreamingResponse](
                ModelResponse | StreamingResponse, config=ConfigDict(arbitrary_types_allowed=True)
            ).validate_python(
                await processor.base_process_llm_request(
                    request=request,
                    fastapi_response=Response(),
                    user_api_key_dict=auth,
                    route_type="acompletion",
                    proxy_logging_obj=proxy_logging_obj,
                    general_settings=TypeAdapter(dict[str, object]).validate_python(general_settings),
                    proxy_config=proxy_config,
                    llm_router=llm_router,
                    select_data_generator=select_data_generator,  # pyright: ignore[reportUnknownArgumentType]  # Reuse the proxy streaming serializer
                )
            )
        except (asyncio.CancelledError, GeneratorExit):
            from litellm.proxy.spend_tracking.budget_reservation import (
                release_budget_reservation_on_cancel,  # pyright: ignore[reportUnknownVariableType]  # Shared cancellation helper has legacy dict annotations
            )

            await release_budget_reservation_on_cancel(auth.budget_reservation)
            raise
        except Exception as exc:
            await proxy_logging_obj.post_call_failure_hook(  # pyright: ignore[reportUnknownMemberType]  # Existing proxy failure hook is untyped
                user_api_key_dict=auth,
                original_exception=exc,
                request_data=processor.data,  # pyright: ignore[reportUnknownMemberType]  # Processor owns the enriched request data
            )
            raise
        else:
            return result

    async def complete(
        self, model: str, messages: Sequence[Mapping[str, object]], tools: Sequence[_ToolDef]
    ) -> ModelResponse:
        result: Final = await self._process(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                "temperature": USAGE_AI_TEMPERATURE,
                "drop_params": True,
            }
        )
        return TypeAdapter(ModelResponse).validate_python(result)

    async def stream(
        self, model: str, messages: Sequence[Mapping[str, object]]
    ) -> AsyncGenerator[ModelResponseStream, None]:
        result: Final = await self._process(
            {
                "model": model,
                "messages": messages,
                "stream": True,
                "tools": get_tools_for_role(self.is_admin),
                "temperature": USAGE_AI_TEMPERATURE,
                "drop_params": True,
            }
        )
        if not isinstance(result, StreamingResponse):
            raise HTTPException(status_code=502, detail="Usage AI completion did not return a stream")
        from litellm.proxy.common_request_processing import (
            _relay_late_response,  # pyright: ignore[reportPrivateUsage]  # Reuse the relay's upstream cleanup
        )

        async with aclosing(_relay_late_response(result)) as relay:
            async for event in SSEDecoder().aiter_bytes(relay):
                if event.data == "[DONE]":
                    return
                if not event.data:
                    continue
                payload = TypeAdapter(dict[str, object]).validate_json(event.data)
                if "error" in payload:
                    raise HTTPException(status_code=502, detail="Usage AI completion stream failed")
                chunk = TypeAdapter(ModelResponseStream).validate_python(payload)
                if chunk.choices:
                    yield chunk


@router.post(
    "/usage/ai/chat",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def usage_ai_chat(
    data: UsageAIChatRequest,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    AI chat about usage data. Streams SSE events with the AI response.
    The AI agent has access to tools that query aggregated daily activity data.
    """
    from litellm.proxy.management_endpoints.common_utils import (
        require_caller_user_id_for_non_admin,
        user_api_key_has_admin_view,
    )
    from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
        stream_usage_ai_chat,
    )

    is_admin: Final = user_api_key_has_admin_view(user_api_key_dict)
    if is_admin:
        user_id = user_api_key_dict.user_id
    else:
        user_id = require_caller_user_id_for_non_admin(user_api_key_dict)
    messages: Final = [{"role": m.role, "content": m.content} for m in data.messages]

    return StreamingResponse(
        wrap_sse_stream_with_keepalive_pings(
            stream_usage_ai_chat(
                messages=messages,
                model=data.model,
                user_id=user_id,
                is_admin=is_admin,
                completion=ProxyUsageChatCompletion(request=request, auth=user_api_key_dict, is_admin=is_admin),
            ),
            ping_interval_seconds=litellm.sse_keepalive_ping_interval_seconds,
            ping_chunk=SSE_COMMENT_PING,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
