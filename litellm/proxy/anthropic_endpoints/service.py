"""
Service layer for Anthropic Messages endpoint logic.

This module centralizes:
- Anthropic -> OpenAI request normalization
- Routing and calling underlying LLMs
- OpenAI stream -> Anthropic SSE wrapping
- OpenAI response -> Anthropic response translation

All heavy lifting in endpoints should delegate here to improve testability
and maintainability.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, AsyncGenerator, Dict, Optional, Callable, Type
from fastapi import HTTPException

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.types.utils import ModelResponse, ModelResponseStream

# Adapters to translate Anthropic <-> OpenAI for messages API
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    AnthropicAdapter,
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)


class AnthropicMessagesService:
    """Encapsulates Anthropic Messages processing into testable units."""

    def __init__(
        self,
        *,
        anthropic_adapter_factory: Optional[Callable[[], Any]] = None,
        anthropic_messages_adapter_factory: Optional[Callable[[], Any]] = None,
        anthropic_stream_wrapper_cls: Optional[Type[Any]] = None,
    ) -> None:
        self._anthropic_adapter_factory = anthropic_adapter_factory or AnthropicAdapter
        self._anthropic_messages_adapter_factory = (
            anthropic_messages_adapter_factory or LiteLLMAnthropicMessagesAdapter
        )
        self._anthropic_stream_wrapper_cls = (
            anthropic_stream_wrapper_cls or AnthropicStreamWrapper
        )

    async def build_openai_request(
        self,
        data: Dict[str, Any],
        *,
        user_api_key_dict: UserAPIKeyAuth,
        proxy_logging_obj: Any,
    ) -> Dict[str, Any]:
        """Translate Anthropic request -> OpenAI Chat Completions shape and run pre-call hook.

        Also ensures stream options include usage when streaming is requested.
        """
        original_data = dict(data)
        adapter = self._anthropic_adapter_factory()
        openai_req = adapter.translate_completion_input_params({**original_data})
        if openai_req is None:
            raise ValueError("Failed to translate Anthropic request to OpenAI format")
        openai_data: Dict[str, Any] = {**openai_req}

        # Map anthropic stop_sequences -> openai stop if not already set
        if "stop_sequences" in original_data and "stop" not in openai_data:
            openai_data["stop"] = original_data.get("stop_sequences")

        # Normalize hooks to OpenAI contract
        openai_data = await proxy_logging_obj.pre_call_hook(  # type: ignore
            user_api_key_dict=user_api_key_dict,
            data=openai_data,
            call_type="completion",
        )

        # Ensure usage on streaming
        if openai_data.get("stream", False) is True:
            so = dict(openai_data.get("stream_options", {}) or {})
            so.setdefault("include_usage", True)
            openai_data["stream_options"] = so

        return openai_data

    async def route_and_call(
        self,
        openai_data: Dict[str, Any],
        *,
        proxy_logging_obj: Any,
        user_api_key_dict: UserAPIKeyAuth,
        llm_router: Optional[Any] = None,
        user_model: Optional[str] = None,
    ) -> Any:
        """Route and call model using OpenAI-style completion API."""
        tasks = []
        tasks.append(
            proxy_logging_obj.during_call_hook(
                data=openai_data,
                user_api_key_dict=user_api_key_dict,
                call_type="completion",
            )
        )

        router_model_names = llm_router.model_names if llm_router is not None else []

        if llm_router is not None and openai_data["model"] in router_model_names:
            llm_coro = llm_router.acompletion(**openai_data)
        elif (
            llm_router is not None
            and getattr(llm_router, "model_group_alias", None) is not None
            and openai_data["model"] in llm_router.model_group_alias
        ):
            llm_coro = llm_router.acompletion(**openai_data)
        elif llm_router is not None and openai_data["model"] in getattr(
            llm_router, "deployment_names", set()
        ):
            llm_coro = llm_router.acompletion(**openai_data, specific_deployment=True)
        elif (
            llm_router is not None
            and openai_data["model"] in llm_router.get_model_ids()
        ):
            llm_coro = llm_router.acompletion(**openai_data)
        elif (
            llm_router is not None
            and openai_data["model"] not in router_model_names
            and (
                getattr(llm_router, "default_deployment", None) is not None
                or len(
                    getattr(
                        getattr(llm_router, "pattern_router", object), "patterns", []
                    )
                )
                > 0
            )
        ):
            llm_coro = llm_router.acompletion(**openai_data)
        elif user_model is not None:
            llm_coro = litellm.acompletion(**openai_data)
        else:
            # Preserve original endpoint behavior: invalid model => HTTP 400
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "completion: Invalid model name passed in model="
                    + str(openai_data.get("model", ""))
                },
            )

        tasks.append(llm_coro)
        responses = await asyncio.gather(*tasks)
        return responses[1]

    async def wrap_openai_stream_as_anthropic(
        self,
        *,
        openai_stream: AsyncIterator[Any],
        model: str,
        proxy_logging_obj: Any,
        user_api_key_dict: UserAPIKeyAuth,
        request_data: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Apply stream hooks to OpenAI chunks and present Anthropic SSE as text iterator."""

        async def _hook_wrapped_openai_iterator():
            str_so_far = ""
            stream_iter = proxy_logging_obj.async_post_call_streaming_iterator_hook(
                user_api_key_dict=user_api_key_dict,
                response=openai_stream,
                request_data=request_data,
            )
            async for chunk in stream_iter:  # type: ignore
                modified_chunk = await proxy_logging_obj.async_post_call_streaming_hook(
                    user_api_key_dict=user_api_key_dict,
                    response=chunk,
                    data=request_data,
                    str_so_far=str_so_far,
                )
                if isinstance(modified_chunk, (ModelResponse, ModelResponseStream)):
                    try:
                        response_str = litellm.get_response_string(
                            response_obj=modified_chunk
                        )
                        str_so_far += response_str
                    except Exception:
                        pass
                yield modified_chunk  # type: ignore

        anthropic_stream = self._anthropic_stream_wrapper_cls(
            completion_stream=_hook_wrapped_openai_iterator(),
            model=model or (request_data.get("model", "") or "unknown-model"),
        ).async_anthropic_sse_wrapper()

        async for _b in anthropic_stream:
            if isinstance(_b, (bytes, bytearray)):
                yield _b.decode("utf-8", errors="ignore")
            else:
                yield str(_b)

    def to_anthropic_response(self, *, openai_response: Any) -> Dict[str, Any]:
        adapter = self._anthropic_messages_adapter_factory()
        return adapter.translate_openai_response_to_anthropic(response=openai_response)  # type: ignore

    @staticmethod
    def build_custom_headers(
        *,
        openai_response: Any,
        user_api_key_dict: UserAPIKeyAuth,
        request_data: Dict[str, Any],
        version: str,
    ) -> Dict[str, str]:
        hidden_params = getattr(openai_response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            model_id=model_id,
            cache_key=cache_key,
            api_base=api_base,
            version=version,
            response_cost=response_cost,
            request_data=request_data,
            hidden_params=hidden_params,
        )
        return headers
