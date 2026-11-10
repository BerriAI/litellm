"""Token counting for hosted_vllm deployments through the server's own ``POST /tokenize``.

vLLM renders the chat template server-side, so the only exact count of what a request
costs is the one the server computes. The local tokenizer fallback also counts
``thinking`` blocks that the chat path drops before sending, which doubles the answer
on long agentic conversations.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from types import MappingProxyType
from typing import Final, cast

import httpx
from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # upstream helper's signature is partially untyped
)
from litellm.types.utils import (
    CountContextManagement,
    LlmProviders,
    TokenCountMessagesFormat,
    TokenCountResponse,
)

TOKENIZER_TYPE: Final = "hosted_vllm_tokenize"
HOSTED_VLLM_API_BASE_ENV: Final = "HOSTED_VLLM_API_BASE"
_OPENAI_ONLY_ROLES: Final = frozenset({"system", "developer", "tool", "function"})
_ANTHROPIC_BLOCK_TYPES: Final = frozenset(
    {"tool_use", "tool_result", "thinking", "redacted_thinking", "image", "document"}
)

_FIELDS: Final = TypeAdapter(Mapping[str, object])
_EMPTY_FIELDS: Final[Mapping[str, object]] = MappingProxyType({})


def _as_mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _FIELDS.validate_python(value)
    except ValidationError:
        return None


def tokenize_url(api_base: str) -> str:
    """``/tokenize`` lives at the server root, next to the ``/v1`` prefix a deployment's api_base names."""
    trimmed: Final = api_base.rstrip("/")
    return f"{trimmed.removesuffix('/v1')}/tokenize"


def is_anthropic_shaped(messages: Sequence[Mapping[str, object]], system: object) -> bool:
    """Whether ``messages`` use the Anthropic Messages layout rather than OpenAI chat.

    A separate ``system`` value only exists in the Anthropic layout. Otherwise the roles and
    content block types decide; plain-text turns fit both layouts and count as OpenAI chat,
    which needs no translation.
    """
    if system is not None:
        return True
    for message in messages:
        if message.get("role") in _OPENAI_ONLY_ROLES or "tool_calls" in message:
            return False
        if _has_anthropic_blocks(message):
            return True
    return False


def _block_type(block: object) -> object:
    block_fields: Final = _as_mapping(block)
    return block_fields.get("type") if block_fields is not None else None


def _has_anthropic_blocks(message: Mapping[str, object]) -> bool:
    content: Final = message.get("content")
    if not isinstance(content, list):
        return False
    blocks: Final = cast(Sequence[object], content)  # cast-ok: raw content list from the request body
    return any(_block_type(block) in _ANTHROPIC_BLOCK_TYPES for block in blocks)


def chat_request_for_tokenize(
    model: str,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]] | None,
    system: object,
    *,
    request_model: str | None = None,
) -> Mapping[str, object]:
    """The ``/tokenize`` body for ``messages``: what the chat completion path would send.

    Anthropic-shaped input goes through the same adapter translation as ``/v1/messages``,
    under the model name the client asked for (``request_model``) as that path does, and
    every input through the provider's message transform, so the count matches the prompt
    the server later renders for the real request. ``model`` is the served model name the
    tokenize call names.
    """
    from litellm.llms.anthropic.messages.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )
    from litellm.llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig
    from litellm.types.llms.anthropic import AnthropicMessagesRequest
    from litellm.types.llms.openai import AllMessageValues

    if is_anthropic_shaped(messages, system):
        anthropic_request: Final = cast(  # cast-ok: raw Anthropic-shaped request dicts from the count endpoint
            AnthropicMessagesRequest,
            {
                "model": request_model or model,
                "messages": list(messages),
                "max_tokens": 1,
                **({"system": system} if system is not None else {}),
                **({"tools": list(tools)} if tools else {}),
            },
        )
        translated, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
            anthropic_request, custom_llm_provider=LlmProviders.HOSTED_VLLM.value
        )
        translated_fields = _as_mapping(translated) or _EMPTY_FIELDS  # rebind-ok: single assignment per branch
    else:
        translated_fields = _EMPTY_FIELDS  # rebind-ok: single assignment per branch
    chat_messages, chat_tools = (
        (
            cast(  # cast-ok: adapter output follows the OpenAI chat shape
                Sequence[AllMessageValues], translated_fields.get("messages") or ()
            ),
            translated_fields.get("tools"),
        )
        if translated_fields
        else (
            cast(Sequence[AllMessageValues], messages),  # cast-ok: OpenAI chat dicts as received
            tuple(tools) if tools else None,
        )
    )
    chat_request: Final = _FIELDS.validate_python(
        HostedVLLMChatConfig().transform_request(  # pyright: ignore[reportUnknownMemberType]  # upstream transform's signature is partially untyped
            # deepcopy: the transform mutates message dicts in place; counting must not
            # change the caller's request (the local-tokenizer fallback counts it next).
            model=model,
            messages=deepcopy(list(chat_messages)),
            optional_params={},
            litellm_params={},
            headers={},
        )
    )
    body: Final[dict[str, object]] = {  # mutable-ok: JSON request body, sent once
        "model": model,
        "messages": chat_request.get("messages"),
        "add_generation_prompt": True,
    }
    if isinstance(chat_tools, Sequence) and chat_tools:
        body["tools"] = list(chat_tools)  # pyright: ignore[reportUnknownArgumentType]  # tool dicts pass through to JSON untouched
    return body


class HostedVLLMTokenCounter(BaseTokenCounter):
    """Counts with the deployment's ``/tokenize``; returns None so local counting takes over when it can't."""

    def should_use_token_counting_api(self, custom_llm_provider: str | None = None) -> bool:
        return custom_llm_provider == LlmProviders.HOSTED_VLLM.value

    async def count_tokens(
        self,
        model_to_use: str,
        messages: list[dict[str, object]] | None,  # mutable-ok: mirrors the BaseTokenCounter override contract
        contents: list[dict[str, object]] | None,  # mutable-ok: mirrors the BaseTokenCounter override contract
        deployment: dict[str, object] | None = None,  # mutable-ok: mirrors the BaseTokenCounter override contract
        request_model: str = "",
        tools: list[dict[str, object]] | None = None,  # mutable-ok: mirrors the BaseTokenCounter override contract
        system: object | None = None,
        messages_format: TokenCountMessagesFormat = "openai",
        output_format: JsonValue = None,
        context_management: CountContextManagement | None = None,
        chat_params: Mapping[str, JsonValue] = MappingProxyType({}),
    ) -> TokenCountResponse | None:
        if not messages:
            return None
        params: Final = _as_mapping((deployment or {}).get("litellm_params")) or _EMPTY_FIELDS
        api_base: Final = params.get("api_base") or os.getenv(HOSTED_VLLM_API_BASE_ENV)
        if not isinstance(api_base, str) or not api_base:
            verbose_logger.warning("hosted_vllm token counting needs an api_base; using the local tokenizer")
            return None
        api_key: Final = params.get("api_key")
        headers: Final = {"Authorization": f"Bearer {api_key}"} if isinstance(api_key, str) and api_key else {}
        try:
            # The sync message transform can fetch user-supplied file URLs
            # (_handle_pdf_url); a worker thread keeps that off the event loop.
            body: Final = await asyncio.to_thread(
                chat_request_for_tokenize, model_to_use, messages, tools, system, request_model=request_model or None
            )
            client: Final = get_async_httpx_client(llm_provider=LlmProviders.HOSTED_VLLM)
            response: Final[httpx.Response] = await client.post(  # pyright: ignore[reportUnknownMemberType]  # upstream client helper is partially untyped
                tokenize_url(api_base), json=dict(body), headers=headers
            )
            response.raise_for_status()
            raw_payload: Final = cast(object, response.json())  # cast-ok: httpx json() is untyped
            payload: Final = _as_mapping(raw_payload) or _EMPTY_FIELDS
            count: Final = payload.get("count")
            max_model_len: Final = payload.get("max_model_len")
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            verbose_logger.warning("hosted_vllm /tokenize failed (%s); using the local tokenizer", error)
            return None
        if not isinstance(count, int):
            verbose_logger.warning("hosted_vllm /tokenize returned no count; using the local tokenizer")
            return None
        return TokenCountResponse(
            total_tokens=count,
            request_model=request_model,
            model_used=model_to_use,
            tokenizer_type=TOKENIZER_TYPE,
            # the token id list is as long as the prompt; keep the scalar fields only
            original_response={"count": count, "max_model_len": max_model_len},
        )
