"""
Completion dispatch for the OpenCode providers.

Resolves the key, base URL and session header, then sends the request over the
wire format the model speaks: Anthropic Messages through the Anthropic chat
handler, which translates OpenAI input and output, or OpenAI Chat Completions.
"""

from collections.abc import Callable, Coroutine, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
import tiktoken

import litellm
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.anthropic.chat.handler import AnthropicChatCompletion
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.opencode.chat.messages_transformation import (
    OpenCodeMessagesConfig,
    is_messages_model,
)
from litellm.llms.opencode.chat.transformation import OpenCodeConfig
from litellm.llms.opencode.common_utils import (
    resolve_opencode_api_base,
    resolve_opencode_api_key,
    with_opencode_session_header,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from aiohttp import ClientSession

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


def complete_opencode(
    *,
    custom_llm_provider: str,
    model: str,
    messages: list[AllMessageValues],  # mutable-ok: both handlers take the message list as-is
    api_key: str | None,
    api_base: str | None,
    headers: Mapping[str, str] | None,
    litellm_params: dict[str, object],  # mutable-ok: both handlers take litellm_params as a dict
    optional_params: dict[str, object],  # mutable-ok: both handlers take optional_params as a dict
    model_response: ModelResponse,
    logging_obj: "LiteLLMLoggingObj",
    acompletion: bool,
    stream: bool | None,
    timeout: float | str | httpx.Timeout | None,
    client: HTTPHandler | AsyncHTTPHandler | None,
    shared_session: "ClientSession | None",
    anthropic_chat_handler: AnthropicChatCompletion,
    chat_handler: BaseLLMHTTPHandler,
    encoding: tiktoken.Encoding,
    print_verbose: Callable[[str], object],
) -> ModelResponse | CustomStreamWrapper | Coroutine[object, object, ModelResponse | CustomStreamWrapper]:
    surface: Final = "go" if custom_llm_provider == "opencode_go" else "zen"
    provider: Final = f"opencode_{surface}"
    resolved_key: Final = resolve_opencode_api_key(surface, api_key)
    resolved_base: Final = resolve_opencode_api_base(surface, api_base)
    session_headers: Final = with_opencode_session_header(
        surface,
        headers or litellm.headers or {},  # mutable-ok: empty dict fallback for headers
        litellm_params,
    )
    no_params: Final = MappingProxyType({})

    if is_messages_model(surface, model):
        # AnthropicChatCompletion builds its headers from AnthropicConfig
        # directly, and that resolves a missing key from ANTHROPIC_API_KEY, so
        # a missing OpenCode key has to be rejected here rather than sending a
        # first-party Anthropic credential to opencode.ai.
        if resolved_key is None:
            raise ValueError(
                f"OpenCode API key is required. Set OPENCODE_{surface.upper()}_API_KEY "
                f"or OPENCODE_API_KEY, or pass api_key."
            )
        return anthropic_chat_handler.completion(
            model=model,
            messages=messages,
            api_base=OpenCodeMessagesConfig(surface=surface).get_complete_url(
                api_base=resolved_base,
                api_key=resolved_key,
                model=model,
                optional_params=no_params,
                litellm_params=no_params,
            ),
            acompletion=acompletion,
            custom_prompt_dict=litellm.custom_prompt_dict,
            model_response=model_response,
            print_verbose=print_verbose,
            optional_params=optional_params,
            litellm_params=litellm_params,
            logger_fn=None,
            encoding=encoding,
            api_key=resolved_key,
            logging_obj=logging_obj,
            headers=session_headers,  # Bearer is chat-arm only; the config injects x-api-key
            timeout=timeout,  # pyright: ignore[reportArgumentType]  # the dispatch context widens timeout to str|None for every provider; narrowing it is a repo-wide gap, not opencode-specific
            client=client,
            custom_llm_provider=provider,
        )

    response: Final = chat_handler.completion(
        model=model,
        stream=stream,
        messages=messages,
        acompletion=acompletion,
        api_base=OpenCodeConfig(surface=surface).get_complete_url(
            api_base=resolved_base,
            api_key=resolved_key,
            model=model,
            optional_params=no_params,
            litellm_params=no_params,
        ),
        model_response=model_response,
        optional_params=optional_params,
        litellm_params=litellm_params,
        shared_session=shared_session,
        custom_llm_provider=provider,
        timeout=timeout,  # pyright: ignore[reportArgumentType]  # the dispatch context widens timeout to str|None for every provider; narrowing it is a repo-wide gap, not opencode-specific
        headers=(
            {**session_headers, "Authorization": f"Bearer {resolved_key}"}  # mutable-ok: handler mutates headers
            if resolved_key is not None
            else session_headers
        ),
        encoding=encoding,
        api_key=resolved_key,
        logging_obj=logging_obj,
        client=client,
    )
    logging_obj.post_call(input=messages, api_key=resolved_key, original_response=response)

    return response
