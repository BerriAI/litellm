import os
from typing import TYPE_CHECKING, Final

import litellm
from litellm.rust_bridge import chat_completions as rust_chat_completions_bridge
from litellm.rust_bridge.chat_completions import rust_chat_completions_accepts
from litellm.secret_managers.main import get_secret
from litellm.types.completion import _CompletionDispatchContext, _CompletionDispatchResult

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_OLLAMA_CHAT_PROVIDER: Final = "ollama_chat"
_DEFAULT_OLLAMA_API_BASE: Final = "http://localhost:11434"


def ollama_chat_completion(ctx: _CompletionDispatchContext) -> _CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    client: Final = ctx.client
    headers = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params = ctx.optional_params
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    # Host config resolution: the litellm module attributes and secrets the Rust
    # core cannot see. The core's own env fallback (OLLAMA_API_BASE /
    # OLLAMA_API_KEY) covers the cases the host did not resolve, the same split
    # the Anthropic path uses.
    api_base = litellm.api_base or api_base or get_secret("OLLAMA_API_BASE") or _DEFAULT_OLLAMA_API_BASE
    api_key = api_key or litellm.ollama_key or os.environ.get("OLLAMA_API_KEY") or litellm.api_key
    if api_key is not None and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {api_key}"

    # `optional_params` is already Ollama-mapped by `get_optional_params`
    # (`OllamaChatConfig.map_openai_params` runs there for every provider before
    # dispatch), so the Rust gate receives provider-named keys, as the route
    # contract requires. Asked before either path commits so `pre_call` fires
    # exactly once, on whichever path actually runs.
    serves_via_rust: Final = rust_chat_completions_accepts(
        model=model,
        messages=messages,
        optional_params=optional_params,
        custom_llm_provider=_OLLAMA_CHAT_PROVIDER,
        litellm_params=litellm_params,
        stream=stream,
    )
    if serves_via_rust:
        rust_logging_args: Final = {  # mutable-ok: logging callbacks read additional_args as a plain dict
            "complete_input_dict": {  # mutable-ok: same, and it is serialized alongside its parent
                "model": model,
                "messages": messages,
                **optional_params,
            },
            "api_base": api_base,
            "headers": headers,
        }
        logging.pre_call(input=messages, api_key=api_key, additional_args=rust_logging_args)
        log_rust_post_call: Final = rust_chat_completions_bridge.response_logger(
            logging_obj=logging,
            messages=messages,
            api_key=api_key or "",
            additional_args=rust_logging_args,
        )
        if acompletion is True:
            return rust_chat_completions_bridge.achat_completions_or_fallback(
                model=model,
                messages=messages,
                optional_params=optional_params,
                model_response=model_response,
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=_OLLAMA_CHAT_PROVIDER,
                extra_headers=headers,
                timeout=timeout,
                on_response=log_rust_post_call,
                # The bridge awaits this and only runs it when the Rust async
                # path declines before the provider is called. pre_call already
                # fired above for this request, so the fallback skips the base
                # handler's pre_call; this is the same attempt continuing.
                python_fallback=lambda: _python_completion(
                    acompletion=True,
                    api_base=api_base,
                    api_key=api_key,
                    client=client,
                    headers=headers,
                    litellm_params=litellm_params,
                    logging=logging,
                    messages=messages,
                    model=model,
                    model_response=model_response,
                    optional_params=optional_params,
                    shared_session=shared_session,
                    stream=stream,
                    timeout=timeout,
                    skip_pre_call_logging=True,
                ),
            )
        rust_response: Final = rust_chat_completions_bridge.chat_completions(
            model=model,
            messages=messages,
            optional_params=optional_params,
            model_response=model_response,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=_OLLAMA_CHAT_PROVIDER,
            extra_headers=headers,
            timeout=timeout,
            on_response=log_rust_post_call,
        )
        if rust_response is not None:
            return rust_response
        # Reaching here means the synchronous Rust attempt declined at call time,
        # before the provider was called, and already logged this request. That
        # is the same attempt continuing, so the fallback skips the base handler's
        # pre_call. The decline is reachable because the gate runs the Rust
        # `decline` capability check while the call runs the full provider call.

    return _python_completion(
        acompletion=acompletion,
        api_base=api_base,
        api_key=api_key,
        client=client,
        headers=headers,
        litellm_params=litellm_params,
        logging=logging,
        messages=messages,
        model=model,
        model_response=model_response,
        optional_params=optional_params,
        shared_session=shared_session,
        stream=stream,
        timeout=timeout,
        skip_pre_call_logging=serves_via_rust,
    )


def _python_completion(
    *,
    acompletion: bool,
    api_base: str | None,
    api_key: str | None,
    client: object,
    headers: dict,
    litellm_params: dict,
    logging: LiteLLMLoggingObj,
    messages: list,
    model: str,
    model_response: object,
    optional_params: dict,
    shared_session: object,
    stream: object,
    timeout: object,
    skip_pre_call_logging: bool = False,
) -> _CompletionDispatchResult:
    from litellm.main import _get_encoding, base_llm_http_handler

    return base_llm_http_handler.completion(
        model=model,
        stream=stream,
        messages=messages,
        acompletion=acompletion,
        api_base=api_base,
        model_response=model_response,
        optional_params=optional_params,
        litellm_params=litellm_params,
        shared_session=shared_session,
        custom_llm_provider=_OLLAMA_CHAT_PROVIDER,
        timeout=timeout,
        headers=headers,
        encoding=_get_encoding(),
        api_key=api_key,
        logging_obj=logging,
        client=client,
        skip_pre_call_logging=skip_pre_call_logging,
    )
