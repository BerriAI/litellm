from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from litellm.main import _CompletionDispatchContext, _CompletionDispatchResult


def complete_scaledown(
    base_llm_http_handler: "BaseLLMHTTPHandler",
    client: "HTTPHandler | AsyncHTTPHandler | None",
    encoding: "Tokenizer",
    ctx: "_CompletionDispatchContext",
) -> "_CompletionDispatchResult":
    """Send a ScaleDown completion through the shared HTTP handler.

    The OpenAI passthrough is not used because ScaleDown authenticates with x-api-key and its
    native endpoints take non-chat request bodies. Credentials and URLs are resolved by
    ScaleDownChatConfig, not here.
    """
    try:
        return base_llm_http_handler.completion(
            model=ctx.model,
            messages=ctx.messages,
            headers=ctx.headers,
            model_response=ctx.model_response,
            api_key=ctx.api_key,
            api_base=ctx.api_base,
            acompletion=ctx.acompletion,
            logging_obj=ctx.logging,
            optional_params=ctx.optional_params,
            litellm_params=ctx.litellm_params,
            shared_session=ctx.shared_session,
            timeout=ctx.timeout,
            client=client,
            custom_llm_provider=ctx.custom_llm_provider,
            encoding=encoding,
            stream=ctx.stream,
            provider_config=ctx.provider_config,
        )
    except Exception as e:
        ctx.logging.post_call(
            input=ctx.messages,
            api_key=ctx.api_key,
            original_response=str(e),
            additional_args={"headers": ctx.headers},
        )
        raise
