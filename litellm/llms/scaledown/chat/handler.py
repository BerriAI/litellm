from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from litellm.main import _CompletionDispatchContext


def complete_scaledown(
    base_llm_http_handler: Any, client: Any, encoding: Any, ctx: "_CompletionDispatchContext"
) -> Any:
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
        raise e
