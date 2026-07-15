"""
Unified /v1/messages endpoint - (Anthropic Spec)
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.anthropic_interface.exceptions import AnthropicExceptionMapping
from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.llms.anthropic.common_utils import (
    normalize_anthropic_server_side_fallbacks,
)
from litellm.llms.anthropic.experimental_pass_through.context_management import (
    AnthropicContextManagementError,
)
from litellm.llms.base_llm.guardrail_translation.utils import (
    blocked_response_usage as _blocked_response_usage,
)
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    create_response,
)
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.types.utils import TokenCountResponse

router = APIRouter()


def _strip_total_tokens_from_anthropic_response(response: Any) -> None:
    """Remove the OpenAI-flavored `usage.total_tokens` field that LiteLLM
    injects into Anthropic /v1/messages responses.

    The Anthropic /v1/messages spec only defines:
        input_tokens, output_tokens, cache_creation_input_tokens,
        cache_read_input_tokens, cache_creation.{ephemeral_5m,ephemeral_1h}
    The streaming SSE path (message_delta.usage) already does not include
    total_tokens; this brings the non-streaming path into the same shape.

    Handles both shapes returned by `base_process_llm_request`:
    - plain `dict` (most common — `AnthropicMessagesResponse` is a TypedDict
      and is `dict` at runtime)
    - Pydantic model whose `usage` attribute is dict-shaped (e.g. a
      BaseModel that holds raw Anthropic usage as a `dict[str, int]`)

    Streaming results (StreamingResponse, AsyncIterator, etc.) and Pydantic
    models with strongly-typed Usage sub-models are left untouched —
    those paths either have separate serialization handling or impose
    type constraints the helper does not try to subvert.
    """
    if response is None:
        return
    if isinstance(response, dict):
        usage = response.get("usage")
        if isinstance(usage, dict) and "total_tokens" in usage:
            usage.pop("total_tokens", None)
        return
    # Pydantic-model fallback: only mutate if `usage` is a dict.
    usage = getattr(response, "usage", None)
    if isinstance(usage, dict) and "total_tokens" in usage:
        usage.pop("total_tokens", None)


@router.post(
    "/v1/messages",
    tags=["[beta] Anthropic `/v1/messages`"],
    dependencies=[Depends(user_api_key_auth)],
)
async def anthropic_response(
    fastapi_response: Response,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Use `{PROXY_BASE_URL}/anthropic/v1/messages` instead - [Docs](https://docs.litellm.ai/docs/pass_through/anthropic_completion).

    This was a BETA endpoint that calls 100+ LLMs in the anthropic format.
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = normalize_anthropic_server_side_fallbacks(
        request_data=await _read_request_body(request=request),
        headers=dict(request.headers),
    )
    base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        result = await base_llm_response_processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="anthropic_messages",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=None,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
        # Optionally strip the non-Anthropic `usage.total_tokens` field
        # LiteLLM adds internally. Anthropic's official /v1/messages spec
        # only defines input_tokens / output_tokens / cache_*_input_tokens;
        # total_tokens is an OpenAI convention. Default off
        # (`litellm.strip_anthropic_total_tokens = False`) to preserve
        # backward compatibility for clients that currently read it; set
        # to True to align the wire response with the spec (and with the
        # streaming SSE path, which already omits total_tokens).
        # spend_logs / Prometheus still compute total internally — this
        # only affects the wire response.
        if litellm.strip_anthropic_total_tokens:
            _strip_total_tokens_from_anthropic_response(result)
        return result
    except ModifyResponseException as e:
        # Guardrail flagged content in passthrough mode - return 200 with violation message
        _data = e.request_data
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=_data,
        )

        # Create Anthropic-formatted response with violation message
        import uuid

        from litellm.types.utils import AnthropicMessagesResponse

        # Report the blocked LLM response's real token usage (carried on the
        # exception) instead of discarding it; zero for pre-call blocks.
        _usage = _blocked_response_usage(e.original_response)

        _anthropic_response = AnthropicMessagesResponse(
            id=f"msg_{str(uuid.uuid4())}",
            type="message",
            role="assistant",
            content=[{"type": "text", "text": e.message}],
            model=e.model,
            stop_reason="end_turn",
            usage=_usage,
        )

        if data.get("stream", None) is not None and data["stream"] is True:
            # For streaming, use the standard SSE data generator
            async def _passthrough_stream_generator():
                yield _anthropic_response

            selected_data_generator = ProxyBaseLLMRequestProcessing.async_sse_data_generator(
                response=_passthrough_stream_generator(),
                user_api_key_dict=user_api_key_dict,
                request_data=_data,
                proxy_logging_obj=proxy_logging_obj,
            )

            return await create_response(
                generator=selected_data_generator,
                media_type="text/event-stream",
                headers={},
            )

        return _anthropic_response
    except AnthropicContextManagementError as e:
        if e.status_code >= 500:
            # Server-side polyfill failures hit the failure hook for spend/alert
            # parity with the generic handler; 4xx validation errors do not.
            await proxy_logging_obj.post_call_failure_hook(
                user_api_key_dict=user_api_key_dict,
                original_exception=e,
                request_data=data,
            )
        body = AnthropicExceptionMapping.transform_to_anthropic_error(
            status_code=e.status_code,
            raw_message=e.message,
            request_id=request.headers.get("x-request-id"),
        )
        return JSONResponse(status_code=e.status_code, content=body)
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.anthropic_response(): Exception occured - {}".format(str(e))
        )

        # Extract model_id from request metadata (same as success path)
        litellm_metadata = data.get("litellm_metadata", {}) or {}
        model_info = litellm_metadata.get("model_info", {}) or {}
        model_id = model_info.get("id", "") or ""

        # Get headers
        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            call_id=data.get("litellm_call_id", ""),
            model_id=model_id,
            version=version,
            response_cost=0,
            model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            request_data=data,
            timeout=getattr(e, "timeout", None),
            litellm_logging_obj=None,
        )

        error_msg = f"{str(e)}"
        raise ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
            headers=headers,
        )


@router.post(
    "/v1/messages/count_tokens",
    tags=["[beta] Anthropic Messages Token Counting"],
    dependencies=[Depends(user_api_key_auth)],
)
async def count_tokens(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # Used for auth
):
    """
    Count tokens for Anthropic Messages API format.
    
    This endpoint follows the Anthropic Messages API token counting specification.
    It accepts the same parameters as the /v1/messages endpoint but returns
    token counts instead of generating a response.
    
    Example usage:
    ```
    curl -X POST "http://localhost:4000/v1/messages/count_tokens?beta=true" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer your-key" \
      -d '{
        "model": "claude-3-sonnet-20240229",
        "messages": [{"role": "user", "content": "Hello Claude!"}]
      }'
    ```
    
    Returns: {"input_tokens": <number>}
    """
    from litellm.proxy.proxy_server import token_counter as internal_token_counter

    try:
        request_data = await _read_request_body(request=request)
        data: dict = {**request_data}

        # Extract required fields
        model_name = data.get("model")
        messages = data.get("messages", [])

        if not model_name:
            raise HTTPException(status_code=400, detail={"error": "model parameter is required"})

        if not messages:
            raise HTTPException(status_code=400, detail={"error": "messages parameter is required"})

        # Create TokenCountRequest for the internal endpoint
        from litellm.proxy._types import TokenCountRequest

        token_request = TokenCountRequest(
            model=model_name,
            messages=messages,
            tools=data.get("tools"),
            system=data.get("system"),
        )

        # Call the internal token counter function with direct request flag set to False
        token_response = await internal_token_counter(
            request=token_request,
            call_endpoint=True,
        )
        _token_response_dict: dict = {}
        if isinstance(token_response, TokenCountResponse):
            _token_response_dict = token_response.model_dump()
        elif isinstance(token_response, dict):
            _token_response_dict = token_response

        # Convert the internal response to Anthropic API format
        return {"input_tokens": _token_response_dict.get("total_tokens", 0)}

    except HTTPException:
        raise
    except ProxyException as e:
        status_code = int(e.code) if e.code and e.code.isdigit() else 500
        detail = AnthropicExceptionMapping.transform_to_anthropic_error(
            status_code=status_code,
            raw_message=e.message,
        )
        raise HTTPException(
            status_code=status_code,
            detail=detail,
        )
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.anthropic_endpoints.count_tokens(): Exception occurred - {}".format(str(e))
        )
        raise HTTPException(status_code=500, detail={"error": f"Internal server error: {str(e)}"})


@router.post(
    "/api/event_logging/batch",
    tags=["[beta] Anthropic Event Logging"],
)
async def event_logging_batch(
    request: Request,
):
    """
    Stubbed endpoint for Anthropic event logging batch requests.

    This endpoint accepts event logging requests but does nothing with them.
    It exists to prevent 404 errors from Claude Code clients that send telemetry.
    """
    return {"status": "ok"}
