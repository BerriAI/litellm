"""
Unified /v1/messages endpoint - (Anthropic Spec)
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    create_streaming_response,
)
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.types.utils import TokenCountResponse
from litellm.proxy.anthropic_endpoints.service import AnthropicMessagesService

router = APIRouter()


@router.post(
    "/v1/messages",
    tags=["[beta] Anthropic `/v1/messages`"],
    dependencies=[Depends(user_api_key_auth)],
)
async def anthropic_response(  # noqa: PLR0915
    fastapi_response: Response,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Use `{PROXY_BASE_URL}/anthropic/v1/messages` instead - [Docs](https://docs.litellm.ai/docs/anthropic_completion).

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

    request_data = await _read_request_body(request=request)
    data: dict = {**request_data}
    try:
        data["model"] = (
            general_settings.get("completion_model", None)  # server default
            or user_model  # model name passed via cli args
            or data.get("model", None)  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        data = await add_litellm_data_to_request(
            data=data,  # type: ignore
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # override with user settings, these are params passed via cli
        if user_temperature:
            data["temperature"] = user_temperature
        if user_request_timeout:
            data["request_timeout"] = user_request_timeout
        if user_max_tokens:
            data["max_tokens"] = user_max_tokens
        if user_api_base:
            data["api_base"] = user_api_base

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if data["model"] in litellm.model_alias_map:
            data["model"] = litellm.model_alias_map[data["model"]]

        # Use service to build OpenAI-shaped request and execute call
        service = AnthropicMessagesService()
        original_anthropic_model: str = str(data.get("model", ""))
        try:
            openai_data: dict = await service.build_openai_request(
                data=data,
                user_api_key_dict=user_api_key_dict,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Failed translating Anthropic request to OpenAI format"
                },
            )

        response = await service.route_and_call(
            openai_data=openai_data,
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_dict=user_api_key_dict,
            llm_router=llm_router,
            user_model=user_model,
        )

        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        verbose_proxy_logger.debug("final response: %s", response)

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                request_data=data,
                hidden_params=hidden_params,
            )
        )

        # STREAMING PATH: delegate wrapping to the service
        if openai_data.get("stream", False) is True:
            generator = AnthropicMessagesService().wrap_openai_stream_as_anthropic(
                openai_stream=response,
                model=original_anthropic_model
                or (openai_data.get("model", "") or "unknown-model"),
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_dict=user_api_key_dict,
                request_data=openai_data,
            )

            return await create_streaming_response(
                generator=generator,
                media_type="text/event-stream",
                headers=dict(fastapi_response.headers),
            )

        # NON-STREAMING PATH
        # Run post-call hook on OpenAI response
        response = await proxy_logging_obj.post_call_success_hook(
            data=openai_data, user_api_key_dict=user_api_key_dict, response=response  # type: ignore
        )
        # Translate OpenAI response back to Anthropic format via service
        anthropic_response_payload = AnthropicMessagesService().to_anthropic_response(
            openai_response=response  # type: ignore
        )
        verbose_proxy_logger.debug(
            "\nResponse from LiteLLM (Anthropic):\n{}".format(
                anthropic_response_payload
            )
        )
        return anthropic_response_payload
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.anthropic_response(): Exception occured - {}".format(
                str(e)
            )
        )
        error_msg = f"{str(e)}"
        raise ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
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
            raise HTTPException(
                status_code=400, detail={"error": "model parameter is required"}
            )

        if not messages:
            raise HTTPException(
                status_code=400, detail={"error": "messages parameter is required"}
            )

        # Create TokenCountRequest for the internal endpoint
        from litellm.proxy._types import TokenCountRequest

        token_request = TokenCountRequest(model=model_name, messages=messages)

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
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.anthropic_endpoints.count_tokens(): Exception occurred - {}".format(
                str(e)
            )
        )
        raise HTTPException(
            status_code=500, detail={"error": f"Internal server error: {str(e)}"}
        )
