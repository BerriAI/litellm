"""
Unified /v1/messages endpoint - (Anthropic Spec)
"""

import asyncio
import json
import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
import tiktoken
from litellm import CustomStreamWrapper, cast

import litellm
import httpx
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.chat.handler import ModelResponseIterator as AnthropicModelResponseIterator
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.completion.transformation import AnthropicTextConfig
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import AnthropicAdapter, LiteLLMAnthropicMessagesAdapter
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    create_streaming_response,
)
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.types.utils import Choices, ModelResponse, ModelResponseStream, TokenCountResponse
from litellm.utils import get_non_default_params

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

        openai_request = AnthropicAdapter().translate_completion_input_params({**data})
        if openai_request is None:
            raise ValueError("Failed to translate Anthropic request to OpenAI format")
        completion_kwargs: Dict[str, Any] = dict(openai_request)

        completion_kwargs = await proxy_logging_obj.pre_call_hook(  # type: ignore
            user_api_key_dict=user_api_key_dict, data=completion_kwargs, call_type="text_completion"
        )

        tasks = []
        tasks.append(
            proxy_logging_obj.during_call_hook(
                data=completion_kwargs,
                user_api_key_dict=user_api_key_dict,
                call_type=ProxyBaseLLMRequestProcessing._get_pre_call_type(
                    route_type="anthropic_messages"  # type: ignore
                ),
            )
        )

        if "model" not in completion_kwargs or "messages" not in completion_kwargs:
            raise ValueError("Translated Anthropic request missing required fields")

        non_default_params = get_non_default_params(completion_kwargs)
        optional_params = AnthropicConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params={
                **(completion_kwargs.get("optional_params") or {})
            },
            model=data["model"],
            drop_params=bool(os.getenv("LITELLM_DROP_PARAMS", False)),
        )

        litellm_params: Dict[str, Any] = {}
        _litellm_params_value = completion_kwargs.get("litellm_params") or data.get(
            "litellm_params", {}
        )
        if isinstance(_litellm_params_value, dict):
            litellm_params = _litellm_params_value

        headers: Dict[str, Any] = {}
        _headers_value = completion_kwargs.get("headers") or data.get("headers", {})
        if isinstance(_headers_value, dict):
            headers = _headers_value

        transformed_request = AnthropicConfig().transform_request(
            model=completion_kwargs["model"],
            messages=completion_kwargs["messages"],
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        data.update(transformed_request)
        data["litellm_params"] = litellm_params
        data["headers"] = headers

        ### ROUTE THE REQUESTs ###
        router_model_names = llm_router.model_names if llm_router is not None else []

        # skip router if user passed their key
        if (
            llm_router is not None and data["model"] in router_model_names
        ):  # model in router model list
            llm_coro = llm_router.aanthropic_messages(**data)
        elif (
            llm_router is not None
            and llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):  # model set in model_group_alias
            llm_coro = llm_router.aanthropic_messages(**data)
        elif (
            llm_router is not None and data["model"] in llm_router.deployment_names
        ):  # model in router deployments, calling a specific deployment on the router
            llm_coro = llm_router.aanthropic_messages(**data, specific_deployment=True)
        elif (
            llm_router is not None and llm_router.has_model_id(data["model"])
        ):  # model in router model list
            llm_coro = llm_router.aanthropic_messages(**data)
        elif (
            llm_router is not None
            and data["model"] not in router_model_names
            and (
                llm_router.default_deployment is not None
                or len(llm_router.pattern_router.patterns) > 0
            )
        ):  # model in router deployments, calling a specific deployment on the router
            llm_coro = llm_router.aanthropic_messages(**data)
        elif user_model is not None:  # `litellm --model <your-model-name>`
            llm_coro = litellm.anthropic_messages(**data)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "completion: Invalid model name passed in model="
                    + data.get("model", "")
                },
            )

        tasks.append(llm_coro)

        # wait for call to end
        llm_responses = asyncio.gather(
            *tasks
        )  # run the moderation check in parallel to the actual llm api call

        responses = await llm_responses

        response = responses[1]

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

        json_mode: bool = optional_params.pop("json_mode", False)
        if (
            "stream" in data and data["stream"] is True
        ):  # use generate_responses to stream responses
            if isinstance(response, CustomStreamWrapper):
                raw_stream = await response.fetch_stream()
            else:
                raw_stream = response

            if isinstance(raw_stream, AnthropicModelResponseIterator):
                completion_stream = raw_stream
            else:
                completion_stream = AnthropicModelResponseIterator(
                    streaming_response=raw_stream,
                    sync_stream=False,
                    json_mode=json_mode,
                )

            if hasattr(completion_stream, "json_mode"):
                completion_stream.json_mode = json_mode

            async def openai_stream_generator():
                """Yield OpenAI-formatted chunks after proxy hooks run."""

                str_so_far = ""
                async for chunk in proxy_logging_obj.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=user_api_key_dict,
                    response=completion_stream,
                    request_data=data,
                ):
                    verbose_proxy_logger.debug(
                        "anthropic_endpoint: openai_stream_generator chunk - %s", chunk
                    )
                    processed_chunk = await proxy_logging_obj.async_post_call_streaming_hook(
                        user_api_key_dict=user_api_key_dict,
                        response=chunk,
                        data=data,
                        str_so_far=str_so_far,
                    )

                    if isinstance(processed_chunk, (ModelResponse, ModelResponseStream)):
                        response_str = litellm.get_response_string(
                            response_obj=processed_chunk
                        )
                        str_so_far += response_str

                    yield processed_chunk

            transformed_stream = AnthropicAdapter().translate_completion_output_params_streaming(
                openai_stream_generator(),
                model=data["model"],
            )

            if transformed_stream is None:
                raise ValueError(
                    "Failed to translate OpenAI stream to Anthropic stream format"
                )

            async def anthropic_sse_generator():
                async for chunk in transformed_stream:
                    yield ProxyBaseLLMRequestProcessing._process_chunk_with_cost_injection(
                        chunk=chunk,
                        model_name=data.get("model", ""),
                    )

            return await create_streaming_response(
                generator=anthropic_sse_generator(),
                media_type="text/event-stream",
                headers=dict(fastapi_response.headers),
            )


        ### CALL HOOKS ### - modify outgoing data
        model_response = litellm.ModelResponse()

        dummy_raw_response = httpx.Response(
            status_code=200,
            headers={},
            request=httpx.Request("POST", "https://dummy.local/anthropic/messages"),
            content=b"{}",
        )
        openai_response = AnthropicConfig().transform_parsed_response(
            completion_response=response,
            raw_response=dummy_raw_response,
            model_response=model_response,
            json_mode=json_mode,
        )

        response = await proxy_logging_obj.post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=openai_response # type: ignore
        )

        anthropic_response = LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(response=openai_response)

        verbose_proxy_logger.debug("\nResponse from Litellm:\n{}".format(response))
        return anthropic_response
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
                status_code=400,
                detail={"error": "model parameter is required"}
            )
        
        if not messages:
            raise HTTPException(
                status_code=400,
                detail={"error": "messages parameter is required"}
            )
        
        # Create TokenCountRequest for the internal endpoint
        from litellm.proxy._types import TokenCountRequest
        
        token_request = TokenCountRequest(
            model=model_name,
            messages=messages
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
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.anthropic_endpoints.count_tokens(): Exception occurred - {}".format(str(e))
        )
        raise HTTPException(
            status_code=500,
            detail={"error": f"Internal server error: {str(e)}"}
        )
