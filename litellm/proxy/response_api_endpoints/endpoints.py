import asyncio
from typing import Any, AsyncIterator, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.types.responses.main import DeleteResponseResult

router = APIRouter()


@router.post(
    "/v1/responses",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.post(
    "/responses",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.post(
    "/openai/v1/responses",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
async def responses_api(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Follows the OpenAI Responses API spec: https://platform.openai.com/docs/api-reference/responses

    Supports background mode with polling_via_cache for partial response retrieval.
    When background=true and polling_via_cache is enabled, returns a polling_id immediately
    and streams the response in the background, updating Redis cache.

    ```bash
    # Normal request
    curl -X POST http://localhost:4000/v1/responses \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-1234" \
    -d '{
        "model": "gpt-4o",
        "input": "Tell me about AI"
    }'

    # Background request with polling
    curl -X POST http://localhost:4000/v1/responses \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-1234" \
    -d '{
        "model": "gpt-4o",
        "input": "Tell me about AI",
        "background": true
    }'
    ```
    """
    from litellm.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        polling_cache_ttl,
        polling_via_cache_enabled,
        proxy_config,
        proxy_logging_obj,
        redis_usage_cache,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = await _read_request_body(request=request)
    
    # Check if polling via cache should be used for this request
    from litellm.proxy.response_polling.polling_handler import (
        should_use_polling_for_request,
    )
    
    should_use_polling = should_use_polling_for_request(
        background_mode=data.get("background", False),
        polling_via_cache_enabled=polling_via_cache_enabled,
        redis_cache=redis_usage_cache,
        model=data.get("model", ""),
        llm_router=llm_router,
    )
    
    # If polling is enabled, use polling mode
    if should_use_polling:
        from litellm.proxy.response_polling.background_streaming import (
            background_streaming_task,
        )
        from litellm.proxy.response_polling.polling_handler import (
            ResponsePollingHandler,
        )
        
        verbose_proxy_logger.info(
            f"Starting background response with polling for model={data.get('model')}"
        )
        
        # Initialize polling handler with configured TTL (from global config)
        polling_handler = ResponsePollingHandler(
            redis_cache=redis_usage_cache,
            ttl=polling_cache_ttl  # Global var set at startup
        )
        
        # Generate polling ID
        polling_id = ResponsePollingHandler.generate_polling_id()
        
        # Create initial state in Redis
        initial_state = await polling_handler.create_initial_state(
            polling_id=polling_id,
            request_data=data,
        )
        
        # Start background task to stream and update cache
        asyncio.create_task(
            background_streaming_task(
                polling_id=polling_id,
                data=data.copy(),
                polling_handler=polling_handler,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                general_settings=general_settings,
                llm_router=llm_router,
                proxy_config=proxy_config,
                proxy_logging_obj=proxy_logging_obj,
                select_data_generator=select_data_generator,
                user_model=user_model,
                user_temperature=user_temperature,
                user_request_timeout=user_request_timeout,
                user_max_tokens=user_max_tokens,
                user_api_base=user_api_base,
                version=version,
            )
        )
        
        # Return OpenAI Response object format (initial state)
        # https://platform.openai.com/docs/api-reference/responses/object
        return initial_state
    
    # Normal response flow
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aresponses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/cursor/chat/completions",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
async def cursor_chat_completions(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Cursor-specific endpoint that accepts Responses API input format but returns chat completions format.
    
    This endpoint handles requests from Cursor IDE which sends Responses API format (`input` field)
    but expects chat completions format response (`choices`, `messages`, etc.).
    
    ```bash
    curl -X POST http://localhost:4000/cursor/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-1234" \
    -d '{
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "Hello"}]
    }'
    Responds back in chat completions format.
    ```
    """
    from litellm.completion_extras.litellm_responses_transformation.handler import (
        responses_api_bridge,
    )
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.proxy.proxy_server import (
        _read_request_body,
        async_data_generator,
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
    from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
    from litellm.types.llms.openai import ResponsesAPIResponse
    from litellm.types.utils import ModelResponse

    data = await _read_request_body(request=request)
    
    # Convert 'messages' to 'input' for Responses API compatibility
    # Cursor sends 'messages' but Responses API expects 'input'
    if "messages" in data and "input" not in data:
        data["input"] = data.pop("messages")
    
    processor = ProxyBaseLLMRequestProcessing(data=data)

    def cursor_data_generator(response, user_api_key_dict, request_data):
        """
        Custom generator that transforms Responses API streaming chunks to chat completion chunks.
        
        This generator is used for the cursor endpoint to convert Responses API format responses
        to chat completion format that Cursor IDE expects.
        
        Args:
            response: The streaming response (BaseResponsesAPIStreamingIterator or other)
            user_api_key_dict: User API key authentication dict
            request_data: Request data containing model, logging_obj, etc.
        
        Returns:
            Async generator that yields SSE-formatted chat completion chunks
        """
        # If response is a BaseResponsesAPIStreamingIterator, transform it first
        if isinstance(response, BaseResponsesAPIStreamingIterator):
            # Transform Responses API iterator to chat completion iterator
            # Cast to AsyncIterator[str] since BaseResponsesAPIStreamingIterator implements __aiter__/__anext__
            completion_stream = responses_api_bridge.transformation_handler.get_model_response_iterator(
                streaming_response=cast(AsyncIterator[str], response),
                sync_stream=False,
                json_mode=False,
            )
            # Wrap in CustomStreamWrapper to get the async generator
            logging_obj = request_data.get("litellm_logging_obj")
            streamwrapper = CustomStreamWrapper(
                completion_stream=completion_stream,
                model=request_data.get("model", ""),
                custom_llm_provider=None,
                logging_obj=logging_obj,
            )
            # Use async_data_generator to format as SSE
            return async_data_generator(
                response=streamwrapper,
                user_api_key_dict=user_api_key_dict,
                request_data=request_data,
            )
        # Otherwise, use the default generator
        return async_data_generator(
            response=response,
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
        )

    try:
        response = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aresponses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=cursor_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )

        # Transform non-streaming Responses API response to chat completions format
        if isinstance(response, ResponsesAPIResponse):
            logging_obj = processor.data.get("litellm_logging_obj")
            transformed_response = responses_api_bridge.transformation_handler.transform_response(
                model=processor.data.get("model", ""),
                raw_response=response,
                model_response=ModelResponse(),
                logging_obj=cast(Any, logging_obj),
                request_data=processor.data,
                messages=processor.data.get("input", []),
                optional_params={},
                litellm_params={},
                encoding=None,
                api_key=None,
                json_mode=None,
            )
            return transformed_response

        # Streaming responses are already transformed by cursor_select_data_generator
        return response
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.get(
    "/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.get(
    "/openai/v1/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
async def get_response(
    response_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get a response by ID.
    
    Supports both:
    - Polling IDs (litellm_poll_*): Returns cumulative cached content from background responses
    - Provider response IDs: Passes through to provider API
    
    Follows the OpenAI Responses API spec: https://platform.openai.com/docs/api-reference/responses/get
    
    ```bash
    # Get polling response
    curl -X GET http://localhost:4000/v1/responses/litellm_poll_abc123 \
    -H "Authorization: Bearer sk-1234"
    
    # Get provider response
    curl -X GET http://localhost:4000/v1/responses/resp_abc123 \
    -H "Authorization: Bearer sk-1234"
    ```
    """
    from litellm.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        redis_usage_cache,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )
    from litellm.proxy.response_polling.polling_handler import ResponsePollingHandler

    # Check if this is a polling ID
    if ResponsePollingHandler.is_polling_id(response_id):
        # Handle polling response
        if not redis_usage_cache:
            raise HTTPException(
                status_code=500,
                detail="Redis cache not configured. Polling requires Redis."
            )
        
        polling_handler = ResponsePollingHandler(redis_cache=redis_usage_cache)
        
        # Get current state from cache
        state = await polling_handler.get_state(response_id)
        
        if not state:
            raise HTTPException(
                status_code=404,
                detail=f"Polling response {response_id} not found or expired"
            )
        
        # Return the whole state directly (OpenAI Response object format)
        # https://platform.openai.com/docs/api-reference/responses/object
        return state
    
    # Normal provider response flow
    data = await _read_request_body(request=request)
    data["response_id"] = response_id
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aget_responses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.delete(
    "/v1/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.delete(
    "/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.delete(
    "/openai/v1/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
async def delete_response(
    response_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a response by ID.
    
    Supports both:
    - Polling IDs (litellm_poll_*): Deletes from Redis cache
    - Provider response IDs: Passes through to provider API
    
    Follows the OpenAI Responses API spec: https://platform.openai.com/docs/api-reference/responses/delete
    
    ```bash
    curl -X DELETE http://localhost:4000/v1/responses/resp_abc123 \
    -H "Authorization: Bearer sk-1234"
    ```
    """
    from litellm.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        redis_usage_cache,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )
    from litellm.proxy.response_polling.polling_handler import ResponsePollingHandler

    # Check if this is a polling ID
    if ResponsePollingHandler.is_polling_id(response_id):
        # Handle polling response deletion
        if not redis_usage_cache:
            raise HTTPException(
                status_code=500,
                detail="Redis cache not configured."
            )
        
        polling_handler = ResponsePollingHandler(redis_cache=redis_usage_cache)
        
        # Get state to verify access
        state = await polling_handler.get_state(response_id)
        
        if not state:
            raise HTTPException(
                status_code=404,
                detail=f"Polling response {response_id} not found"
            )
        
        # Delete from cache
        success = await polling_handler.delete_polling(response_id)
        
        if success:
            return DeleteResponseResult(
                id=response_id,
                object="response",
                deleted=True
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete polling response"
            )
    
    # Normal provider response flow
    data = await _read_request_body(request=request)
    data["response_id"] = response_id
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="adelete_responses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/responses/{response_id}/input_items",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.get(
    "/responses/{response_id}/input_items",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.get(
    "/openai/v1/responses/{response_id}/input_items",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
async def get_response_input_items(
    response_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List input items for a response."""
    from litellm.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = await _read_request_body(request=request)
    data["response_id"] = response_id
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="alist_input_items",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/responses/{response_id}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.post(
    "/responses/{response_id}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.post(
    "/openai/v1/responses/{response_id}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
async def cancel_response(
    response_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Cancel a response by ID.
    
    Supports both:
    - Polling IDs (litellm_poll_*): Cancels background response and updates status in Redis
    - Provider response IDs: Passes through to provider API
    
    Follows the OpenAI Responses API spec: https://platform.openai.com/docs/api-reference/responses/cancel
    
    ```bash
    # Cancel polling response
    curl -X POST http://localhost:4000/v1/responses/litellm_poll_abc123/cancel \
    -H "Authorization: Bearer sk-1234"
    
    # Cancel provider response
    curl -X POST http://localhost:4000/v1/responses/resp_abc123/cancel \
    -H "Authorization: Bearer sk-1234"
    ```
    """
    from litellm.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        redis_usage_cache,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )
    from litellm.proxy.response_polling.polling_handler import ResponsePollingHandler

    # Check if this is a polling ID
    if ResponsePollingHandler.is_polling_id(response_id):
        # Handle polling response cancellation
        if not redis_usage_cache:
            raise HTTPException(
                status_code=500,
                detail="Redis cache not configured."
            )
        
        polling_handler = ResponsePollingHandler(redis_cache=redis_usage_cache)
        
        # Get current state to verify it exists
        state = await polling_handler.get_state(response_id)
        
        if not state:
            raise HTTPException(
                status_code=404,
                detail=f"Polling response {response_id} not found"
            )
        
        # Cancel the polling response (sets status to "cancelled")
        success = await polling_handler.cancel_polling(response_id)
        
        if success:
            # Fetch the updated state with cancelled status
            updated_state = await polling_handler.get_state(response_id)
            
            # Return the whole state directly (now with status="cancelled")
            return updated_state
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to cancel polling response"
            )
    
    # Normal provider response flow
    data = await _read_request_body(request=request)
    data["response_id"] = response_id
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acancel_responses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
