from fastapi import APIRouter, Depends, HTTPException, Request, Response
import json
from typing import Any, Dict

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

router = APIRouter()


async def _background_streaming_task(
    polling_id: str,
    data: dict,
    polling_handler,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    general_settings: dict,
    llm_router,
    proxy_config,
    proxy_logging_obj,
    select_data_generator,
    user_model,
    user_temperature,
    user_request_timeout,
    user_max_tokens,
    user_api_base,
    version,
):
    """
    Background task to stream response and update cache
    
    Follows OpenAI Response Streaming format:
    https://platform.openai.com/docs/api-reference/responses-streaming
    
    Processes streaming events and builds Response object:
    https://platform.openai.com/docs/api-reference/responses/object
    """
    
    try:
        verbose_proxy_logger.info(f"Starting background streaming for {polling_id}")
        
        # Update status to in_progress (OpenAI format)
        await polling_handler.update_state(
            polling_id=polling_id,
            status="in_progress",
        )
        
        # Force streaming mode and remove background flag
        data["stream"] = True
        data.pop("background", None)
        
        # Create processor
        processor = ProxyBaseLLMRequestProcessing(data=data)
        
        # Make streaming request
        response = await processor.base_process_llm_request(
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
        
        # Process streaming response following OpenAI events format
        output_items = {}  # Track output items by ID
        usage_data = None
        
        # Handle StreamingResponse
        if hasattr(response, 'body_iterator'):
            async for chunk in response.body_iterator:
                # Parse chunk
                if isinstance(chunk, bytes):
                    chunk = chunk.decode('utf-8')
                
                if isinstance(chunk, str) and chunk.startswith("data: "):
                    chunk_data = chunk[6:].strip()
                    if chunk_data == "[DONE]":
                        break
                    
                    try:
                        event = json.loads(chunk_data)
                        event_type = event.get("type", "")
                        
                        # Process different event types
                        if event_type == "response.output_item.added":
                            # New output item added
                            item = event.get("item", {})
                            item_id = item.get("id")
                            if item_id:
                                output_items[item_id] = item
                                await polling_handler.update_state(
                                    polling_id=polling_id,
                                    output_item=item,
                                )
                        
                        elif event_type == "response.content_part.added":
                            # Content part added to an output item
                            item_id = event.get("item_id")
                            output_index = event.get("output_index")
                            content_part = event.get("part", {})
                            
                            if item_id and item_id in output_items:
                                # Update the output item with new content
                                if "content" not in output_items[item_id]:
                                    output_items[item_id]["content"] = []
                                output_items[item_id]["content"].append(content_part)
                                
                                await polling_handler.update_state(
                                    polling_id=polling_id,
                                    output_item=output_items[item_id],
                                )
                        
                        elif event_type == "response.content_part.done":
                            # Content part completed
                            item_id = event.get("item_id")
                            content_part = event.get("part", {})
                            
                            if item_id and item_id in output_items:
                                # Update final content
                                output_items[item_id]["content"] = content_part.get("content", "")
                                await polling_handler.update_state(
                                    polling_id=polling_id,
                                    output_item=output_items[item_id],
                                )
                        
                        elif event_type == "response.output_item.done":
                            # Output item completed
                            item = event.get("item", {})
                            item_id = item.get("id")
                            if item_id:
                                output_items[item_id] = item
                                await polling_handler.update_state(
                                    polling_id=polling_id,
                                    output_item=item,
                                )
                        
                        elif event_type == "response.done":
                            # Response completed - includes usage
                            response_data = event.get("response", {})
                            usage_data = response_data.get("usage")
                        
                        # Handle generic response format (for non-OpenAI providers)
                        elif "output" in event:
                            output = event.get("output", [])
                            if isinstance(output, list):
                                for item in output:
                                    item_id = item.get("id")
                                    if item_id:
                                        output_items[item_id] = item
                                        await polling_handler.update_state(
                                            polling_id=polling_id,
                                            output_item=item,
                                        )
                            
                            # Check for usage in generic format
                            if "usage" in event:
                                usage_data = event.get("usage")
                        
                    except json.JSONDecodeError as e:
                        verbose_proxy_logger.warning(
                            f"Failed to parse streaming chunk: {e}"
                        )
                        pass
        
        # Mark as completed
        await polling_handler.update_state(
            polling_id=polling_id,
            status="completed",
            usage=usage_data,
        )
        
        verbose_proxy_logger.info(
            f"Completed background streaming for {polling_id}, output_items={len(output_items)}"
        )
        
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error in background streaming task for {polling_id}: {str(e)}"
        )
        import traceback
        verbose_proxy_logger.error(traceback.format_exc())
        
        await polling_handler.update_state(
            polling_id=polling_id,
            status="failed",
            error={
                "type": "internal_error",
                "message": str(e),
                "code": "background_streaming_error"
            },
        )


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
    from datetime import datetime, timezone
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
    
    # Check if polling via cache is enabled (using global config vars)
    background_mode = data.get("background", False)
    
    # Check if polling is enabled (can be "all" or a list of providers)
    should_use_polling = False
    if background_mode and polling_via_cache_enabled and redis_usage_cache:
        if polling_via_cache_enabled == "all":
            # Enable for all models/providers
            should_use_polling = True
        elif isinstance(polling_via_cache_enabled, list):
            # Check if provider is in the list (e.g., ["openai", "anthropic"])
            model = data.get("model", "")
            # Extract provider from model (e.g., "openai/gpt-4" -> "openai")
            provider = model.split("/")[0] if "/" in model else model
            if provider in polling_via_cache_enabled:
                should_use_polling = True
    
    # If all conditions are met, use polling mode
    if should_use_polling:
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
        await polling_handler.create_initial_state(
            polling_id=polling_id,
            request_data=data,
        )
        
        # Start background task to stream and update cache
        import asyncio
        asyncio.create_task(
            _background_streaming_task(
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
        return {
            "id": polling_id,
            "object": "response",
            "status": "queued",
            "output": [],
            "usage": None,
            "metadata": data.get("metadata", {}),
            "created_at": int(datetime.now(timezone.utc).timestamp()),
        }
    
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
            return {
                "id": response_id,
                "object": "response",
                "deleted": True
            }
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
