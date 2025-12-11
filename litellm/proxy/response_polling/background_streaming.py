"""
Background Streaming Task for Polling Via Cache Feature

Handles streaming responses from LLM providers and updates Redis cache
with partial results for polling.

Follows OpenAI Response Streaming format:
https://platform.openai.com/docs/api-reference/responses-streaming
"""
import asyncio
import json
from typing import Any

from fastapi import Request, Response

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.response_polling.polling_handler import ResponsePollingHandler


async def background_streaming_task(  # noqa: PLR0915
    polling_id: str,
    data: dict,
    polling_handler: ResponsePollingHandler,
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
        # https://platform.openai.com/docs/api-reference/responses-streaming
        output_items: dict[str, dict[str, Any]] = {}  # Track output items by ID
        accumulated_text = {}  # Track accumulated text deltas by (item_id, content_index)
        
        # ResponsesAPIResponse fields to extract from response.completed
        usage_data = None
        reasoning_data = None
        tool_choice_data = None
        tools_data = None
        model_data = None
        instructions_data = None
        temperature_data = None
        top_p_data = None
        max_output_tokens_data = None
        previous_response_id_data = None
        text_data = None
        truncation_data = None
        parallel_tool_calls_data = None
        user_data = None
        store_data = None
        incomplete_details_data = None
        
        state_dirty = False  # Track if state needs to be synced
        last_update_time = asyncio.get_event_loop().time()
        UPDATE_INTERVAL = 0.150  # 150ms batching interval
        
        async def flush_state_if_needed(force: bool = False) -> None:
            """Flush accumulated state to Redis if interval elapsed or forced"""
            nonlocal state_dirty, last_update_time
            
            current_time = asyncio.get_event_loop().time()
            if state_dirty and (force or (current_time - last_update_time) >= UPDATE_INTERVAL):
                # Convert output_items dict to list for update
                output_list = list(output_items.values())
                await polling_handler.update_state(
                    polling_id=polling_id,
                    output=output_list,
                )
                state_dirty = False
                last_update_time = current_time
        
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
                        
                        # Process different event types based on OpenAI streaming spec
                        if event_type == "response.output_item.added":
                            # New output item added
                            item = event.get("item", {})
                            item_id = item.get("id")
                            if item_id:
                                output_items[item_id] = item
                                state_dirty = True
                        
                        elif event_type == "response.content_part.added":
                            # Content part added to an output item
                            item_id = event.get("item_id")
                            content_part = event.get("part", {})
                            
                            if item_id and item_id in output_items:
                                # Update the output item with new content
                                if "content" not in output_items[item_id]:
                                    output_items[item_id]["content"] = []
                                output_items[item_id]["content"].append(content_part)
                                state_dirty = True
                        
                        elif event_type == "response.output_text.delta":
                            # Text delta - accumulate text content
                            # https://platform.openai.com/docs/api-reference/responses-streaming/response-text-delta
                            item_id = event.get("item_id")
                            content_index = event.get("content_index", 0)
                            delta = event.get("delta", "")
                            
                            if item_id and item_id in output_items:
                                # Accumulate text delta
                                key = (item_id, content_index)
                                if key not in accumulated_text:
                                    accumulated_text[key] = ""
                                accumulated_text[key] += delta
                                
                                # Update the content in output_items
                                if "content" in output_items[item_id]:
                                    content_list = output_items[item_id]["content"]
                                    if content_index < len(content_list):
                                        # Update existing content part with accumulated text
                                        if isinstance(content_list[content_index], dict):
                                            content_list[content_index]["text"] = accumulated_text[key]
                                state_dirty = True
                        
                        elif event_type == "response.content_part.done":
                            # Content part completed
                            item_id = event.get("item_id")
                            content_part = event.get("part", {})
                            content_index = event.get("content_index", 0)
                            
                            if item_id and item_id in output_items:
                                # Update with final content from event
                                if "content" in output_items[item_id]:
                                    content_list = output_items[item_id]["content"]
                                    if content_index < len(content_list):
                                        content_list[content_index] = content_part
                                state_dirty = True
                        
                        elif event_type == "response.output_item.done":
                            # Output item completed - use final item data
                            item = event.get("item", {})
                            item_id = item.get("id")
                            if item_id:
                                output_items[item_id] = item
                                state_dirty = True
                        
                        elif event_type == "response.in_progress":
                            # Response is now in progress
                            # https://platform.openai.com/docs/api-reference/responses-streaming/response-in-progress
                            await polling_handler.update_state(
                                polling_id=polling_id,
                                status="in_progress",
                            )
                        
                        elif event_type == "response.completed":
                            # Response completed - extract all ResponsesAPIResponse fields
                            # https://platform.openai.com/docs/api-reference/responses-streaming/response-completed
                            response_data = event.get("response", {})
                            
                            # Core response fields
                            usage_data = response_data.get("usage")
                            reasoning_data = response_data.get("reasoning")
                            tool_choice_data = response_data.get("tool_choice")
                            tools_data = response_data.get("tools")
                            
                            # Additional ResponsesAPIResponse fields
                            model_data = response_data.get("model")
                            instructions_data = response_data.get("instructions")
                            temperature_data = response_data.get("temperature")
                            top_p_data = response_data.get("top_p")
                            max_output_tokens_data = response_data.get("max_output_tokens")
                            previous_response_id_data = response_data.get("previous_response_id")
                            text_data = response_data.get("text")
                            truncation_data = response_data.get("truncation")
                            parallel_tool_calls_data = response_data.get("parallel_tool_calls")
                            user_data = response_data.get("user")
                            store_data = response_data.get("store")
                            incomplete_details_data = response_data.get("incomplete_details")
                            
                            # Also update output from final response if available
                            if "output" in response_data:
                                final_output = response_data.get("output", [])
                                for item in final_output:
                                    item_id = item.get("id")
                                    if item_id:
                                        output_items[item_id] = item
                                state_dirty = True
                        
                        # Flush state to Redis if interval elapsed
                        await flush_state_if_needed()
                        
                    except json.JSONDecodeError as e:
                        verbose_proxy_logger.warning(
                            f"Failed to parse streaming chunk: {e}"
                        )
                        pass
            
            # Final flush to ensure all accumulated state is saved
            await flush_state_if_needed(force=True)
        
        # Mark as completed with all ResponsesAPIResponse fields
        await polling_handler.update_state(
            polling_id=polling_id,
            status="completed",
            usage=usage_data,
            reasoning=reasoning_data,
            tool_choice=tool_choice_data,
            tools=tools_data,
            model=model_data,
            instructions=instructions_data,
            temperature=temperature_data,
            top_p=top_p_data,
            max_output_tokens=max_output_tokens_data,
            previous_response_id=previous_response_id_data,
            text=text_data,
            truncation=truncation_data,
            parallel_tool_calls=parallel_tool_calls_data,
            user=user_data,
            store=store_data,
            incomplete_details=incomplete_details_data,
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

