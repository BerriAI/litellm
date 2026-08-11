"""
Background Streaming Task for Polling Via Cache Feature

Handles streaming responses from LLM providers and updates Redis cache
with partial results for polling.

Follows OpenAI Response Streaming format:
https://platform.openai.com/docs/api-reference/responses-streaming
"""

import asyncio
from collections.abc import AsyncIterable, Callable
from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable

from fastapi import Request, Response
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.response_polling.polling_handler import ResponsePollingHandler
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.llms.openai import ResponsesAPIStatus

if TYPE_CHECKING:
    from litellm.proxy.proxy_server import ProxyConfig


@runtime_checkable
class StreamingBody(Protocol):
    """Any response exposing a server sent event body, such as a fastapi StreamingResponse"""

    body_iterator: AsyncIterable[str | bytes | memoryview]


class ResponsesRequestProcessor(Protocol):
    """Proxy request processor able to run a Responses API request"""

    async def base_process_llm_request(
        self,
        *,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: UserAPIKeyAuth,
        route_type: Literal["aresponses"],
        proxy_logging_obj: ProxyLogging,
        llm_router: Router | None,
        general_settings: dict[str, object],
        proxy_config: "ProxyConfig",
        select_data_generator: Callable[..., object] | None,
        model: str | None,
        user_model: str | None,
        user_temperature: float | None,
        user_request_timeout: float | None,
        user_max_tokens: int | None,
        user_api_base: str | None,
        version: str | None,
        skip_pre_call_logic: bool,
    ) -> object: ...


class StreamedResponseSnapshot(BaseModel):
    """Terminal `response` payload of a Responses API stream"""

    model_config = ConfigDict(extra="ignore")

    status: ResponsesAPIStatus | None = None
    error: dict[str, JsonValue] | None = None
    incomplete_details: dict[str, JsonValue] | None = None
    usage: dict[str, JsonValue] | None = None
    reasoning: dict[str, JsonValue] | None = None
    tool_choice: JsonValue = None
    tools: list[JsonValue] | None = None
    model: str | None = None
    instructions: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    previous_response_id: str | None = None
    text: dict[str, JsonValue] | None = None
    truncation: str | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None
    store: bool | None = None
    output: list[dict[str, JsonValue]] | None = None


class StreamedEvent(BaseModel):
    """Server sent event of a Responses API stream"""

    model_config = ConfigDict(extra="ignore")

    type: str = ""
    item: dict[str, JsonValue] | None = None
    item_id: str | None = None
    part: dict[str, JsonValue] | None = None
    content_index: int = 0
    delta: str = ""
    response: StreamedResponseSnapshot | None = None


_EVENT_ADAPTER: Final = TypeAdapter(StreamedEvent)

_EVENT_TO_STATUS: Final[dict[str, ResponsesAPIStatus]] = {
    "response.completed": "completed",
    "response.failed": "failed",
    "response.incomplete": "incomplete",
    "response.cancelled": "cancelled",
}

_UPDATE_INTERVAL: Final = 0.150


def _request_processor(data: dict[str, object]) -> ResponsesRequestProcessor:
    return ProxyBaseLLMRequestProcessing(data=data)


def _item_content(item: dict[str, JsonValue]) -> list[JsonValue] | None:
    content: Final = item.get("content")
    return content if isinstance(content, list) else None


def _apply_event(
    event: StreamedEvent,
    output_items: dict[str, dict[str, JsonValue]],
    accumulated_text: dict[tuple[str, int], str],
) -> bool:
    """Apply one streaming event to the accumulated output, reporting whether it changed"""
    if event.type == "response.output_item.added" or event.type == "response.output_item.done":
        item: Final = event.item or {}
        item_id: Final = item.get("id")
        if isinstance(item_id, str) and item_id:
            output_items[item_id] = item
            return True
        return False

    tracked_item: Final = output_items.get(event.item_id or "")
    if tracked_item is None:
        return False

    content: Final = _item_content(tracked_item)

    if event.type == "response.content_part.added":
        # Update the output item with new content
        part: Final = event.part or {}
        if content is None:
            tracked_item["content"] = [part]
        else:
            content.append(part)
        return True

    if event.type == "response.output_text.delta":
        if event.item_id is None:
            return False

        # Accumulate text delta
        # https://platform.openai.com/docs/api-reference/responses-streaming/response-text-delta
        key: Final = (event.item_id, event.content_index)
        accumulated_text[key] = accumulated_text.get(key, "") + event.delta

        if content is not None and event.content_index < len(content):
            # Update existing content part with accumulated text
            content_part: Final = content[event.content_index]
            if isinstance(content_part, dict):
                content_part["text"] = accumulated_text[key]
        return True

    if event.type == "response.content_part.done":
        # Update with final content from the event
        if content is not None and event.content_index < len(content):
            content[event.content_index] = event.part or {}
        return True

    return False


async def background_streaming_task(
    polling_id: str,
    data: dict[str, object],
    polling_handler: ResponsePollingHandler,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    general_settings: dict[str, object],
    llm_router: Router | None,
    proxy_config: "ProxyConfig",
    proxy_logging_obj: ProxyLogging,
    select_data_generator: Callable[..., object] | None,
    user_model: str | None,
    user_temperature: float | None,
    user_request_timeout: float | None,
    user_max_tokens: int | None,
    user_api_base: str | None,
    version: str | None,
) -> None:
    """
    Background task to stream response and update cache

    Follows OpenAI Response Streaming format:
    https://platform.openai.com/docs/api-reference/responses-streaming

    Processes streaming events and builds Response object:
    https://platform.openai.com/docs/api-reference/responses/object
    """

    try:
        verbose_proxy_logger.info("Starting background streaming for %s", polling_id)

        # Update status to in_progress (OpenAI format)
        await polling_handler.update_state(
            polling_id=polling_id,
            status="in_progress",
        )

        # Force streaming mode and remove background flag
        data["stream"] = True
        data.pop("background", None)

        # Create processor
        processor: Final = _request_processor(data)

        # Make streaming request.
        # Pre-call checks (rate limits, guardrails, budget) were already run
        # before polling ID creation, so skip them here to avoid double-counting.
        response: Final[object] = await processor.base_process_llm_request(
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
            skip_pre_call_logic=True,
        )

        # Process streaming response following OpenAI events format
        # https://platform.openai.com/docs/api-reference/responses-streaming
        output_items: Final[dict[str, dict[str, JsonValue]]] = {}  # Track output items by ID
        accumulated_text: Final[dict[tuple[str, int], str]] = {}  # Text deltas by (item_id, content_index)

        # ResponsesAPIResponse fields extracted from the terminal event
        terminal_response: StreamedResponseSnapshot | None = None

        state_dirty = False  # Track if state needs to be synced
        last_update_time = asyncio.get_event_loop().time()

        # Track the terminal event from the stream (may not be "completed")
        terminal_status: ResponsesAPIStatus | None = (
            None  # Will be set by response.completed/failed/incomplete/cancelled
        )

        async def flush_state_if_needed(force: bool = False) -> None:
            """Flush accumulated state to Redis if interval elapsed or forced"""
            nonlocal state_dirty, last_update_time

            current_time: Final = asyncio.get_event_loop().time()
            if state_dirty and (force or (current_time - last_update_time) >= _UPDATE_INTERVAL):
                # Convert output_items dict to list for update
                output_list: Final = list(output_items.values())
                await polling_handler.update_state(
                    polling_id=polling_id,
                    output=output_list,
                )
                state_dirty = False
                last_update_time = current_time

        # Handle StreamingResponse
        if not isinstance(response, StreamingBody):
            verbose_proxy_logger.warning(
                "background_streaming_task: response for %s has no body_iterator; this may indicate a misconfiguration or provider error",
                polling_id,
            )

        if isinstance(response, StreamingBody):
            async for raw_chunk in response.body_iterator:
                # Parse chunk
                chunk = raw_chunk.decode("utf-8") if isinstance(raw_chunk, bytes) else raw_chunk

                if isinstance(chunk, str) and chunk.startswith("data: "):
                    chunk_data = chunk[6:].strip()
                    if chunk_data == "[DONE]":
                        break

                    try:
                        event = _EVENT_ADAPTER.validate_json(chunk_data)

                        # Process different event types based on OpenAI streaming spec
                        if event.type == "response.in_progress":
                            # Response is now in progress
                            # https://platform.openai.com/docs/api-reference/responses-streaming/response-in-progress
                            await polling_handler.update_state(
                                polling_id=polling_id,
                                status="in_progress",
                            )

                        elif event.type in _EVENT_TO_STATUS:
                            # Terminal event - extract all ResponsesAPIResponse fields
                            # https://platform.openai.com/docs/api-reference/responses-streaming
                            terminal_response = event.response or StreamedResponseSnapshot()
                            terminal_status = terminal_response.status or _EVENT_TO_STATUS[event.type]

                            # Also update output from final response if available
                            if terminal_response.output is not None:
                                for final_item in terminal_response.output:
                                    final_item_id = final_item.get("id")
                                    if isinstance(final_item_id, str) and final_item_id:
                                        output_items[final_item_id] = final_item
                                state_dirty = True

                        elif _apply_event(event, output_items, accumulated_text):
                            state_dirty = True

                        # Flush state to Redis if interval elapsed
                        await flush_state_if_needed()

                    except ValidationError as e:
                        verbose_proxy_logger.warning("Failed to parse streaming chunk: %s", e)

            # Final flush to ensure all accumulated state is saved
            await flush_state_if_needed(force=True)

        # Use the terminal status from the stream, default to "completed"
        final_status: Final = terminal_status or "completed"
        final_response: Final = terminal_response or StreamedResponseSnapshot()
        terminal_error: Final = (
            final_response.error if final_status == "failed" or final_status == "incomplete" else None
        )

        await polling_handler.update_state(
            polling_id=polling_id,
            status=final_status,
            usage=final_response.usage,
            error=terminal_error,
            reasoning=final_response.reasoning,
            tool_choice=final_response.tool_choice,
            tools=final_response.tools,
            model=final_response.model,
            instructions=final_response.instructions,
            temperature=final_response.temperature,
            top_p=final_response.top_p,
            max_output_tokens=final_response.max_output_tokens,
            previous_response_id=final_response.previous_response_id,
            text=final_response.text,
            truncation=final_response.truncation,
            parallel_tool_calls=final_response.parallel_tool_calls,
            user=final_response.user,
            store=final_response.store,
            incomplete_details=final_response.incomplete_details,
        )

        verbose_proxy_logger.info(
            "Finished background streaming for %s, status=%s, error=%s, incomplete_details=%s, output_items=%s",
            polling_id,
            final_status,
            terminal_error,
            final_response.incomplete_details,
            len(output_items),
        )

    except Exception as e:
        verbose_proxy_logger.error("Error in background streaming task for %s: %s", polling_id, e)
        import traceback

        verbose_proxy_logger.error(traceback.format_exc())

        await polling_handler.update_state(
            polling_id=polling_id,
            status="failed",
            error={
                "type": "internal_error",
                "message": str(e),
                "code": "background_streaming_error",
            },
        )
