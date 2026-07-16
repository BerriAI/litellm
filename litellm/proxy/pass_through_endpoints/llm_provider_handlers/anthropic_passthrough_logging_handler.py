import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union, cast

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import use_custom_pricing_for_model
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_content_from_model_response,
)
from litellm.llms.anthropic import get_anthropic_config
from litellm.llms.anthropic.chat.handler import (
    ModelResponseIterator as AnthropicModelResponseIterator,
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import (
    Choices,
    LiteLLMBatch,
    Message,
    ModelResponse,
    TextCompletionResponse,
)

if TYPE_CHECKING:
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

    from ..success_handler import PassThroughEndpointLogging
else:
    PassThroughEndpointLogging = Any
    EndpointType = Any


class AnthropicPassthroughLoggingHandler:
    @staticmethod
    def anthropic_passthrough_handler(
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: Optional[dict] = None,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Transforms Anthropic response to OpenAI response, generates a standard logging object so downstream logging can be handled
        """
        # Check if this is a batch creation request
        if "/v1/messages/batches" in url_route and httpx_response.status_code == 200:
            # Get request body from parameter or kwargs
            request_body = request_body or kwargs.get("request_body", {})
            return AnthropicPassthroughLoggingHandler.batch_creation_handler(
                httpx_response=httpx_response,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                **kwargs,
            )

        model = response_body.get("model", "")
        anthropic_config = get_anthropic_config(url_route)
        litellm_model_response: ModelResponse = anthropic_config().transform_response(
            raw_response=httpx_response,
            model_response=litellm.ModelResponse(),
            model=model,
            messages=[],
            logging_obj=logging_obj,
            optional_params={},
            api_key="",
            request_data={},
            encoding=litellm.encoding,
            json_mode=False,
            litellm_params={},
        )

        kwargs = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=litellm_model_response,
            model=model,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
        )

        return {
            "result": litellm_model_response,
            "kwargs": kwargs,
        }

    @staticmethod
    def _get_user_from_metadata(
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
    ) -> Optional[str]:
        request_body = passthrough_logging_payload.get("request_body")
        if request_body:
            return get_end_user_id_from_request_body(request_body)
        return None

    @staticmethod
    def _resolve_costing_model(model: str, logging_obj: LiteLLMLoggingObj) -> str:
        if model and model != "unknown":
            return model
        litellm_params = (getattr(logging_obj, "model_call_details", {}) or {}).get("litellm_params", {}) or {}
        deployment_model = litellm_params.get("model")
        if deployment_model and deployment_model != "unknown":
            return deployment_model
        model_group = (litellm_params.get("metadata", {}) or {}).get("model_group")
        if model_group:
            return model_group.removeprefix("passthrough/")
        return model

    @staticmethod
    def _extract_model_from_anthropic_chunks(
        all_chunks: Sequence[Union[str, bytes]],
    ) -> Optional[str]:
        for raw in all_chunks:
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            for line in text.splitlines():
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[len("data:") :].strip())
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(data, dict):
                    continue
                if data.get("type") == "message_start":
                    model = (data.get("message") or {}).get("model")
                    if model:
                        return model
        return None

    @staticmethod
    def _stream_was_interrupted(
        all_chunks: Sequence[Union[str, bytes]],
    ) -> bool:
        """
        Anthropic ends a stream with ``content_block_stop`` -> ``message_delta``
        -> ``message_stop``; a client disconnect leaves the last event mid
        ``content_block_delta``. Scan from the tail and decide on the first
        terminal-region event, so the common completed case is O(1) rather than
        re-deserializing every line of the stream.
        """
        for raw in reversed(all_chunks):
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            for line in reversed(text.splitlines()):
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[len("data:") :].strip())
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(data, dict):
                    continue
                etype = data.get("type")
                if etype == "message_delta":
                    return False
                if etype in (
                    "content_block_delta",
                    "content_block_stop",
                    "message_start",
                ):
                    return True
        return True

    @staticmethod
    def _recover_interrupted_stream_output_tokens(
        response: Union[ModelResponse, TextCompletionResponse],
        all_chunks: Sequence[Union[str, bytes]],
        model: str,
    ) -> None:
        """
        An Anthropic stream interrupted before its terminal ``message_delta``
        (client disconnect) carries only the ``message_start`` ``output_tokens``
        placeholder (typically 1-3), so completion tokens and spend are
        undercounted ~20x. Re-tokenize the buffered output text to recover a
        realistic ``output_tokens`` for usage/cost. Completed streams are
        untouched because their terminal ``message_delta`` short-circuits here.
        """
        if not isinstance(response, ModelResponse):
            return
        if not AnthropicPassthroughLoggingHandler._stream_was_interrupted(all_chunks):
            return
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        output_text = get_content_from_model_response(response)
        if not output_text:
            return
        try:
            recovered_output_tokens = litellm.token_counter(model=model, text=output_text, count_response_tokens=True)
        except Exception:
            verbose_proxy_logger.warning(
                "Could not re-tokenize interrupted stream output; keeping placeholder completion token count."
            )
            return
        if recovered_output_tokens <= (usage.completion_tokens or 0):
            return
        usage.completion_tokens = recovered_output_tokens
        usage.total_tokens = (usage.prompt_tokens or 0) + recovered_output_tokens
        # Anthropic costing reads completion_tokens_details.text_tokens, so the
        # stale message_start placeholder there must be corrected too or spend
        # stays undercounted even after completion_tokens is fixed.
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None and getattr(details, "text_tokens", None) is not None:
            details.text_tokens = recovered_output_tokens

    @staticmethod
    def _create_anthropic_response_logging_payload(
        litellm_model_response: Union[ModelResponse, TextCompletionResponse],
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: LiteLLMLoggingObj,
    ):
        """
        Create the standard logging object for Anthropic passthrough

        handles streaming and non-streaming responses
        """
        # Only record complete_streaming_response for actual streaming responses.
        # perform_redaction scrubs this field only when stream is True, so setting
        # it on a non-streaming response would bypass message redaction.
        if logging_obj.model_call_details.get("stream") is True:
            logging_obj.model_call_details["complete_streaming_response"] = litellm_model_response
        try:
            # Get custom_llm_provider from logging object if available (e.g., azure_ai for Azure Anthropic)
            custom_llm_provider = logging_obj.model_call_details.get("custom_llm_provider")

            model = AnthropicPassthroughLoggingHandler._resolve_costing_model(model, logging_obj)

            # Prepend custom_llm_provider to model if not already present
            model_for_cost = model
            if custom_llm_provider and not model.startswith(f"{custom_llm_provider}/"):
                model_for_cost = f"{custom_llm_provider}/{model}"

            router_model_id = logging_obj.get_router_model_id()
            custom_pricing = use_custom_pricing_for_model(
                litellm_params=(logging_obj.litellm_params if hasattr(logging_obj, "litellm_params") else None)
            )

            response_cost = litellm.completion_cost(
                completion_response=litellm_model_response,
                model=model_for_cost,
                custom_llm_provider=custom_llm_provider,
                custom_pricing=custom_pricing,
                router_model_id=router_model_id,
            )

            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            # the pass-through success path reads spend from
            # model_call_details["response_cost"], not from kwargs
            logging_obj.model_call_details["response_cost"] = response_cost
            passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload] = (  # type: ignore
                kwargs.get("passthrough_logging_payload")
            )
            if passthrough_logging_payload:
                user = AnthropicPassthroughLoggingHandler._get_user_from_metadata(
                    passthrough_logging_payload=passthrough_logging_payload,
                )
                if user:
                    kwargs.setdefault("litellm_params", {})
                    kwargs["litellm_params"].update({"proxy_server_request": {"body": {"user": user}}})

            # pretty print standard logging object
            verbose_proxy_logger.debug(
                "kwargs= %s",
                json.dumps(kwargs, indent=4, default=str),
            )

            # set litellm_call_id to logging response object
            litellm_model_response.id = logging_obj.litellm_call_id
            litellm_model_response.model = model
            logging_obj.model_call_details["model"] = model
            if not logging_obj.model_call_details.get("custom_llm_provider"):
                logging_obj.model_call_details["custom_llm_provider"] = litellm.LlmProviders.ANTHROPIC.value
            return kwargs
        except Exception as e:
            verbose_proxy_logger.exception("Error creating Anthropic response logging payload: %s", e)
            return kwargs

    @staticmethod
    def _handle_logging_anthropic_collected_chunks(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: List[str],
        end_time: datetime,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Takes raw chunks from Anthropic passthrough endpoint and logs them in litellm callbacks

        - Builds complete response from chunks
        - Creates standard logging object
        - Logs in litellm callbacks
        """

        model = request_body.get("model", "")
        # Check if it's available in the logging object
        if (
            not model
            and hasattr(litellm_logging_obj, "model_call_details")
            and litellm_logging_obj.model_call_details.get("model")
        ):
            model = cast(str, litellm_logging_obj.model_call_details.get("model"))

        if not model or model == "unknown":
            chunk_model = AnthropicPassthroughLoggingHandler._extract_model_from_anthropic_chunks(all_chunks)
            if chunk_model:
                model = chunk_model

        try:
            complete_streaming_response = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
                all_chunks=all_chunks,
                litellm_logging_obj=litellm_logging_obj,
                model=model,
            )
        except Exception as e:
            # stream_chunk_builder re-raises assembly failures (as litellm.APIError)
            # on large agentic tool-use / thinking streams; treat that the same as a
            # None result so the usage-only fallback below still recovers cost
            verbose_proxy_logger.warning(
                "Anthropic passthrough: stream assembly raised (model=%s): %s; falling "
                "back to usage-only cost from raw SSE events.",
                model,
                e,
            )
            complete_streaming_response = None
        if complete_streaming_response is None:
            # stream_chunk_builder cannot always reassemble large agentic streams, but
            # Anthropic still emits token usage in the message_start / message_delta SSE
            # events regardless of content shape; recover usage-only so cost is tracked.
            # Guard it too: a raise here would defeat the point and drop the request
            try:
                complete_streaming_response = AnthropicPassthroughLoggingHandler._build_usage_only_response_from_chunks(
                    all_chunks=all_chunks,
                    model=model,
                )
            except Exception as e:
                verbose_proxy_logger.warning(
                    "Anthropic passthrough: usage-only fallback failed (model=%s): %s",
                    model,
                    e,
                )
                complete_streaming_response = None
        if complete_streaming_response is None:
            verbose_proxy_logger.error(
                "Unable to build complete streaming response for Anthropic passthrough endpoint, not logging..."
            )
            return {
                "result": None,
                "kwargs": {},
            }
        AnthropicPassthroughLoggingHandler._recover_interrupted_stream_output_tokens(
            response=complete_streaming_response,
            all_chunks=all_chunks,
            model=model,
        )
        kwargs = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=complete_streaming_response,
            model=model,
            kwargs={},
            start_time=start_time,
            end_time=end_time,
            logging_obj=litellm_logging_obj,
        )

        return {
            "result": complete_streaming_response,
            "kwargs": kwargs,
        }

    @staticmethod
    def _split_sse_chunk_into_events(chunk: Union[str, bytes]) -> List[str]:
        """
        Split a chunk that may contain multiple SSE events into individual events.

        SSE format: "event: type\ndata: {...}\n\n"
        Multiple events in a single chunk are separated by double newlines.

        Args:
            chunk: Raw chunk string that may contain multiple SSE events

        Returns:
            List of individual SSE event strings (each containing "event: X\ndata: {...}")
        """
        # Handle bytes input
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")

        # Split on double newlines to separate SSE events
        # Filter out empty strings
        events = [event.strip() for event in chunk.split("\n\n") if event.strip()]

        return events

    @staticmethod
    def _build_complete_streaming_response(
        all_chunks: Sequence[Union[str, bytes]],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        """
        Builds complete response from raw Anthropic chunks.

        Fast path: for the dominant case of a pure-text streaming response
        (no tool_use / thinking / non-text content blocks), the long run of
        ``content_block_delta`` text deltas is collapsed into a single
        equivalent SSE event before conversion. ``chunk_parser`` and
        ``stream_chunk_builder`` remain the single source of truth for chunk
        shape, usage math and finish-reason mapping, so the rebuilt response
        (and therefore the logged/billed payload) is identical -- this is
        asserted by a parity test. Anything non-trivial falls back to the
        unchanged legacy reconstruction.

        Per-event Pydantic ``ModelResponseStream`` construction dominated
        event-loop CPU under concurrent streaming; collapsing the homogeneous
        text run removes O(num_output_tokens) of it.
        """
        collapsed = AnthropicPassthroughLoggingHandler._collapse_pure_text_chunks(all_chunks)
        if collapsed is not None:
            return AnthropicPassthroughLoggingHandler._build_complete_streaming_response_legacy(
                all_chunks=collapsed,
                litellm_logging_obj=litellm_logging_obj,
                model=model,
            )
        return AnthropicPassthroughLoggingHandler._build_complete_streaming_response_legacy(
            all_chunks=all_chunks,
            litellm_logging_obj=litellm_logging_obj,
            model=model,
        )

    # Anthropic SSE block/delta types that the fast path is NOT allowed to
    # collapse -- their presence forces the unchanged legacy path so tool
    # calls, thinking, citations, etc. keep byte-identical reconstruction.
    _FAST_PATH_DISALLOWED_DELTA_TYPES = frozenset(
        {
            "input_json_delta",
            "thinking_delta",
            "signature_delta",
            "citations_delta",
        }
    )

    @staticmethod
    def _collapse_pure_text_chunks(
        all_chunks: Sequence[Union[str, bytes]],
    ) -> Optional[List[str]]:
        """
        Return a new chunk list with the contiguous run of text-only
        ``content_block_delta`` events replaced by a single equivalent event,
        or ``None`` if the stream is not a pure single-text-block response
        (in which case the caller uses the legacy path unchanged).

        Only ``message_start`` / ``content_block_start(text)`` /
        ``content_block_delta(text_delta)`` / ``content_block_stop`` /
        ``message_delta`` / ``message_stop`` / ``ping`` events are accepted.
        Any other content-block type or delta type returns ``None``.
        """
        normalized: List[str] = []
        for raw in all_chunks:
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            for ev in line.split("\n\n"):
                ev = ev.strip()
                if ev:
                    normalized.append(ev)

        text_block_indexes: set = set()
        out: List[str] = []
        pending_text: List[str] = []
        pending_index: Optional[int] = None
        saw_any_text_delta = False

        def flush() -> None:
            nonlocal pending_text, pending_index
            if pending_text:
                merged = {
                    "type": "content_block_delta",
                    "index": pending_index if pending_index is not None else 0,
                    "delta": {"type": "text_delta", "text": "".join(pending_text)},
                }
                out.append("data: " + json.dumps(merged))
                pending_text = []
                pending_index = None

        for ev in normalized:
            idx = ev.find("data:")
            if idx == -1:
                # Bare "event: <name>" line. The legacy converter turns this
                # into an empty ModelResponseStream that contributes nothing
                # to stream_chunk_builder. Drop the high-frequency interior
                # markers (content_block_delta / ping); keep every other
                # bare event line verbatim so chunk ordering and the
                # load-bearing chunks[0] (event: message_start) are retained.
                name = ev[len("event:") :].strip() if ev.startswith("event:") else ""
                if name in ("content_block_delta", "ping"):
                    continue
                flush()
                out.append(ev)
                continue

            json_str = ev[idx + len("data:") :].strip()
            try:
                data = json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                return None

            etype = data.get("type")
            if etype == "content_block_start":
                block = data.get("content_block") or {}
                if block.get("type") != "text":
                    return None
                text_block_indexes.add(data.get("index"))
                flush()
                out.append(ev)
            elif etype == "content_block_delta":
                delta = data.get("delta") or {}
                dtype = delta.get("type")
                if dtype in AnthropicPassthroughLoggingHandler._FAST_PATH_DISALLOWED_DELTA_TYPES:
                    return None
                if dtype != "text_delta":
                    return None
                cur_index = data.get("index")
                if cur_index not in text_block_indexes:
                    return None
                # Defensive: Anthropic sends blocks strictly sequentially
                # (start/deltas/stop, then next block), so pending_text from
                # block N must be flushed by content_block_stop before block
                # N+1's deltas arrive. If we ever see a delta whose index
                # disagrees with the current pending buffer, the stream is
                # interleaved -- fall back to legacy rather than risk merging
                # text from different blocks under a single index.
                if pending_text and pending_index is not None and cur_index != pending_index:
                    return None
                saw_any_text_delta = True
                pending_index = cur_index
                pending_text.append(delta.get("text") or "")
            elif etype == "ping":
                # Interior no-op; legacy maps it to an empty chunk.
                continue
            else:
                # message_start / content_block_stop / message_delta /
                # message_stop / error: pass through unchanged.
                flush()
                out.append(ev)

        flush()

        if not saw_any_text_delta:
            return None
        return out

    @staticmethod
    def _build_complete_streaming_response_legacy(
        all_chunks: Sequence[Union[str, bytes]],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        """
        Original reconstruction: convert every SSE event to a generic chunk
        and assemble via stream_chunk_builder. Kept verbatim as the fallback
        / source of truth for the fast path's parity test.

        - Splits multi-event chunks into individual SSE events
        - Converts str chunks to generic chunks
        - Converts generic chunks to litellm chunks (OpenAI format)
        - Builds complete response from litellm chunks
        """
        verbose_proxy_logger.debug("Building complete streaming response from %d chunks", len(all_chunks))
        anthropic_model_response_iterator = AnthropicModelResponseIterator(
            streaming_response=None,
            sync_stream=False,
        )
        all_openai_chunks = []

        # Process each chunk - a chunk may contain multiple SSE events
        for _chunk_str in all_chunks:
            # Split chunk into individual SSE events
            individual_events = AnthropicPassthroughLoggingHandler._split_sse_chunk_into_events(_chunk_str)

            # Process each individual event
            for event_str in individual_events:
                try:
                    # Skip OpenAI-style [DONE] sentinels some Anthropic-compatible
                    # providers emit. Match the whole SSE line so a valid chunk whose
                    # text payload happens to contain "[DONE]" is not dropped.
                    if any(line.strip() == "data: [DONE]" for line in event_str.split("\n")):
                        continue
                    transformed_openai_chunk = anthropic_model_response_iterator.convert_str_chunk_to_generic_chunk(
                        chunk=event_str
                    )
                    if transformed_openai_chunk is not None:
                        all_openai_chunks.append(transformed_openai_chunk)

                except (StopIteration, StopAsyncIteration):
                    break
                except json.JSONDecodeError:
                    # Some upstreams emit non-JSON SSE lines; skip them so the
                    # logging pipeline is not broken by a single bad frame.
                    verbose_proxy_logger.debug(
                        "Skipping non-JSON SSE event: %s",
                        event_str[:200],
                    )
                    continue

        complete_streaming_response = litellm.stream_chunk_builder(
            chunks=all_openai_chunks,
            logging_obj=litellm_logging_obj,
        )
        verbose_proxy_logger.debug("Complete streaming response built: %s", complete_streaming_response)
        return complete_streaming_response

    @staticmethod
    def _extract_sse_data(event_str: str) -> Optional[dict]:
        """Parse the JSON object from the ``data:`` line of an Anthropic SSE event."""
        for line in event_str.splitlines():
            stripped = line.strip()
            if stripped.startswith("data:"):
                payload = stripped[len("data:") :].strip()
                if not payload or payload == "[DONE]":
                    return None
                try:
                    return cast(dict, json.loads(payload))
                except (ValueError, TypeError):
                    return None
        return None

    @staticmethod
    def _build_usage_only_response_from_chunks(
        all_chunks: Sequence[Union[str, bytes]],
        model: str,
    ) -> Optional[ModelResponse]:
        """
        Build a usage-bearing ModelResponse from Anthropic SSE token-usage events, for
        cost tracking when stream_chunk_builder cannot reassemble the stream.

        Anthropic emits usage in ``message_start`` (uncached input + cache tokens, and an
        initial output_tokens) and the final ``message_delta`` (cumulative output_tokens)
        regardless of the content/tool shape, so cost is recoverable even when full
        content assembly fails. Returns ``None`` if no usage event is found.
        """
        input_tokens = 0
        cache_read = 0
        cache_creation = 0
        cache_creation_5m: Optional[int] = None
        cache_creation_1h: Optional[int] = None
        output_tokens = 0
        web_search_requests: Optional[int] = None
        tool_search_requests: Optional[int] = None
        inference_geo: Optional[str] = None
        stop_reason: Optional[str] = None
        found_usage = False
        resolved_model = model
        for _chunk_str in all_chunks:
            for event_str in AnthropicPassthroughLoggingHandler._split_sse_chunk_into_events(_chunk_str):
                data = AnthropicPassthroughLoggingHandler._extract_sse_data(event_str)
                if not data:
                    continue
                event_type = data.get("type")
                if event_type == "message_start":
                    message = data.get("message") or {}
                    if not resolved_model or resolved_model == "unknown":
                        resolved_model = message.get("model") or resolved_model
                    usage = message.get("usage") or {}
                    input_tokens = usage.get("input_tokens") or input_tokens
                    cache_read = usage.get("cache_read_input_tokens") or cache_read
                    cache_creation = usage.get("cache_creation_input_tokens") or cache_creation
                    _cc = usage.get("cache_creation")
                    if isinstance(_cc, dict):
                        cache_creation_5m = _cc.get("ephemeral_5m_input_tokens")
                        cache_creation_1h = _cc.get("ephemeral_1h_input_tokens")
                    if usage.get("inference_geo") is not None:
                        inference_geo = usage.get("inference_geo")
                    if usage.get("output_tokens") is not None:
                        output_tokens = usage.get("output_tokens")
                    found_usage = True
                elif event_type == "message_delta":
                    _delta_stop = (data.get("delta") or {}).get("stop_reason")
                    if _delta_stop:
                        stop_reason = _delta_stop
                    usage = data.get("usage") or {}
                    if usage.get("output_tokens") is not None:
                        output_tokens = usage.get("output_tokens")
                    _stu = usage.get("server_tool_use")
                    if isinstance(_stu, dict):
                        if _stu.get("web_search_requests") is not None:
                            web_search_requests = _stu.get("web_search_requests")
                        if _stu.get("tool_search_requests") is not None:
                            tool_search_requests = _stu.get("tool_search_requests")
                    if usage.get("cache_read_input_tokens") is not None:
                        cache_read = usage.get("cache_read_input_tokens")
                    if usage.get("inference_geo") is not None:
                        inference_geo = usage.get("inference_geo")
                    found_usage = True
        if not found_usage:
            return None
        # If only the 5m/1h split was provided, derive the cache_creation total from it.
        if not cache_creation and (cache_creation_5m or cache_creation_1h):
            cache_creation = (cache_creation_5m or 0) + (cache_creation_1h or 0)
        # build usage via the same AnthropicConfig.calculate_usage path the success
        # cases use, so prompt_tokens are cache-inclusive and cache / server_tool_use /
        # inference_geo tokens are priced instead of left at $0
        usage_object: dict = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        if cache_read:
            usage_object["cache_read_input_tokens"] = cache_read
        if cache_creation:
            usage_object["cache_creation_input_tokens"] = cache_creation
        if cache_creation_5m is not None or cache_creation_1h is not None:
            usage_object["cache_creation"] = {
                "ephemeral_5m_input_tokens": cache_creation_5m or 0,
                "ephemeral_1h_input_tokens": cache_creation_1h or 0,
            }
        if web_search_requests is not None or tool_search_requests is not None:
            _server_tool_use: dict = {}
            if web_search_requests is not None:
                _server_tool_use["web_search_requests"] = web_search_requests
            if tool_search_requests is not None:
                _server_tool_use["tool_search_requests"] = tool_search_requests
            usage_object["server_tool_use"] = _server_tool_use
        if inference_geo is not None:
            usage_object["inference_geo"] = inference_geo
        usage_obj = AnthropicConfig().calculate_usage(usage_object=usage_object, reasoning_content=None)
        return ModelResponse(
            model=resolved_model,
            choices=[
                Choices(
                    finish_reason=(map_finish_reason(stop_reason) if stop_reason else "stop"),
                    index=0,
                    message=Message(role="assistant", content=""),
                )
            ],
            usage=usage_obj,
        )

    @staticmethod
    def batch_creation_handler(
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: Optional[dict] = None,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Handle Anthropic batch creation passthrough logging.
        Creates a managed object for cost tracking when batch job is successfully created.
        """
        import base64

        from litellm._uuid import uuid
        from litellm.llms.anthropic.batches.transformation import AnthropicBatchesConfig
        from litellm.types.utils import Choices, SpecialEnums

        try:
            _json_response = httpx_response.json()

            # Only handle successful batch job creation (POST requests with 201 status)
            if httpx_response.status_code == 200 and "id" in _json_response:
                # Transform Anthropic response to LiteLLM batch format
                anthropic_batches_config = AnthropicBatchesConfig()
                litellm_batch_response = anthropic_batches_config.transform_retrieve_batch_response(
                    model=None,
                    raw_response=httpx_response,
                    logging_obj=logging_obj,
                    litellm_params={},
                )
                # Set status to "validating" for newly created batches so polling mechanism picks them up
                # The polling mechanism only looks for status="validating" jobs
                litellm_batch_response.status = "validating"

                # Extract batch ID from the response
                batch_id = _json_response.get("id", "")

                # Get model from request body (batch response doesn't include model)
                request_body = request_body or {}
                # Try to extract model from the batch request body, supporting Anthropic's nested structure
                model_name: str = "unknown"
                if isinstance(request_body, dict):
                    # Standard: {"model": ...}
                    model_name = request_body.get("model") or "unknown"
                    if model_name == "unknown":
                        # Anthropic batches: look under requests[0].params.model
                        requests_list = request_body.get("requests", [])
                        if isinstance(requests_list, list) and len(requests_list) > 0:
                            first_req = requests_list[0]
                            if isinstance(first_req, dict):
                                params = first_req.get("params", {})
                                if isinstance(params, dict):
                                    extracted_model = params.get("model")
                                    if extracted_model:
                                        model_name = extracted_model

                # Create unified object ID for tracking
                # Format: base64(litellm_proxy;model_id:{};llm_batch_id:{})
                # For Anthropic passthrough, prefix model with "anthropic/" so router can determine provider
                actual_model_id = AnthropicPassthroughLoggingHandler.get_actual_model_id_from_router(model_name)

                # If model not in router, use "anthropic/{model_name}" format so router can determine provider
                if actual_model_id == model_name and not actual_model_id.startswith("anthropic/"):
                    actual_model_id = f"anthropic/{model_name}"

                unified_id_string = SpecialEnums.LITELLM_MANAGED_BATCH_COMPLETE_STR.value.format(
                    actual_model_id, batch_id
                )
                unified_object_id = base64.urlsafe_b64encode(unified_id_string.encode()).decode().rstrip("=")

                # Store the managed object for cost tracking
                # This will be picked up by check_batch_cost polling mechanism
                AnthropicPassthroughLoggingHandler._store_batch_managed_object(
                    unified_object_id=unified_object_id,
                    batch_object=litellm_batch_response,
                    model_object_id=batch_id,
                    logging_obj=logging_obj,
                    **kwargs,
                )

                # Create a batch job response for logging
                litellm_model_response = ModelResponse()
                litellm_model_response.id = str(uuid.uuid4())
                litellm_model_response.model = model_name
                litellm_model_response.object = "batch"
                litellm_model_response.created = int(start_time.timestamp())

                # Add batch-specific metadata to indicate this is a pending batch job
                litellm_model_response.choices = [
                    Choices(
                        finish_reason="stop",
                        index=0,
                        message={
                            "role": "assistant",
                            "content": f"Batch job {batch_id} created and is pending. Status will be updated when the batch completes.",
                            "tool_calls": None,
                            "function_call": None,
                            "provider_specific_fields": {
                                "batch_job_id": batch_id,
                                "batch_job_state": "in_progress",
                                "unified_object_id": unified_object_id,
                            },
                        },
                    )
                ]

                # Set response cost to 0 initially (will be updated when batch completes)
                response_cost = 0.0
                kwargs["response_cost"] = response_cost
                kwargs["model"] = model_name
                kwargs["batch_id"] = batch_id
                kwargs["unified_object_id"] = unified_object_id
                kwargs["batch_job_state"] = "in_progress"

                logging_obj.model = model_name
                logging_obj.model_call_details["model"] = logging_obj.model
                logging_obj.model_call_details["response_cost"] = response_cost
                logging_obj.model_call_details["batch_id"] = batch_id

                return {
                    "result": litellm_model_response,
                    "kwargs": kwargs,
                }
            else:
                # Handle non-successful responses
                litellm_model_response = ModelResponse()
                litellm_model_response.id = str(uuid.uuid4())
                litellm_model_response.model = "anthropic_batch"
                litellm_model_response.object = "batch"
                litellm_model_response.created = int(start_time.timestamp())

                # Add error-specific metadata
                litellm_model_response.choices = [
                    Choices(
                        finish_reason="stop",
                        index=0,
                        message={
                            "role": "assistant",
                            "content": f"Batch job creation failed. Status: {httpx_response.status_code}",
                            "tool_calls": None,
                            "function_call": None,
                            "provider_specific_fields": {
                                "batch_job_state": "failed",
                                "status_code": httpx_response.status_code,
                            },
                        },
                    )
                ]

                kwargs["response_cost"] = 0.0
                kwargs["model"] = "anthropic_batch"
                kwargs["batch_job_state"] = "failed"

                return {
                    "result": litellm_model_response,
                    "kwargs": kwargs,
                }

        except Exception as e:
            verbose_proxy_logger.error(f"Error in batch_creation_handler: {e}")
            # Return basic response on error
            litellm_model_response = ModelResponse()
            litellm_model_response.id = str(uuid.uuid4())
            litellm_model_response.model = "anthropic_batch"
            litellm_model_response.object = "batch"
            litellm_model_response.created = int(start_time.timestamp())

            # Add error-specific metadata
            litellm_model_response.choices = [
                Choices(
                    finish_reason="stop",
                    index=0,
                    message={
                        "role": "assistant",
                        "content": f"Error creating batch job: {str(e)}",
                        "tool_calls": None,
                        "function_call": None,
                        "provider_specific_fields": {
                            "batch_job_state": "failed",
                            "error": str(e),
                        },
                    },
                )
            ]

            kwargs["response_cost"] = 0.0
            kwargs["model"] = "anthropic_batch"
            kwargs["batch_job_state"] = "failed"

            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }

    @staticmethod
    def _store_batch_managed_object(
        unified_object_id: str,
        batch_object: LiteLLMBatch,
        model_object_id: str,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> None:
        """
        Store batch managed object for cost tracking.
        This will be picked up by the check_batch_cost polling mechanism.
        """
        try:
            # Get the managed files hook from the logging object
            # This is a bit of a hack, but we need access to the proxy logging system
            from litellm.proxy.proxy_server import proxy_logging_obj

            managed_files_hook = proxy_logging_obj.get_proxy_hook("managed_files")
            if managed_files_hook is not None and hasattr(managed_files_hook, "store_unified_object_id"):
                # Create a mock user API key dict for the managed object storage
                from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

                _request_metadata = (kwargs.get("litellm_params", {}) or {}).get("metadata", {}) or {}

                _request_tags = _request_metadata.get("tags")
                user_api_key_dict = UserAPIKeyAuth(
                    user_id=_request_metadata.get("user_api_key_user_id", "default-user"),
                    api_key=_request_metadata.get("user_api_key") or "",
                    team_id=_request_metadata.get("user_api_key_team_id"),
                    team_alias=_request_metadata.get("user_api_key_team_alias"),
                    user_role=LitellmUserRoles.CUSTOMER,  # Use proper enum value
                    user_email=None,
                    max_budget=None,
                    spend=0.0,  # Set to 0.0 instead of None
                    models=[],  # Set to empty list instead of None
                    tpm_limit=None,
                    rpm_limit=None,
                    budget_duration=None,
                    budget_reset_at=None,
                    max_parallel_requests=None,
                    allowed_model_region=None,
                    metadata={},  # Set to empty dict instead of None
                    key_alias=_request_metadata.get("user_api_key_alias"),
                    permissions={},  # Set to empty dict instead of None
                    model_max_budget={},  # Set to empty dict instead of None
                    model_spend={},  # Set to empty dict instead of None
                )

                # Store the unified object for batch cost tracking
                import asyncio

                asyncio.create_task(
                    managed_files_hook.store_unified_object_id(  # type: ignore
                        unified_object_id=unified_object_id,
                        file_object=batch_object,
                        litellm_parent_otel_span=None,
                        model_object_id=model_object_id,
                        file_purpose="batch",
                        user_api_key_dict=user_api_key_dict,
                        request_tags=_request_tags if isinstance(_request_tags, list) else None,
                    )
                )

                verbose_proxy_logger.info(
                    f"Stored Anthropic batch managed object with unified_object_id={unified_object_id}, batch_id={model_object_id}"
                )
            else:
                verbose_proxy_logger.warning(
                    "Managed files hook not available, cannot store batch object for cost tracking"
                )

        except Exception as e:
            verbose_proxy_logger.error(f"Error storing Anthropic batch managed object: {e}")

    @staticmethod
    def get_actual_model_id_from_router(model_name: str) -> str:
        from litellm.proxy.proxy_server import llm_router

        if llm_router is not None:
            # Try to find the model in the router by the model name
            # Use the existing get_model_ids method from router
            model_ids = llm_router.get_model_ids(model_name=model_name)
            if model_ids and len(model_ids) > 0:
                # Use the first model ID found
                actual_model_id = model_ids[0]
                verbose_proxy_logger.info(f"Found model ID in router: {actual_model_id}")
                return actual_model_id
            else:
                # Fallback to model name
                actual_model_id = model_name
                verbose_proxy_logger.warning(f"Model not found in router, using model name: {actual_model_id}")
                return actual_model_id
        else:
            # Fallback if router is not available
            verbose_proxy_logger.warning(f"Router not available, using model name: {model_name}")
            return model_name
