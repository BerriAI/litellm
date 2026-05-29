import asyncio
from datetime import datetime
from typing import List, Optional, Tuple

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.proxy._types import PassThroughEndpointLoggingResultValues
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.utils import StandardPassThroughResponseObject

from .llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from .llm_provider_handlers.openai_passthrough_logging_handler import (
    OpenAIPassthroughLoggingHandler,
)
from .llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from .success_handler import PassThroughEndpointLogging


class PassThroughStreamingHandler:
    @staticmethod
    async def chunk_processor(
        response: httpx.Response,
        request_body: Optional[dict],
        litellm_logging_obj: LiteLLMLoggingObj,
        endpoint_type: EndpointType,
        start_time: datetime,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
    ):
        raw_bytes: List[bytes] = []
        logging_scheduled = False
        model_name = PassThroughStreamingHandler._extract_model_for_cost_injection(
            request_body=request_body,
            url_route=url_route,
            endpoint_type=endpoint_type,
            litellm_logging_obj=litellm_logging_obj,
        )

        # Resolve once per stream rather than re-reading the global +
        # re-branching on every chunk. ``include_cost_in_streaming_usage`` is
        # set at config load and stable for the process, matching how the
        # proxy-level streaming fast path resolves it.
        cost_injection_active = (
            bool(getattr(litellm, "include_cost_in_streaming_usage", False))
            and bool(model_name)
            and endpoint_type in (EndpointType.VERTEX_AI, EndpointType.ANTHROPIC)
        )
        try:
            if not cost_injection_active:
                # Hot path: just buffer for end-of-stream logging and forward.
                async for chunk in response.aiter_bytes():
                    raw_bytes.append(chunk)
                    yield chunk
            else:
                # ``cost_injection_active`` already requires ``model_name`` to
                # be truthy; pin to a typed local so mypy narrows ``Optional[str]``
                # -> ``str`` for the per-chunk call site.
                assert model_name is not None
                resolved_model_name: str = model_name
                async for chunk in response.aiter_bytes():
                    raw_bytes.append(chunk)
                    if endpoint_type == EndpointType.VERTEX_AI:
                        if "streamRawPredict" in url_route or "rawPredict" in url_route:
                            modified_chunk = ProxyBaseLLMRequestProcessing._process_chunk_with_cost_injection(
                                chunk, resolved_model_name
                            )
                            if modified_chunk is not None:
                                chunk = modified_chunk
                    else:  # EndpointType.ANTHROPIC
                        modified_chunk = ProxyBaseLLMRequestProcessing._process_chunk_with_cost_injection(
                            chunk, resolved_model_name
                        )
                        if modified_chunk is not None:
                            chunk = modified_chunk

                    yield chunk
        except Exception as e:
            verbose_proxy_logger.error(f"Error in chunk_processor: {str(e)}")
            raise
        finally:
            # GeneratorExit (raised on client disconnect) is not caught by
            # `except Exception`; the finally block ensures partial usage
            # still gets logged for spend tracking. See LIT-2642.
            if not logging_scheduled and raw_bytes:
                logging_scheduled = True
                try:
                    asyncio.create_task(
                        PassThroughStreamingHandler._route_streaming_logging_to_handler(
                            litellm_logging_obj=litellm_logging_obj,
                            passthrough_success_handler_obj=passthrough_success_handler_obj,
                            url_route=url_route,
                            request_body=request_body or {},
                            endpoint_type=endpoint_type,
                            start_time=start_time,
                            raw_bytes=raw_bytes,
                            end_time=datetime.now(),
                        )
                    )
                except Exception as e:
                    verbose_proxy_logger.error(
                        f"Error scheduling chunk_processor logging: {str(e)}"
                    )

    @staticmethod
    async def _route_streaming_logging_to_handler(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        raw_bytes: List[bytes],
        end_time: datetime,
        model: Optional[str] = None,
    ):
        """
        Route the logging for the collected chunks to the appropriate handler

        Supported endpoint types:
        - Anthropic
        - Vertex AI
        - OpenAI
        """
        try:
            (
                standard_logging_response_object,
                kwargs,
            ) = PassThroughStreamingHandler._build_passthrough_logging_result(
                litellm_logging_obj=litellm_logging_obj,
                passthrough_success_handler_obj=passthrough_success_handler_obj,
                url_route=url_route,
                request_body=request_body,
                endpoint_type=endpoint_type,
                start_time=start_time,
                raw_bytes=raw_bytes,
                end_time=end_time,
                model=model,
            )
            await litellm_logging_obj.async_success_handler(
                result=standard_logging_response_object,
                start_time=start_time,
                end_time=end_time,
                cache_hit=False,
                **kwargs,
            )
            if (
                litellm_logging_obj._should_run_sync_callbacks_for_async_calls()
                is False
            ):
                return

            executor.submit(
                litellm_logging_obj.success_handler,
                result=standard_logging_response_object,
                end_time=end_time,
                cache_hit=False,
                start_time=start_time,
                **kwargs,
            )
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error in _route_streaming_logging_to_handler: {str(e)}"
            )

    @staticmethod
    def _build_passthrough_logging_result(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        raw_bytes: List[bytes],
        end_time: datetime,
        model: Optional[str],
    ) -> Tuple[PassThroughEndpointLoggingResultValues, dict]:
        """
        Synchronous, CPU-bound reconstruction of the standard logging payload
        from collected raw SSE bytes. Extracted from
        _route_streaming_logging_to_handler so the per-endpoint dispatch can
        be unit-tested in isolation. Still invoked synchronously on the event
        loop; an off-loop dispatch is a future change, not part of this PR.
        """
        all_chunks = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(
            raw_bytes
        )
        standard_logging_response_object: Optional[
            PassThroughEndpointLoggingResultValues
        ] = None
        kwargs: dict = {}
        if endpoint_type == EndpointType.ANTHROPIC:
            anthropic_passthrough_logging_handler_result = AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
                litellm_logging_obj=litellm_logging_obj,
                passthrough_success_handler_obj=passthrough_success_handler_obj,
                url_route=url_route,
                request_body=request_body,
                endpoint_type=endpoint_type,
                start_time=start_time,
                all_chunks=all_chunks,
                end_time=end_time,
            )
            standard_logging_response_object = (
                anthropic_passthrough_logging_handler_result["result"]
            )
            kwargs = anthropic_passthrough_logging_handler_result["kwargs"]
        elif endpoint_type == EndpointType.VERTEX_AI:
            vertex_passthrough_logging_handler_result = (
                VertexPassthroughLoggingHandler._handle_logging_vertex_collected_chunks(
                    litellm_logging_obj=litellm_logging_obj,
                    passthrough_success_handler_obj=passthrough_success_handler_obj,
                    url_route=url_route,
                    request_body=request_body,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    all_chunks=all_chunks,
                    end_time=end_time,
                    model=model,
                )
            )
            standard_logging_response_object = (
                vertex_passthrough_logging_handler_result["result"]
            )
            kwargs = vertex_passthrough_logging_handler_result["kwargs"]
        elif endpoint_type == EndpointType.OPENAI:
            openai_passthrough_logging_handler_result = (
                OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
                    litellm_logging_obj=litellm_logging_obj,
                    passthrough_success_handler_obj=passthrough_success_handler_obj,
                    url_route=url_route,
                    request_body=request_body,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    all_chunks=all_chunks,
                    end_time=end_time,
                )
            )
            standard_logging_response_object = (
                openai_passthrough_logging_handler_result["result"]
            )
            kwargs = openai_passthrough_logging_handler_result["kwargs"]

        if standard_logging_response_object is None:
            standard_logging_response_object = StandardPassThroughResponseObject(
                response=f"cannot parse chunks to standard response object. Chunks={all_chunks}"
            )
        return standard_logging_response_object, kwargs

    @staticmethod
    def _extract_model_for_cost_injection(
        request_body: Optional[dict],
        url_route: str,
        endpoint_type: EndpointType,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> Optional[str]:
        """
        Extract model name for cost injection from various sources.
        """
        # Try to get model from request body
        if request_body:
            model = request_body.get("model")
            if model:
                return model

        # Try to get model from logging object
        if hasattr(litellm_logging_obj, "model_call_details"):
            model = litellm_logging_obj.model_call_details.get("model")
            if model:
                return model

        # For Vertex AI, try to extract from URL
        if endpoint_type == EndpointType.VERTEX_AI:
            model = VertexPassthroughLoggingHandler.extract_model_from_url(url_route)
            if model and model != "unknown":
                return model

        return None

    @staticmethod
    def _convert_raw_bytes_to_str_lines(raw_bytes: List[bytes]) -> List[str]:
        """
        Converts a list of raw bytes into a list of string lines, similar to aiter_lines()

        Args:
            raw_bytes: List of bytes chunks from aiter.bytes()

        Returns:
            List of string lines, with each line being a complete data: {} chunk
        """
        # Combine all bytes and decode to string
        combined_str = b"".join(raw_bytes).decode("utf-8")

        # Split by newlines and filter out empty lines
        lines = [line.strip() for line in combined_str.split("\n") if line.strip()]

        return lines
