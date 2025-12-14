"""
A2A Streaming Iterator with token tracking and logging support.
"""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

import litellm
from litellm._logging import verbose_logger
from litellm.a2a_protocol.cost_calculator import A2ACostCalculator
from litellm.a2a_protocol.utils import A2ARequestUtils
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.thread_pool_executor import executor

if TYPE_CHECKING:
    from a2a.types import SendStreamingMessageRequest, SendStreamingMessageResponse


class A2AStreamingIterator:
    """
    Async iterator for A2A streaming responses with token tracking.

    Collects chunks, extracts text, and logs usage on completion.
    """

    def __init__(
        self,
        stream: AsyncIterator["SendStreamingMessageResponse"],
        request: "SendStreamingMessageRequest",
        logging_obj: LiteLLMLoggingObj,
        agent_name: str = "unknown",
    ):
        self.stream = stream
        self.request = request
        self.logging_obj = logging_obj
        self.agent_name = agent_name
        self.start_time = datetime.now()

        # Collect chunks for token counting
        self.chunks: List[Any] = []
        self.collected_text_parts: List[str] = []
        self.final_chunk: Optional[Any] = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> "SendStreamingMessageResponse":
        try:
            chunk = await self.stream.__anext__()

            # Store chunk
            self.chunks.append(chunk)

            # Extract text from chunk for token counting
            self._collect_text_from_chunk(chunk)

            # Check if this is the final chunk (completed status)
            if self._is_completed_chunk(chunk):
                self.final_chunk = chunk

            return chunk

        except StopAsyncIteration:
            # Stream ended - handle logging
            if self.final_chunk is None and self.chunks:
                self.final_chunk = self.chunks[-1]
            await self._handle_stream_complete()
            raise

    def _collect_text_from_chunk(self, chunk: Any) -> None:
        """Extract text from a streaming chunk and add to collected parts."""
        try:
            chunk_dict = chunk.model_dump(mode="json", exclude_none=True) if hasattr(chunk, "model_dump") else {}
            text = A2ARequestUtils.extract_text_from_response(chunk_dict)
            if text:
                self.collected_text_parts.append(text)
        except Exception:
            verbose_logger.debug("Failed to extract text from A2A streaming chunk")

    def _is_completed_chunk(self, chunk: Any) -> bool:
        """Check if chunk indicates stream completion."""
        try:
            chunk_dict = chunk.model_dump(mode="json", exclude_none=True) if hasattr(chunk, "model_dump") else {}
            result = chunk_dict.get("result", {})
            if isinstance(result, dict):
                status = result.get("status", {})
                if isinstance(status, dict):
                    return status.get("state") == "completed"
        except Exception:
            pass
        return False

    async def _handle_stream_complete(self) -> None:
        """Handle logging and token counting when stream completes."""
        try:
            end_time = datetime.now()

            # Calculate tokens from collected text
            input_message = A2ARequestUtils.get_input_message_from_request(self.request)
            input_text = A2ARequestUtils.extract_text_from_message(input_message)
            prompt_tokens = A2ARequestUtils.count_tokens(input_text)

            # Use the last (most complete) text from chunks
            output_text = self.collected_text_parts[-1] if self.collected_text_parts else ""
            completion_tokens = A2ARequestUtils.count_tokens(output_text)

            total_tokens = prompt_tokens + completion_tokens

            # Create usage object
            usage = litellm.Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

            # Set usage on logging obj
            self.logging_obj.model_call_details["usage"] = usage
            # Mark stream flag for downstream callbacks
            self.logging_obj.model_call_details["stream"] = False

            # Calculate cost using A2ACostCalculator
            response_cost = A2ACostCalculator.calculate_a2a_cost(self.logging_obj)
            self.logging_obj.model_call_details["response_cost"] = response_cost

            # Build result for logging
            result = self._build_logging_result(usage)

            # Call success handlers - they will build standard_logging_object
            asyncio.create_task(
                self.logging_obj.async_success_handler(
                    result=result,
                    start_time=self.start_time,
                    end_time=end_time,
                    cache_hit=None,
                )
            )

            executor.submit(
                self.logging_obj.success_handler,
                result=result,
                cache_hit=None,
                start_time=self.start_time,
                end_time=end_time,
            )

            verbose_logger.info(
                f"A2A streaming completed: prompt_tokens={prompt_tokens}, "
                f"completion_tokens={completion_tokens}, total_tokens={total_tokens}, "
                f"response_cost={response_cost}"
            )

        except Exception as e:
            verbose_logger.debug(f"Error in A2A streaming completion handler: {e}")

    def _build_logging_result(self, usage: litellm.Usage) -> Dict[str, Any]:
        """Build a result dict for logging."""
        result: Dict[str, Any] = {
            "id": getattr(self.request, "id", "unknown"),
            "jsonrpc": "2.0",
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else dict(usage),
        }

        # Add final chunk result if available
        if self.final_chunk:
            try:
                chunk_dict = self.final_chunk.model_dump(mode="json", exclude_none=True)
                result["result"] = chunk_dict.get("result", {})
            except Exception:
                pass

        return result

