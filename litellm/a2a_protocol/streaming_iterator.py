"""
A2A Streaming Iterator with token tracking and logging support.
"""

import asyncio
from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Final

import anyio
from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.a2a_protocol.cost_calculator import A2ACostCalculator
from litellm.a2a_protocol.utils import A2ARequestUtils
from litellm.constants import A2A_COMPLETION_BRIDGE_WORKING_MESSAGE
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

if TYPE_CHECKING:
    from a2a.compat.v0_3.types import SendStreamingMessageRequest, SendStreamingMessageResponse

_OPAQUE_PAYLOAD: Final = TypeAdapter(object)


def _chunk_payload(chunk: "SendStreamingMessageResponse | Mapping[str, object]") -> Mapping[str, object]:
    if isinstance(chunk, Mapping):
        return chunk
    return chunk.model_dump(mode="json", exclude_none=True) if hasattr(chunk, "model_dump") else {}


def _artifact_update(payload: Mapping[str, object]) -> tuple[str, bool] | None:
    result: Final = as_str_mapping(payload.get("result"))
    if result is None or result.get("kind") != "artifact-update":
        return None
    artifact: Final = as_str_mapping(result.get("artifact"))
    artifact_id: Final = artifact.get("artifactId") if artifact is not None else None
    if not isinstance(artifact_id, str):
        return None
    return artifact_id, result.get("append") is True


class A2AStreamingIterator:
    """
    Async iterator for A2A streaming responses with token tracking.

    Collects chunks, extracts text, and logs usage on completion.
    """

    def __init__(
        self,
        stream: AsyncIterator["SendStreamingMessageResponse | Mapping[str, object]"],
        request: "SendStreamingMessageRequest",
        logging_obj: LiteLLMLoggingObj,
        agent_name: str = "unknown",
    ):
        self.stream = stream
        self.request = request
        self.logging_obj = logging_obj
        self.agent_name = agent_name
        self.start_time = logging_obj.start_time

        # Collect chunks for token counting
        self.chunks: list[SendStreamingMessageResponse | Mapping[str, object]] = []
        self.latest_message_text = ""
        self.artifact_texts: Mapping[str, str] = {}
        self.saw_error = False
        self.final_chunk: SendStreamingMessageResponse | Mapping[str, object] | None = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> "SendStreamingMessageResponse | Mapping[str, object]":
        try:
            chunk: Final = await self.stream.__anext__()
            if self.logging_obj.completion_start_time is None:
                self.logging_obj.update_completion_start_time(completion_start_time=datetime.now())

            # Store chunk
            self.chunks.append(chunk)
            payload: Final = _chunk_payload(chunk)
            self.saw_error = self.saw_error or payload.get("error") is not None

            # Extract text from chunk for token counting
            self._collect_text_from_chunk(payload)

            # Check if this is the final chunk (completed status)
            if self._is_completed_chunk(payload):
                self.final_chunk = chunk

            return chunk

        except StopAsyncIteration:
            await self._handle_stream_complete()
            raise

    async def abill_on_disconnect(self) -> None:
        if not self.chunks:
            return
        with anyio.CancelScope(shield=True):
            await self._handle_stream_complete()

    def _collect_text_from_chunk(self, payload: Mapping[str, object]) -> None:
        """Record the reply text a streaming chunk carries."""
        try:
            text: Final = A2ARequestUtils.extract_text_from_stream_event(payload)
            if not text:
                return
            artifact_update: Final = _artifact_update(payload)
            if artifact_update is None and text == A2A_COMPLETION_BRIDGE_WORKING_MESSAGE:
                return
            if artifact_update is None:
                self.latest_message_text = text
                return
            artifact_id, append = artifact_update
            artifact_text: Final = self.artifact_texts.get(artifact_id, "") + text if append else text
            self.artifact_texts = {**self.artifact_texts, artifact_id: artifact_text}
        except Exception:
            verbose_logger.debug("Failed to extract text from A2A streaming chunk")

    def _reply_text(self) -> str:
        artifact_text: Final = " ".join(text for text in self.artifact_texts.values() if text)
        return artifact_text or self.latest_message_text

    def _is_completed_chunk(self, payload: Mapping[str, object]) -> bool:
        """Check if chunk indicates stream completion."""
        try:
            result: Final = payload.get("result", {})
            if isinstance(result, dict):
                status: Final = result.get("status", {})
                if isinstance(status, dict):
                    return status.get("state") == "completed"
        except Exception:
            pass
        return False

    async def _handle_stream_complete(self) -> None:
        """Handle logging and token counting when stream completes."""
        if self.final_chunk is None and self.chunks:
            self.final_chunk = self.chunks[-1]
        try:
            end_time: Final = datetime.now()

            # Calculate tokens from collected text
            input_message: Final = A2ARequestUtils.get_input_message_from_request(self.request)
            input_text: Final = A2ARequestUtils.extract_text_from_message(input_message)
            prompt_tokens: Final = await asyncify(A2ARequestUtils.count_tokens)(input_text)

            completion_tokens: Final = await asyncify(A2ARequestUtils.count_tokens)(self._reply_text())

            total_tokens: Final = prompt_tokens + completion_tokens

            # Create usage object
            usage: Final = litellm.Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

            # Set usage on logging obj
            self.logging_obj.model_call_details["usage"] = usage
            # Mark stream flag for downstream callbacks
            self.logging_obj.model_call_details["stream"] = False

            # Calculate cost using A2ACostCalculator
            response_cost: Final = 0.0 if self.saw_error else A2ACostCalculator.calculate_a2a_cost(self.logging_obj)
            self.logging_obj.model_call_details["response_cost"] = response_cost

            # Build result for logging
            result: Final = self._build_logging_result(usage)

            # Call success handlers - they will build standard_logging_object
            asyncio.create_task(
                self.logging_obj.dispatch_success_handlers(
                    result,
                    start_time=self.start_time,
                    end_time=end_time,
                    cache_hit=None,
                    prefer_async_handlers=True,
                )
            )

            verbose_logger.info(
                "A2A streaming completed: prompt_tokens=%s, completion_tokens=%s, total_tokens=%s, response_cost=%s",
                prompt_tokens,
                completion_tokens,
                total_tokens,
                response_cost,
            )

        except Exception as e:
            verbose_logger.debug("Error in A2A streaming completion handler: %s", e)

    def _build_logging_result(self, usage: litellm.Usage) -> dict[str, object]:
        """Build a result dict for logging."""
        result: Final[dict[str, object]] = {
            "id": getattr(self.request, "id", "unknown"),
            "jsonrpc": "2.0",
            "usage": (usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)),
        }

        # Add final chunk result if available
        if self.final_chunk:
            try:
                chunk_dict: Final = as_str_mapping(_OPAQUE_PAYLOAD.validate_python(_chunk_payload(self.final_chunk)))
                if chunk_dict is not None:
                    result["result"] = chunk_dict.get("result", {})
            except Exception:
                pass

        return result
