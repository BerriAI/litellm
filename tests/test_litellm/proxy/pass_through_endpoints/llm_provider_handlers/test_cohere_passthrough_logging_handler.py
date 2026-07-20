import json
import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.cohere_passthrough_logging_handler import (
    CoherePassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)


class TestCoherePassthroughLoggingHandler:
    """Test the Cohere passthrough logging handler for embed cost tracking."""

    def setup_method(self):
        """Set up test fixtures"""
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.handler = CoherePassthroughLoggingHandler()

        # Mock Cohere embed response
        self.mock_cohere_embed_response = {
            "embeddings": [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9, 1.0],
            ],
            "meta": {
                "billed_units": {
                    "input_tokens": 3,
                }
            },
        }

    def _create_mock_logging_obj(self) -> LiteLLMLoggingObj:
        """Create a mock logging object"""
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        return mock_logging_obj

    def _create_mock_httpx_response(self, response_data: dict = None) -> httpx.Response:
        """Create a mock httpx response"""
        if response_data is None:
            response_data = self.mock_cohere_embed_response

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = json.dumps(response_data)
        mock_response.json.return_value = response_data
        mock_response.headers = {"content-type": "application/json"}
        return mock_response

    def _create_passthrough_logging_payload(self) -> PassthroughStandardLoggingPayload:
        """Create a mock passthrough logging payload"""
        return PassthroughStandardLoggingPayload(
            url="https://api.cohere.com/v1/embed",
            request_body={"model": "embed-english-v3.0", "texts": ["test passthrough"]},
            request_method="POST",
        )

    @patch("litellm.completion_cost")
    @patch(
        "litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload"
    )
    @patch(
        "litellm.llms.cohere.embed.v1_transformation.CohereEmbeddingConfig._transform_response"
    )
    def test_cohere_embed_passthrough_cost_tracking(
        self, mock_transform_response, mock_get_standard_logging, mock_completion_cost
    ):
        """Test successful cost tracking for Cohere embed passthrough"""
        # Arrange
        from litellm.types.utils import EmbeddingResponse

        # Create a mock embedding response
        mock_embedding_response = EmbeddingResponse()
        mock_embedding_response.data = [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
            {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
        ]
        mock_embedding_response.model = "embed-english-v3.0"
        mock_embedding_response.object = "list"
        from litellm.types.utils import Usage

        mock_embedding_response.usage = Usage(
            prompt_tokens=3, completion_tokens=0, total_tokens=3
        )

        mock_transform_response.return_value = mock_embedding_response
        mock_completion_cost.return_value = 3.6e-07  # Expected cost for embed-v4.0
        mock_get_standard_logging.return_value = {"test": "logging_payload"}

        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()

        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
        }

        request_body = {
            "model": "embed-english-v3.0",
            "texts": ["test passthrough"],
        }

        # Act
        result = self.handler.cohere_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_cohere_embed_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.cohere.com/v1/embed",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body=request_body,
            **kwargs,
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["model"] == "embed-english-v3.0"
        assert result["kwargs"]["custom_llm_provider"] == "cohere"

        # Verify cost calculation was called with correct parameters
        mock_completion_cost.assert_called_once()
        call_args = mock_completion_cost.call_args
        assert call_args.kwargs["model"] == "embed-english-v3.0"
        assert call_args.kwargs["custom_llm_provider"] == "cohere"
        assert call_args.kwargs["call_type"] == "aembedding"

        # Verify logging object was updated
        assert mock_logging_obj.model_call_details["response_cost"] == 3.6e-07
        assert mock_logging_obj.model_call_details["model"] == "embed-english-v3.0"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "cohere"

        # Verify result is an EmbeddingResponse
        assert hasattr(result["result"], "data")
        assert hasattr(result["result"], "model")
        assert result["result"].model == "embed-english-v3.0"


class TestCohereStreamingPassthroughCostTracking:
    """Streamed Cohere chat used to record $0.

    `api.cohere.com` classified as `EndpointType.GENERIC`, which reassembles
    nothing and costs nothing, so every streamed Cohere pass-through was billed
    against our Cohere account and logged at zero spend. The handler already
    had a `_build_complete_streaming_response` — it simply had no call site,
    and had never been exercised, which hid two bugs that would have kept it
    returning $0 even once called (v1 iterator with no usage parsing, and no
    handling of SSE `data:` framing).
    """

    URL = "https://api.cohere.com/v2/chat"
    MODEL = "command-r-plus"
    PROMPT_TOKENS = 1000
    COMPLETION_TOKENS = 500

    def _sse_lines(self, include_usage: bool = True) -> list:
        """Raw SSE lines exactly as `_convert_raw_bytes_to_str_lines` yields them."""
        message_end = {
            "type": "message-end",
            "delta": {"finish_reason": "COMPLETE"},
        }
        if include_usage:
            message_end["delta"]["usage"] = {
                "billed_units": {
                    "input_tokens": self.PROMPT_TOKENS,
                    "output_tokens": self.COMPLETION_TOKENS,
                },
                "tokens": {
                    "input_tokens": self.PROMPT_TOKENS,
                    "output_tokens": self.COMPLETION_TOKENS,
                },
            }
        events = [
            ("message-start", {"id": "abc", "type": "message-start"}),
            (
                "content-delta",
                {
                    "type": "content-delta",
                    "index": 0,
                    "delta": {"message": {"content": {"text": "Hello"}}},
                },
            ),
            (
                "content-delta",
                {
                    "type": "content-delta",
                    "index": 0,
                    "delta": {"message": {"content": {"text": " world"}}},
                },
            ),
            ("message-end", message_end),
        ]
        lines = []
        for event_name, data in events:
            lines.append(f"event: {event_name}")
            lines.append(f"data: {json.dumps(data)}")
        return lines

    def _create_mock_logging_obj(self) -> LiteLLMLoggingObj:
        mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {}
        mock_logging_obj.optional_params = {}
        mock_logging_obj.litellm_call_id = "test-call-id-123"
        mock_logging_obj.litellm_trace_id = "test-trace-id-123"
        return mock_logging_obj

    def test_streamed_cohere_chat_is_costed_not_zero(self):
        """The wired dispatch must produce real spend from the billed usage."""
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )
        from litellm.types.passthrough_endpoints.pass_through_endpoints import (
            EndpointType,
        )

        mock_logging_obj = self._create_mock_logging_obj()
        raw_bytes = [("\n".join(self._sse_lines()) + "\n").encode("utf-8")]

        (
            standard_logging_response_object,
            kwargs,
        ) = PassThroughStreamingHandler._build_passthrough_logging_result(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=PassThroughEndpointLogging(),
            url_route=self.URL,
            request_body={"model": self.MODEL},
            endpoint_type=EndpointType.COHERE,
            start_time=datetime.now(),
            raw_bytes=raw_bytes,
            end_time=datetime.now(),
            model=None,
        )

        # Cohere's *billed* token counts must survive, not an estimate.
        usage = standard_logging_response_object.usage
        assert usage.prompt_tokens == self.PROMPT_TOKENS
        assert usage.completion_tokens == self.COMPLETION_TOKENS

        assert kwargs["response_cost"] > 0
        # The pass-through spend path reads cost from model_call_details.
        assert mock_logging_obj.model_call_details["response_cost"] > 0

    def test_streaming_response_builder_handles_sse_framing(self):
        """`_build_complete_streaming_response` receives raw SSE *lines*.

        `convert_str_chunk_to_generic_chunk` only strips a `data:` prefix when
        handed `bytes`, so the `data: {...}` / `event: ...` lines the streaming
        handler collects have to be unwrapped here. Passing them through raw
        raised `JSONDecodeError` on the very first line.
        """
        handler = CoherePassthroughLoggingHandler()
        mock_logging_obj = self._create_mock_logging_obj()

        complete = handler._build_complete_streaming_response(
            all_chunks=self._sse_lines(),
            litellm_logging_obj=mock_logging_obj,
            model=self.MODEL,
        )

        assert complete is not None
        assert complete.usage.prompt_tokens == self.PROMPT_TOKENS
        assert complete.usage.completion_tokens == self.COMPLETION_TOKENS
        assert "Hello world" in complete.choices[0].message.content

    def test_one_malformed_line_does_not_discard_the_stream(self):
        """A single unparseable line must not throw away the rest of the
        stream's usage — that would silently reintroduce a $0 row."""
        handler = CoherePassthroughLoggingHandler()
        mock_logging_obj = self._create_mock_logging_obj()

        lines = self._sse_lines()
        lines.insert(2, "data: {not-valid-json")

        complete = handler._build_complete_streaming_response(
            all_chunks=lines,
            litellm_logging_obj=mock_logging_obj,
            model=self.MODEL,
        )

        assert complete is not None
        assert complete.usage.prompt_tokens == self.PROMPT_TOKENS
        assert complete.usage.completion_tokens == self.COMPLETION_TOKENS

    def test_uses_billed_usage_rather_than_estimating(self):
        """Guards the v1-vs-v2 iterator choice.

        The v1 `ModelResponseIterator` never populates `usage` on any chunk, so
        a response rebuilt with it carries no token counts and gets priced from
        estimated tokens — a number unrelated to what Cohere billed. Only the
        v2 iterator reads the `message-end` usage block.
        """
        handler = CoherePassthroughLoggingHandler()

        with_usage = handler._build_complete_streaming_response(
            all_chunks=self._sse_lines(include_usage=True),
            litellm_logging_obj=self._create_mock_logging_obj(),
            model=self.MODEL,
        )
        without_usage = handler._build_complete_streaming_response(
            all_chunks=self._sse_lines(include_usage=False),
            litellm_logging_obj=self._create_mock_logging_obj(),
            model=self.MODEL,
        )

        # The billed counts are large; an estimate over "Hello world" is tiny.
        assert with_usage.usage.prompt_tokens == self.PROMPT_TOKENS
        assert without_usage.usage.prompt_tokens != self.PROMPT_TOKENS


if __name__ == "__main__":
    pytest.main([__file__])
