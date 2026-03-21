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
    @patch("litellm.llms.cohere.embed.v1_transformation.CohereEmbeddingConfig._transform_response")
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


if __name__ == "__main__":
    pytest.main([__file__])

