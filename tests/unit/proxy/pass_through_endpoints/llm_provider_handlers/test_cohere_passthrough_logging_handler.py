import json
import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.cohere_passthrough_logging_handler import (
    CoherePassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import CallTypes


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
    def test_cohere_embed_passthrough_cost_tracking(self, mock_completion_cost: MagicMock) -> None:
        """Test successful cost tracking for Cohere embed passthrough"""
        mock_completion_cost.return_value = 3.6e-07

        mock_httpx_response: Final = self._create_mock_httpx_response()
        mock_logging_obj: Final = self._create_mock_logging_obj()
        passthrough_payload: Final = self._create_passthrough_logging_payload()

        kwargs: Final = {
            "passthrough_logging_payload": passthrough_payload,
        }

        request_body: Final = {
            "model": "embed-english-v3.0",
            "texts": ["test passthrough"],
        }

        # Act
        result: Final = self.handler.cohere_passthrough_handler(
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
        assert result["result"].hidden_params["response_cost"] == 3.6e-07

        # Verify cost calculation was called with correct parameters
        mock_completion_cost.assert_called_once()
        call_args: Final = mock_completion_cost.call_args
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

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler.BasePassthroughLoggingHandler.passthrough_chat_handler"
    )
    @patch("litellm.completion_cost")
    def test_openai_embeddings_route_does_not_use_cohere_embed_path(self, mock_completion_cost, mock_chat_handler):
        mock_chat_handler.return_value = {"result": None, "kwargs": {}}
        response_body = {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
            "usage": {"prompt_tokens": 6, "total_tokens": 6},
        }
        result = self.handler.cohere_passthrough_handler(
            httpx_response=self._create_mock_httpx_response(response_body),
            response_body=response_body,
            logging_obj=self._create_mock_logging_obj(),
            url_route="https://api.openai.com/v1/embeddings",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "text-embedding-3-small", "input": "PROOF_SENTINEL_TEXT"},
            passthrough_logging_payload=PassthroughStandardLoggingPayload(
                url="https://api.openai.com/v1/embeddings",
                request_body={"model": "text-embedding-3-small", "input": "PROOF_SENTINEL_TEXT"},
                request_method="POST",
            ),
        )
        mock_completion_cost.assert_not_called()
        mock_chat_handler.assert_called_once()
        assert result == {"result": None, "kwargs": {}}


COHERE_EMBED_V2_TEXT_BODY: Final[Mapping[str, object]] = {
    "id": "da6e531f-83f4-4bcb-8e9a-f1a2c6e7a1b3",
    "embeddings": {"float": [[0.1, 0.2], [0.3, 0.4]]},
    "texts": ["hello world", "second text"],
    "meta": {"api_version": {"version": "2"}, "billed_units": {"input_tokens": 9}},
    "response_type": "embeddings_by_type",
}
COHERE_EMBED_V2_IMAGE_BODY: Final[Mapping[str, object]] = {
    "id": "5807ee2e-0cda-445a-9ec8-864c60a06606",
    "embeddings": {"float": [[0.125, -0.375]]},
    "texts": [],
    "images": [{"width": 64, "height": 64, "format": "image/png", "bit_depth": 8}],
    "meta": {"api_version": {"version": "2"}, "billed_units": {"input_tokens": 0, "image_tokens": 6}},
    "response_type": "embeddings_by_type",
}
COHERE_EMBED_V1_FLOATS_BODY: Final[Mapping[str, object]] = {
    "id": "5d4e1f3a-7c1b-4f3e-9a3e-1b2c3d4e5f60",
    "embeddings": [[0.1, 0.2], [0.3, 0.4]],
    "texts": ["hello world", "second text"],
    "meta": {"api_version": {"version": "1"}, "billed_units": {"input_tokens": 3}},
    "response_type": "embeddings_floats",
}


_CALL_START: Final = datetime(2026, 10, 1, 12, 0, 0)
_CALL_END: Final = datetime(2026, 10, 1, 12, 0, 1)


def _passthrough_logging_obj() -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "no-message-pass-through-endpoint"}],
        stream=False,
        call_type=CallTypes.pass_through.value,
        start_time=_CALL_START,
        litellm_call_id="cohere-embed-passthrough-call",
        function_id="cohere-embed-passthrough-function",
    )


@pytest.mark.parametrize(
    ("url_route", "body", "input_tokens", "image_tokens"),
    [
        ("https://api.cohere.com/v2/embed", COHERE_EMBED_V2_TEXT_BODY, 9, 0),
        ("https://api.cohere.com/v2/embed", COHERE_EMBED_V2_IMAGE_BODY, 0, 6),
        ("https://api.cohere.com/v1/embed", COHERE_EMBED_V1_FLOATS_BODY, 3, 0),
    ],
    ids=["v2_text", "v2_image", "v1_floats"],
)
def test_cohere_embed_passthrough_is_costed_from_billed_units(
    url_route: str, body: Mapping[str, object], input_tokens: int, image_tokens: int
) -> None:
    request_body: Final = {
        "model": "embed-v4.0",
        "texts": ["hello world", "second text"],
        "input_type": "search_document",
    }
    model_info: Final = litellm.get_model_info("cohere/embed-v4.0")
    expected_cost: Final = (
        input_tokens * model_info["input_cost_per_token"] + image_tokens * model_info["input_cost_per_image_token"]
    )

    logged: Final = PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
        httpx_response=httpx.Response(200, json=body, request=httpx.Request("POST", url_route)),
        response_body=dict(body),
        request_body=request_body,
        logging_obj=_passthrough_logging_obj(),
        url_route=url_route,
        result="",
        start_time=_CALL_START,
        end_time=_CALL_END,
        cache_hit=False,
    )

    assert expected_cost > 0
    assert logged["kwargs"]["custom_llm_provider"] == "cohere"
    assert logged["kwargs"]["response_cost"] == pytest.approx(expected_cost)



def _normalize_cohere_embed(url_route: str, body: Mapping[str, object], model: str) -> Mapping[str, object]:
    return PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
        httpx_response=httpx.Response(200, json=body, request=httpx.Request("POST", url_route)),
        response_body=dict(body),
        request_body={"model": model, "texts": ["hello world"], "input_type": "search_document"},
        logging_obj=_passthrough_logging_obj(),
        url_route=url_route,
        result="",
        start_time=_CALL_START,
        end_time=_CALL_END,
        cache_hit=False,
    )


def test_cohere_v3_image_embed_passthrough_is_costed_per_billed_image() -> None:
    body: Final = {
        "id": "8a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d",
        "embeddings": {"float": [[0.125, -0.375]]},
        "texts": [],
        "images": [{"width": 64, "height": 64, "format": "image/png", "bit_depth": 8}],
        "meta": {"api_version": {"version": "2"}, "billed_units": {"images": 1}},
        "response_type": "embeddings_by_type",
    }
    per_image: Final = litellm.get_model_info("embed-english-v3.0", custom_llm_provider="cohere")["input_cost_per_image"]

    logged: Final = _normalize_cohere_embed("https://api.cohere.com/v2/embed", body, "embed-english-v3.0")

    assert per_image is not None and per_image > 0
    assert logged["kwargs"]["response_cost"] == pytest.approx(per_image)


def test_cohere_embed_passthrough_rows_keep_their_embedding_type() -> None:
    body: Final = {
        **COHERE_EMBED_V2_TEXT_BODY,
        "embeddings": {"float": [[0.1, 0.2], [0.3, 0.4]], "int8": [[1, 2], [3, 4]]},
    }

    logged: Final = _normalize_cohere_embed("https://api.cohere.com/v2/embed", body, "embed-v4.0")

    assert [(row["type"], row["index"], row["embedding"]) for row in logged["standard_logging_response_object"].data] == [
        ("float", 0, [0.1, 0.2]),
        ("float", 1, [0.3, 0.4]),
        ("int8", 0, [1, 2]),
        ("int8", 1, [3, 4]),
    ]


def test_unpriceable_cohere_embed_passthrough_falls_back_to_the_plain_passthrough_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        logged: Final = _normalize_cohere_embed(
            "https://api.cohere.com/v2/embed", COHERE_EMBED_V2_TEXT_BODY, "embed-model-missing-from-the-cost-map"
        )

    assert logged["standard_logging_response_object"] is None
    assert "response_cost" not in logged["kwargs"]
    assert any(
        "Could not price the Cohere passthrough embed response" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


@pytest.mark.parametrize(
    ("url_route", "body"),
    [
        ("https://api.cohere.com/v1/embed-jobs", {"job_id": "job-1", "meta": {"api_version": {"version": "1"}}}),
        (
            "https://api.cohere.com/v1/embed-jobs/job-1",
            {"job_id": "job-1", "status": "processing", "model": "embed-english-v3.0"},
        ),
    ],
)
def test_cohere_embed_job_passthrough_logs_the_plain_passthrough_row(
    url_route: str, body: Mapping[str, object]
) -> None:
    logged: Final = _normalize_cohere_embed(url_route, body, "embed-english-v3.0")

    assert logged["standard_logging_response_object"] is None
    assert "response_cost" not in logged["kwargs"]

if __name__ == "__main__":
    pytest.main([__file__])
