import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.galileo import GalileoObserve
from litellm.types.utils import Choices, Message, ModelResponse


@pytest.fixture
def galileo_v2_env(monkeypatch):
    monkeypatch.setenv("GALILEO_API_KEY", "test-api-key")
    monkeypatch.setenv("GALILEO_PROJECT_ID", "86ff8ebe-a297-4134-b167-748bdd8d2c20")
    monkeypatch.setenv("GALILEO_LOG_STREAM_ID", "76c4ea50-8aa3-4771-a0d7-8567b112210f")
    monkeypatch.setenv("GALILEO_BASE_URL", "https://api.galileo.ai")


def test_galileo_v2_ingest_url_and_headers(galileo_v2_env):
    logger = GalileoObserve()
    logger.in_memory_records = [
        {
            "latency_ms": 100,
            "status_code": 200,
            "input_text": "hi",
            "output_text": "hello",
            "node_type": "acompletion",
            "model": "gpt-5.2",
            "num_input_tokens": 1,
            "num_output_tokens": 2,
            "created_at": "2026-05-25T12:00:00",
        }
    ]

    url, payload = logger._get_ingest_request()
    assert (
        url
        == "https://api.galileo.ai/v2/projects/86ff8ebe-a297-4134-b167-748bdd8d2c20/spans"
    )
    assert payload["log_stream_id"] == "76c4ea50-8aa3-4771-a0d7-8567b112210f"
    assert payload["spans"][0]["type"] == "llm"
    assert payload["spans"][0]["output"]["content"] == "hello"

    assert logger._ensure_headers() is True
    assert logger.headers["Galileo-API-Key"] == "test-api-key"


def test_galileo_output_text_from_model_response(galileo_v2_env):
    logger = GalileoObserve()
    response = ModelResponse(
        choices=[
            Choices(
                message=Message(
                    content="assistant reply",
                    role="assistant",
                    annotations=[],
                )
            )
        ]
    )

    output = logger.get_output_str_from_response(response, {"call_type": "acompletion"})
    assert output == "assistant reply"


@pytest.mark.asyncio
async def test_galileo_flush_swallows_http_errors(galileo_v2_env):
    logger = GalileoObserve()
    logger.in_memory_records = [
        {
            "latency_ms": 1,
            "status_code": 200,
            "input_text": "a",
            "output_text": "b",
            "node_type": "acompletion",
            "model": "gpt-5.2",
            "num_input_tokens": 0,
            "num_output_tokens": 0,
            "created_at": "2026-05-25T12:00:00",
        }
    ]

    mock_response = AsyncMock()
    mock_response.status_code = 404
    mock_response.text = "not found"

    with patch.object(
        logger.async_httpx_handler, "post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.side_effect = Exception("404 Not Found")
        await logger.flush_in_memory_records()

    assert len(logger.in_memory_records) == 1
