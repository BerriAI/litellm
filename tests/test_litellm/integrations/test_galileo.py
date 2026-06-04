import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.galileo import GalileoObserve
from litellm.types.utils import (
    Choices,
    EmbeddingResponse,
    ImageObject,
    ImageResponse,
    Message,
    ModelResponse,
    TextCompletionResponse,
)


@pytest.fixture
def galileo_v2_env(monkeypatch):
    monkeypatch.setenv("GALILEO_API_KEY", "test-api-key")
    monkeypatch.setenv("GALILEO_PROJECT_ID", "86ff8ebe-a297-4134-b167-748bdd8d2c20")
    monkeypatch.setenv("GALILEO_LOG_STREAM_ID", "76c4ea50-8aa3-4771-a0d7-8567b112210f")
    monkeypatch.setenv("GALILEO_BASE_URL", "https://api.galileo.ai")


@pytest.mark.asyncio
async def test_galileo_v2_ingest_url_and_headers(galileo_v2_env):
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

    assert await logger._ensure_headers() is True
    assert logger.headers["Galileo-API-Key"] == "test-api-key"


def test_galileo_v2_span_preserves_message_roles(galileo_v2_env):
    record = {
        "latency_ms": 1,
        "status_code": 200,
        "input_text": "fallback",
        "output_text": "ok",
        "node_type": "acompletion",
        "model": "gpt-5.2",
        "num_input_tokens": 0,
        "num_output_tokens": 0,
        "created_at": "2026-05-25T12:00:00",
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
        ],
    }
    span = GalileoObserve._record_to_v2_span(record)
    assert span["input"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hello"},
    ]


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

    with patch.object(
        logger.async_httpx_handler, "post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.side_effect = Exception("404 Not Found")
        await logger.flush_in_memory_records()

    assert len(logger.in_memory_records) == 1


@pytest.mark.asyncio
async def test_galileo_flush_clears_records_on_201(galileo_v2_env):
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
    mock_response.is_success = True
    mock_response.status_code = 201

    with patch.object(
        logger.async_httpx_handler, "post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.return_value = mock_response
        await logger.flush_in_memory_records()

    assert logger.in_memory_records == []


def test_galileo_normalize_base_url_none(monkeypatch):
    monkeypatch.delenv("GALILEO_API_KEY", raising=False)
    monkeypatch.delenv("GALILEO_BASE_URL", raising=False)
    monkeypatch.delenv("GALILEO_PROJECT_ID", raising=False)
    logger = GalileoObserve()
    assert logger.base_url is None
    assert logger._normalize_base_url(None) is None
    assert logger._normalize_base_url("https://x.example/") == "https://x.example"


def test_galileo_is_configured_branches(monkeypatch):
    monkeypatch.delenv("GALILEO_API_KEY", raising=False)
    monkeypatch.delenv("GALILEO_BASE_URL", raising=False)
    monkeypatch.delenv("GALILEO_PROJECT_ID", raising=False)
    monkeypatch.delenv("GALILEO_USERNAME", raising=False)
    monkeypatch.delenv("GALILEO_PASSWORD", raising=False)

    no_env = GalileoObserve()
    assert no_env._is_configured() is False

    monkeypatch.setenv("GALILEO_API_KEY", "k")
    monkeypatch.setenv("GALILEO_PROJECT_ID", "p")
    v2 = GalileoObserve()
    assert v2._is_configured() is True

    monkeypatch.delenv("GALILEO_API_KEY", raising=False)
    monkeypatch.setenv("GALILEO_USERNAME", "u")
    monkeypatch.setenv("GALILEO_PASSWORD", "pw")
    monkeypatch.setenv("GALILEO_BASE_URL", "https://galileo.example")
    legacy = GalileoObserve()
    assert legacy._is_configured() is True

    monkeypatch.delenv("GALILEO_PASSWORD", raising=False)
    no_pw = GalileoObserve()
    assert no_pw._is_configured() is False


def test_galileo_input_messages_fallbacks():
    assert GalileoObserve._galileo_input_messages(None, "hi") == [
        {"role": "user", "content": "hi"}
    ]
    assert GalileoObserve._galileo_input_messages(
        ["not-a-dict", {"content": "no role"}], "fallback"
    ) == [{"role": "user", "content": "fallback"}]


def test_galileo_record_to_v2_span_with_tags_and_offset():
    span = GalileoObserve._record_to_v2_span(
        {
            "latency_ms": 5,
            "status_code": 200,
            "input_text": "in",
            "output_text": "out",
            "node_type": "acompletion",
            "model": "gpt-5.2",
            "num_input_tokens": 1,
            "num_output_tokens": 2,
            "created_at": "2026-05-25T12:00:00",
            "tags": ["t1"],
        }
    )
    assert span["tags"] == ["t1"]
    assert span["created_at"].endswith("Z")

    offset = GalileoObserve._record_to_v2_span(
        {"created_at": "2026-05-25T12:00:00-05:00"}
    )
    assert offset["created_at"] == "2026-05-25T12:00:00-05:00"


def test_galileo_get_output_str_variants(galileo_v2_env):
    logger = GalileoObserve()
    assert logger.get_output_str_from_response(None, {}) is None
    assert (
        logger.get_output_str_from_response(
            EmbeddingResponse(), {"call_type": "embedding"}
        )
        is None
    )

    text_resp = TextCompletionResponse()
    text_resp.choices = [MagicMock(text="text-completion-output")]
    assert (
        logger.get_output_str_from_response(text_resp, {"call_type": "text_completion"})
        == "text-completion-output"
    )

    image_resp = ImageResponse(data=[ImageObject(url="https://x/y.png")])
    assert "y.png" in logger.get_output_str_from_response(image_resp, {})

    assert logger.get_output_str_from_response("not-a-supported-type", {}) is None


def test_galileo_get_ingest_request_unconfigured(monkeypatch):
    monkeypatch.delenv("GALILEO_API_KEY", raising=False)
    monkeypatch.delenv("GALILEO_BASE_URL", raising=False)
    monkeypatch.delenv("GALILEO_PROJECT_ID", raising=False)
    logger = GalileoObserve()
    assert logger._get_ingest_request() is None


def test_galileo_get_ingest_request_legacy(monkeypatch):
    monkeypatch.delenv("GALILEO_API_KEY", raising=False)
    monkeypatch.setenv("GALILEO_USERNAME", "u")
    monkeypatch.setenv("GALILEO_PASSWORD", "pw")
    monkeypatch.setenv("GALILEO_BASE_URL", "https://galileo.example/")
    monkeypatch.setenv("GALILEO_PROJECT_ID", "proj")
    logger = GalileoObserve()
    logger.in_memory_records = [{"foo": "bar"}]
    url, payload = logger._get_ingest_request()
    assert url == "https://galileo.example/projects/proj/observe/ingest"
    assert payload == {"records": [{"foo": "bar"}]}


@pytest.mark.asyncio
async def test_galileo_ensure_headers_v2_missing_key(monkeypatch):
    monkeypatch.delenv("GALILEO_API_KEY", raising=False)
    monkeypatch.setenv("GALILEO_PROJECT_ID", "p")
    monkeypatch.setenv("GALILEO_BASE_URL", "https://x")
    logger = GalileoObserve()
    logger.use_v2_api = True
    logger.api_key = None
    assert await logger._ensure_headers() is False


@pytest.mark.asyncio
async def test_galileo_ensure_headers_cached(galileo_v2_env):
    logger = GalileoObserve()
    logger.headers = {"Galileo-API-Key": "already-set"}
    assert await logger._ensure_headers() is True


@pytest.mark.asyncio
async def test_galileo_ensure_headers_legacy_login(monkeypatch):
    monkeypatch.delenv("GALILEO_API_KEY", raising=False)
    monkeypatch.setenv("GALILEO_USERNAME", "u")
    monkeypatch.setenv("GALILEO_PASSWORD", "pw")
    monkeypatch.setenv("GALILEO_BASE_URL", "https://galileo.example")
    monkeypatch.setenv("GALILEO_PROJECT_ID", "p")
    logger = GalileoObserve()

    login_resp = MagicMock()
    login_resp.raise_for_status = MagicMock()
    login_resp.json = MagicMock(return_value={"access_token": "tok"})

    with patch.object(
        logger.async_httpx_handler, "post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.return_value = login_resp
        assert await logger._ensure_headers() is True

    assert logger.headers["Authorization"] == "Bearer tok"


@pytest.mark.asyncio
async def test_galileo_ensure_headers_legacy_login_failure(monkeypatch):
    monkeypatch.delenv("GALILEO_API_KEY", raising=False)
    monkeypatch.setenv("GALILEO_USERNAME", "u")
    monkeypatch.setenv("GALILEO_PASSWORD", "pw")
    monkeypatch.setenv("GALILEO_BASE_URL", "https://galileo.example")
    monkeypatch.setenv("GALILEO_PROJECT_ID", "p")
    logger = GalileoObserve()

    with patch.object(
        logger.async_httpx_handler, "post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.side_effect = Exception("boom")
        assert await logger._ensure_headers() is False


@pytest.mark.asyncio
async def test_galileo_flush_noop_when_unconfigured(monkeypatch):
    monkeypatch.delenv("GALILEO_API_KEY", raising=False)
    monkeypatch.delenv("GALILEO_BASE_URL", raising=False)
    monkeypatch.delenv("GALILEO_PROJECT_ID", raising=False)
    logger = GalileoObserve()
    logger.in_memory_records = [{"foo": "bar"}]
    await logger.flush_in_memory_records()
    assert logger.in_memory_records == [{"foo": "bar"}]


@pytest.mark.asyncio
async def test_galileo_flush_resets_headers_on_401(monkeypatch):
    monkeypatch.delenv("GALILEO_API_KEY", raising=False)
    monkeypatch.setenv("GALILEO_USERNAME", "u")
    monkeypatch.setenv("GALILEO_PASSWORD", "pw")
    monkeypatch.setenv("GALILEO_BASE_URL", "https://galileo.example")
    monkeypatch.setenv("GALILEO_PROJECT_ID", "p")
    logger = GalileoObserve()
    logger.headers = {"Authorization": "Bearer stale"}
    logger.in_memory_records = [{"records": "x"}]

    mock_response = MagicMock()
    mock_response.is_success = False
    mock_response.status_code = 401
    mock_response.text = "unauthorized"

    with patch.object(
        logger.async_httpx_handler, "post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.return_value = mock_response
        await logger.flush_in_memory_records()

    assert logger.headers is None
    assert logger.in_memory_records == [{"records": "x"}]


@pytest.mark.asyncio
async def test_galileo_async_log_success_appends_and_flushes(galileo_v2_env):
    import datetime

    logger = GalileoObserve()
    response = ModelResponse(
        choices=[
            Choices(message=Message(content="reply", role="assistant", annotations=[]))
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 2},
    )

    flushed_url: dict = {}
    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.status_code = 200

    async def fake_post(**kwargs):
        flushed_url["url"] = kwargs.get("url")
        return mock_response

    with patch.object(logger.async_httpx_handler, "post", side_effect=fake_post):
        await logger.async_log_success_event(
            kwargs={
                "call_type": "acompletion",
                "model": "gpt",
                "messages": [{"role": "user", "content": "hi"}],
            },
            response_obj=response,
            start_time=datetime.datetime(2026, 5, 25, 12, 0, 0),
            end_time=datetime.datetime(2026, 5, 25, 12, 0, 1),
        )

    assert "/v2/projects/" in flushed_url["url"]
    assert logger.in_memory_records == []
