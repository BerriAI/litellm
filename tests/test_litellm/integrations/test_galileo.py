import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.galileo import GalileoObserve
from litellm.types.llms.openai import HttpxBinaryResponseContent, ResponsesAPIResponse
from litellm.types.rerank import RerankResponse
from litellm.types.utils import (
    Choices,
    EmbeddingResponse,
    ImageObject,
    ImageResponse,
    Message,
    ModelResponse,
    TextCompletionResponse,
    TranscriptionResponse,
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
        == "https://api.galileo.ai/ingest/traces/86ff8ebe-a297-4134-b167-748bdd8d2c20"
    )
    assert payload["log_stream_id"] == "76c4ea50-8aa3-4771-a0d7-8567b112210f"
    assert payload["is_complete"] is True
    assert payload["traces"][0]["type"] == "trace"
    assert payload["traces"][0]["spans"][0]["type"] == "llm"
    assert payload["traces"][0]["spans"][0]["output"]["content"] == "hello"
    assert payload["traces"][0]["spans"][0]["metrics"]["num_total_tokens"] == 3
    assert payload["traces"][0]["metrics"]["num_input_tokens"] == 1
    assert payload["traces"][0]["metrics"]["num_output_tokens"] == 2
    assert payload["traces"][0]["metrics"]["num_total_tokens"] == 3
    assert payload["traces"][0]["spans"][0]["trace_id"] == payload["traces"][0]["id"]

    assert await logger._ensure_headers() is True
    assert logger.headers["Galileo-API-Key"] == "test-api-key"


def test_galileo_token_metrics_from_record_falls_back_to_sum():
    metrics = GalileoObserve._token_metrics_from_record(
        {"num_input_tokens": 5, "num_output_tokens": 7}
    )
    assert metrics == {
        "num_input_tokens": 5,
        "num_output_tokens": 7,
        "num_total_tokens": 12,
    }


def test_galileo_token_metrics_from_record_sums_zero_total():
    metrics = GalileoObserve._token_metrics_from_record(
        {"num_input_tokens": 5, "num_output_tokens": 7, "num_total_tokens": 0}
    )
    assert metrics == {
        "num_input_tokens": 5,
        "num_output_tokens": 7,
        "num_total_tokens": 12,
    }


def test_galileo_token_metrics_from_record_includes_cost():
    metrics = GalileoObserve._token_metrics_from_record(
        {
            "num_input_tokens": 1,
            "num_output_tokens": 2,
            "num_total_tokens": 3,
            "cost": 0.000855,
        }
    )
    assert metrics["cost"] == 0.000855


def test_galileo_input_text_from_messages():
    assert GalileoObserve._input_text_from_messages("hello") == "hello"
    assert (
        GalileoObserve._input_text_from_messages(
            [{"role": "user", "content": "test responses api 1"}]
        )
        == "test responses api 1"
    )


def test_galileo_get_output_str_responses_api(galileo_v2_env):
    from litellm.types.llms.openai import ResponsesAPIResponse

    logger = GalileoObserve()
    resp_dict = {
        "id": "resp_123",
        "created_at": 1,
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hi! How can I help?",
                        "annotations": [],
                    }
                ],
            }
        ],
    }
    response = ResponsesAPIResponse(**resp_dict)
    result = logger.get_output_str_from_response(response, {"call_type": "aresponses"})
    assert result is not None
    assert '"Hi! How can I help?"' in result
    assert '"type": "message"' in result


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
    span = GalileoObserve._record_to_v2_span(
        record, trace_id="trace-id", span_id="span-id"
    )
    assert span["input"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hello"},
    ]


def test_galileo_v2_span_unwraps_prompt_messages(galileo_v2_env):
    record = {
        "latency_ms": 1,
        "status_code": 200,
        "input_text": "fallback",
        "output_text": "ok",
        "node_type": "pass_through_endpoint",
        "model": "gpt-5.2",
        "num_input_tokens": 0,
        "num_output_tokens": 0,
        "created_at": "2026-05-25T12:00:00",
        "messages": {
            "messages": [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hello"},
            ]
        },
    }
    span = GalileoObserve._record_to_v2_span(
        record, trace_id="trace-id", span_id="span-id"
    )
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
    assert output is not None
    assert '"assistant reply"' in output


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


def test_galileo_format_created_at_converts_local_naive_to_utc():
    from datetime import timedelta

    ist = timezone(timedelta(hours=5, minutes=30))

    with patch.object(GalileoObserve, "_local_timezone", return_value=ist):
        local_naive = datetime(2026, 6, 4, 9, 44, 49)
        assert GalileoObserve._format_created_at(local_naive) == "2026-06-04T04:14:49Z"

    aware_utc = datetime(2026, 6, 4, 4, 14, 49, tzinfo=timezone.utc)
    assert GalileoObserve._format_created_at(aware_utc) == "2026-06-04T04:14:49Z"


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
        },
        trace_id="trace-id",
        span_id="span-id",
    )
    assert span["tags"] == ["t1"]
    assert span["created_at"].endswith("Z")

    offset = GalileoObserve._record_to_v2_span(
        {"created_at": "2026-05-25T12:00:00-05:00"},
        trace_id="trace-id",
        span_id="span-id",
    )
    assert offset["created_at"] == "2026-05-25T12:00:00-05:00"


def test_galileo_get_output_str_variants(galileo_v2_env):
    logger = GalileoObserve()
    assert logger.get_output_str_from_response(None, {}) == ""
    assert (
        logger.get_output_str_from_response(
            EmbeddingResponse(), {"call_type": "embedding"}
        )
        == "embedding-output"
    )
    assert (
        logger.get_output_str_from_response(
            EmbeddingResponse(), {"call_type": "aembedding"}
        )
        == "embedding-output"
    )

    text_resp = TextCompletionResponse()
    text_resp.choices = [MagicMock(text="text-completion-output")]
    assert (
        logger.get_output_str_from_response(text_resp, {"call_type": "text_completion"})
        == "text-completion-output"
    )

    image_resp = ImageResponse(data=[ImageObject(url="https://x/y.png")])
    assert "y.png" in logger.get_output_str_from_response(image_resp, {})

    speech_resp = HttpxBinaryResponseContent(response=MagicMock())
    assert (
        logger.get_output_str_from_response(speech_resp, {"call_type": "aspeech"})
        == "speech-output"
    )

    transcription_resp = TranscriptionResponse(text="hello world")
    assert (
        logger.get_output_str_from_response(
            transcription_resp, {"call_type": "atranscription"}
        )
        == "hello world"
    )

    realtime_output = [{"type": "response", "text": "hi"}]
    assert (
        logger.get_output_str_from_response(
            realtime_output,
            {"call_type": "_arealtime", "input": {"session": "abc"}},
        )
        == '[{"type": "response", "text": "hi"}]'
    )

    pass_through_output = {"response": "passthrough-body", "status": 200}
    assert (
        logger.get_output_str_from_response(
            pass_through_output, {"call_type": "pass_through_endpoint"}
        )
        == "passthrough-body"
    )

    model_resp = ModelResponse(
        choices=[Choices(message=Message(content="chat reply", role="assistant"))]
    )
    assert '"chat reply"' in logger.get_output_str_from_response(
        model_resp,
        {"call_type": "acompletion", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert logger.get_output_str_from_response("not-a-supported-type", {}) == ""


def test_galileo_get_input_output_error_status_message(galileo_v2_env):
    logger = GalileoObserve()
    input_text, output_text, _ = logger._get_galileo_input_output_content(
        kwargs={"messages": [{"role": "user", "content": "fail me"}]},
        response_obj=None,
        level="ERROR",
        status_message="provider timeout",
    )
    assert input_text == "fail me"
    assert output_text == "provider timeout"


def test_galileo_get_output_str_rerank_response(galileo_v2_env):
    logger = GalileoObserve()
    rerank_response = RerankResponse(
        results=[
            {"index": 2, "relevance_score": 0.98},
            {"index": 0, "relevance_score": 0.12},
        ]
    )
    output = logger.get_output_str_from_response(
        rerank_response, {"call_type": "arerank"}
    )
    assert output is not None
    assert '"index": 2' in output
    assert '"relevance_score": 0.98' in output


@pytest.mark.asyncio
async def test_galileo_async_log_success_embedding(galileo_v2_env):
    import datetime

    logger = GalileoObserve()
    embedding_response = EmbeddingResponse(
        data=[{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}]
    )

    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.status_code = 201

    with patch.object(logger.async_httpx_handler, "post", return_value=mock_response):
        await logger.async_log_success_event(
            kwargs={
                "call_type": "aembedding",
                "model": "text-embedding-3-small",
                "input": "hello world",
                "standard_logging_object": {
                    "call_type": "aembedding",
                    "model": "text-embedding-3-small",
                    "prompt_tokens": 2,
                    "completion_tokens": 0,
                    "total_tokens": 2,
                    "response_cost": 0.0,
                    "startTime": datetime.datetime(
                        2026, 5, 25, 12, 0, 0, tzinfo=datetime.timezone.utc
                    ).timestamp(),
                    "endTime": datetime.datetime(
                        2026, 5, 25, 12, 0, 1, tzinfo=datetime.timezone.utc
                    ).timestamp(),
                },
            },
            response_obj=embedding_response,
            start_time=datetime.datetime(2026, 5, 25, 12, 0, 0),
            end_time=datetime.datetime(2026, 5, 25, 12, 0, 1),
        )

    assert logger.in_memory_records == []


@pytest.mark.asyncio
async def test_galileo_async_log_success_rerank(galileo_v2_env):
    import datetime

    logger = GalileoObserve()
    rerank_response = RerankResponse(results=[{"index": 1, "relevance_score": 0.95}])

    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.status_code = 201

    with patch.object(logger.async_httpx_handler, "post", return_value=mock_response):
        await logger.async_log_success_event(
            kwargs={
                "call_type": "arerank",
                "model": "cohere/rerank-english-v3.0",
                "query": "What is the capital of the United States?",
                "documents": ["doc-a", "doc-b"],
                "standard_logging_object": {
                    "call_type": "arerank",
                    "model": "cohere/rerank-english-v3.0",
                    "messages": "What is the capital of the United States?",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "response_cost": 0.0,
                    "startTime": datetime.datetime(
                        2026, 5, 25, 12, 0, 0, tzinfo=datetime.timezone.utc
                    ).timestamp(),
                    "endTime": datetime.datetime(
                        2026, 5, 25, 12, 0, 1, tzinfo=datetime.timezone.utc
                    ).timestamp(),
                },
            },
            response_obj=rerank_response,
            start_time=datetime.datetime(2026, 5, 25, 12, 0, 0),
            end_time=datetime.datetime(2026, 5, 25, 12, 0, 1),
        )

    assert logger.in_memory_records == []


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
    monkeypatch.setenv("GALILEO_LOG_STREAM_ID", "stream-id")
    logger = GalileoObserve()
    logger.in_memory_records = [
        {
            "latency_ms": 1,
            "status_code": 200,
            "input_text": "hi",
            "output_text": "ok",
            "node_type": "acompletion",
            "model": "gpt",
            "num_input_tokens": 1,
            "num_output_tokens": 1,
            "num_total_tokens": 2,
            "created_at": "2026-05-25T12:00:00",
        }
    ]
    url, payload = logger._get_ingest_request()
    assert url == "https://galileo.example/v2/projects/proj/traces"
    assert "traces" in payload
    assert payload["log_stream_id"] == "stream-id"
    assert payload["traces"][0]["input"] == "hi"


@pytest.mark.asyncio
async def test_galileo_async_health_check_success(galileo_v2_env):
    logger = GalileoObserve()
    current_user_resp = MagicMock()
    current_user_resp.status_code = 200

    with patch.object(
        logger.async_httpx_handler, "get", new_callable=AsyncMock
    ) as mock_get:
        mock_get.return_value = current_user_resp
        result = await logger.async_health_check()

    assert result["status"] == "healthy"
    mock_get.assert_awaited_once_with(
        url="https://api.galileo.ai/current_user",
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Galileo-API-Key": "test-api-key",
        },
    )


@pytest.mark.asyncio
async def test_galileo_async_health_check_api_error(galileo_v2_env):
    logger = GalileoObserve()
    current_user_resp = MagicMock()
    current_user_resp.status_code = 401

    with patch.object(
        logger.async_httpx_handler, "get", new_callable=AsyncMock
    ) as mock_get:
        mock_get.return_value = current_user_resp
        result = await logger.async_health_check()

    assert result["status"] == "unhealthy"
    assert "HTTP 401" in result["error_message"]


@pytest.mark.asyncio
async def test_galileo_async_health_check_missing_project_id(monkeypatch):
    monkeypatch.setenv("GALILEO_API_KEY", "test-api-key")
    monkeypatch.setenv("GALILEO_BASE_URL", "https://api.galileo.ai")
    monkeypatch.delenv("GALILEO_PROJECT_ID", raising=False)
    logger = GalileoObserve()

    result = await logger.async_health_check()

    assert result["status"] == "unhealthy"
    assert "GALILEO_PROJECT_ID" in result["error_message"]


@pytest.mark.asyncio
async def test_galileo_async_health_check_missing_base_url(monkeypatch):
    monkeypatch.delenv("GALILEO_API_KEY", raising=False)
    monkeypatch.delenv("GALILEO_BASE_URL", raising=False)
    monkeypatch.setenv("GALILEO_PROJECT_ID", "p")
    monkeypatch.setenv("GALILEO_USERNAME", "u")
    monkeypatch.setenv("GALILEO_PASSWORD", "pw")
    logger = GalileoObserve()

    result = await logger.async_health_check()

    assert result["status"] == "unhealthy"
    assert "GALILEO_BASE_URL" in result["error_message"]


@pytest.mark.asyncio
async def test_galileo_async_health_check_missing_credentials(monkeypatch):
    monkeypatch.delenv("GALILEO_API_KEY", raising=False)
    monkeypatch.delenv("GALILEO_USERNAME", raising=False)
    monkeypatch.delenv("GALILEO_PASSWORD", raising=False)
    monkeypatch.setenv("GALILEO_PROJECT_ID", "p")
    monkeypatch.setenv("GALILEO_BASE_URL", "https://galileo.example")
    logger = GalileoObserve()

    result = await logger.async_health_check()

    assert result["status"] == "unhealthy"
    assert "GALILEO_USERNAME" in result["error_message"]


@pytest.mark.asyncio
async def test_galileo_async_health_check_auth_failed(monkeypatch):
    monkeypatch.delenv("GALILEO_API_KEY", raising=False)
    monkeypatch.setenv("GALILEO_PROJECT_ID", "p")
    monkeypatch.setenv("GALILEO_BASE_URL", "https://galileo.example")
    monkeypatch.setenv("GALILEO_USERNAME", "u")
    monkeypatch.setenv("GALILEO_PASSWORD", "pw")
    logger = GalileoObserve()

    with patch.object(
        logger.async_httpx_handler, "post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.side_effect = Exception("login failed")
        result = await logger.async_health_check()

    assert result["status"] == "unhealthy"
    assert result["error_message"] == "Galileo authentication failed"


@pytest.mark.asyncio
async def test_galileo_async_health_check_request_exception(galileo_v2_env):
    logger = GalileoObserve()

    with patch.object(
        logger.async_httpx_handler, "get", new_callable=AsyncMock
    ) as mock_get:
        mock_get.side_effect = Exception("connection refused")
        result = await logger.async_health_check()

    assert result["status"] == "unhealthy"
    assert "connection refused" in result["error_message"]


@pytest.mark.asyncio
async def test_galileo_async_log_success_empty_model_response(galileo_v2_env):
    import datetime

    logger = GalileoObserve()
    logger.batch_size = 2
    empty_response = ModelResponse(choices=[])

    await logger.async_log_success_event(
        kwargs={
            "call_type": "acompletion",
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "hi"}],
            "standard_logging_object": {
                "call_type": "acompletion",
                "model": "gpt-5.2",
                "prompt_tokens": 1,
                "completion_tokens": 0,
                "total_tokens": 1,
                "response_cost": 0.0,
                "startTime": datetime.datetime(
                    2026, 5, 25, 12, 0, 0, tzinfo=datetime.timezone.utc
                ).timestamp(),
                "endTime": datetime.datetime(
                    2026, 5, 25, 12, 0, 1, tzinfo=datetime.timezone.utc
                ).timestamp(),
            },
        },
        response_obj=empty_response,
        start_time=datetime.datetime(2026, 5, 25, 12, 0, 0),
        end_time=datetime.datetime(2026, 5, 25, 12, 0, 1),
    )

    assert len(logger.in_memory_records) == 1
    assert logger.in_memory_records[0]["output_text"] == ""


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
async def test_galileo_async_log_success_preserves_passthrough_messages(
    galileo_v2_env,
):
    import datetime

    logger = GalileoObserve()
    logger.batch_size = 2
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
    ]

    await logger.async_log_success_event(
        kwargs={
            "call_type": "pass_through_endpoint",
            "model": "gpt",
            "messages": messages,
            "standard_logging_object": {
                "call_type": "pass_through_endpoint",
                "model": "gpt",
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 0,
                "response_cost": 0.001,
                "startTime": datetime.datetime(
                    2026, 5, 25, 12, 0, 0, tzinfo=datetime.timezone.utc
                ).timestamp(),
                "endTime": datetime.datetime(
                    2026, 5, 25, 12, 0, 1, tzinfo=datetime.timezone.utc
                ).timestamp(),
            },
        },
        response_obj={"response": "ok"},
        start_time=datetime.datetime(2026, 5, 25, 12, 0, 0),
        end_time=datetime.datetime(2026, 5, 25, 12, 0, 1),
    )

    assert logger.in_memory_records[0]["messages"] == messages
    assert logger.in_memory_records[0]["num_total_tokens"] == 3


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
                "standard_logging_object": {
                    "call_type": "acompletion",
                    "model": "gpt",
                    "messages": [{"role": "user", "content": "hi"}],
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                    "response_cost": 0.001,
                    "startTime": datetime.datetime(
                        2026, 5, 25, 12, 0, 0, tzinfo=datetime.timezone.utc
                    ).timestamp(),
                    "endTime": datetime.datetime(
                        2026, 5, 25, 12, 0, 1, tzinfo=datetime.timezone.utc
                    ).timestamp(),
                },
            },
            response_obj=response,
            start_time=datetime.datetime(2026, 5, 25, 12, 0, 0),
            end_time=datetime.datetime(2026, 5, 25, 12, 0, 1),
        )

    assert "/ingest/traces/" in flushed_url["url"]
    assert logger.in_memory_records == []
