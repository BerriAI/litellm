import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.integrations.akto.akto_logger import AktoLogger


# ── Fixtures ──


@pytest.fixture
def akto_env():
    with patch.dict(
        os.environ,
        {
            "AKTO_DATA_INGESTION_API_BASE": "http://localhost:9090",
            "AKTO_API_KEY": "test-token",
        },
    ):
        yield


@pytest.fixture
def logger(akto_env):
    return AktoLogger()


@pytest.fixture
def sample_kwargs():
    return {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-4",
        "litellm_params": {
            "metadata": {
                "user_api_key_request_route": "/v1/chat/completions",
                "user_api_key_user_id": "user-1",
                "user_api_key_team_id": "team-1",
            },
            "proxy_server_request": {
                "headers": {
                    "host": "my-litellm.example.com",
                    "content-type": "application/json",
                    "x-forwarded-for": "10.0.0.1",
                },
            },
        },
    }


# ── Init ──


def test_init_requires_env_vars():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("AKTO_DATA_INGESTION_API_BASE", None)
        os.environ.pop("AKTO_API_KEY", None)
        with pytest.raises(Exception, match="AKTO_DATA_INGESTION_API_BASE"):
            AktoLogger()


def test_init_requires_api_key():
    with patch.dict(
        os.environ, {"AKTO_DATA_INGESTION_API_BASE": "http://x"}, clear=False
    ):
        os.environ.pop("AKTO_API_KEY", None)
        with pytest.raises(Exception, match="AKTO_API_KEY"):
            AktoLogger()


def test_init_success(logger):
    assert logger.akto_base_url == "http://localhost:9090"
    assert logger.akto_api_key == "test-token"
    assert logger.akto_account_id == "1000000"
    assert logger.akto_vxlan_id == "0"


# ── extract_logging_data ──


def test_extract_logging_data_promotes_from_litellm_params():
    kwargs = {
        "messages": [{"role": "user", "content": "hi"}],
        "litellm_params": {
            "metadata": {"user_api_key_request_route": "/v1/chat/completions"},
            "proxy_server_request": {"headers": {"host": "example.com"}},
        },
    }
    data = AktoLogger.extract_logging_data(kwargs)
    assert data["metadata"]["user_api_key_request_route"] == "/v1/chat/completions"
    assert data["proxy_server_request"]["headers"]["host"] == "example.com"


def test_extract_logging_data_does_not_overwrite():
    kwargs = {
        "metadata": {"user_api_key_request_route": "/chat/completions"},
        "litellm_params": {
            "metadata": {"user_api_key_request_route": "/v1/chat/completions"},
        },
    }
    data = AktoLogger.extract_logging_data(kwargs)
    assert data["metadata"]["user_api_key_request_route"] == "/chat/completions"


# ── Payload ──


def test_build_akto_payload(logger, sample_kwargs):
    data = AktoLogger.extract_logging_data(sample_kwargs)
    payload = logger.build_akto_payload(data)

    assert payload["path"] == "/v1/chat/completions"
    assert payload["method"] == "POST"
    assert payload["statusCode"] == "200"
    assert payload["source"] == "MIRRORING"
    assert payload["contextSource"] == "AGENTIC"
    assert payload["ip"] == "10.0.0.1"
    assert payload["akto_account_id"] == "1000000"

    req_body = json.loads(payload["requestPayload"])
    assert req_body["model"] == "gpt-4"
    assert req_body["messages"][0]["content"] == "Hello"

    req_headers = json.loads(payload["requestHeaders"])
    assert req_headers["host"] == "my-litellm.example.com"

    tag = json.loads(payload["tag"])
    assert tag == {"gen-ai": "Gen AI", "user_id": "user-1", "team_id": "team-1"}


def test_build_akto_payload_failure_status(logger):
    payload = logger.build_akto_payload({}, status_code=500)
    assert payload["statusCode"] == "500"
    assert payload["status"] == "500"


# ── Sensitive headers ──


def test_build_request_headers_strips_sensitive():
    headers = AktoLogger.build_request_headers(
        {
            "proxy_server_request": {
                "headers": {
                    "host": "myhost.com",
                    "Authorization": "Bearer sk-secret",
                    "X-Api-Key": "key-123",
                    "Cookie": "session=abc",
                    "user-agent": "test",
                }
            }
        }
    )
    assert "authorization" not in headers
    assert "x-api-key" not in headers
    assert "cookie" not in headers
    assert headers["host"] == "myhost.com"
    assert headers["user-agent"] == "test"


# ── get_failure_status_code ──


def test_get_failure_status_code_with_status():
    exc = MagicMock()
    exc.status_code = 403
    assert AktoLogger.get_failure_status_code({"exception": exc}) == 403


def test_get_failure_status_code_no_status():
    assert AktoLogger.get_failure_status_code({"exception": Exception("err")}) == 500


def test_get_failure_status_code_no_exception():
    assert AktoLogger.get_failure_status_code({}) == 500


# ── Async success ──


@pytest.mark.asyncio
async def test_async_log_success_event(logger, sample_kwargs):
    logger.async_http_handler.post = AsyncMock(return_value=MagicMock(status_code=200))
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "choices": [{"message": {"content": "Hi!", "role": "assistant"}}]
    }

    await logger.async_log_success_event(
        kwargs=sample_kwargs, response_obj=mock_resp, start_time=None, end_time=None
    )

    logger.async_http_handler.post.assert_called_once()
    call = logger.async_http_handler.post.call_args.kwargs
    assert call["params"]["ingest_data"] == "true"
    payload = call["json"]
    assert payload["statusCode"] == "200"
    assert (
        json.loads(payload["responsePayload"])["choices"][0]["message"]["content"]
        == "Hi!"
    )


# ── Async failure ──


@pytest.mark.asyncio
async def test_async_log_failure_event(logger, sample_kwargs):
    logger.async_http_handler.post = AsyncMock(return_value=MagicMock(status_code=200))

    await logger.async_log_failure_event(
        kwargs=sample_kwargs, response_obj=None, start_time=None, end_time=None
    )

    logger.async_http_handler.post.assert_called_once()
    payload = logger.async_http_handler.post.call_args.kwargs["json"]
    assert payload["statusCode"] == "500"


# ── Async failure with response_obj ──


@pytest.mark.asyncio
async def test_async_log_failure_with_response_obj(logger, sample_kwargs):
    logger.async_http_handler.post = AsyncMock(return_value=MagicMock(status_code=200))
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {"partial": "data"}

    await logger.async_log_failure_event(
        kwargs=sample_kwargs, response_obj=mock_resp, start_time=None, end_time=None
    )

    payload = logger.async_http_handler.post.call_args.kwargs["json"]
    assert payload["statusCode"] == "500"
    assert json.loads(payload["responsePayload"])["partial"] == "data"


@pytest.mark.asyncio
async def test_async_log_failure_with_status_code(logger, sample_kwargs):
    logger.async_http_handler.post = AsyncMock(return_value=MagicMock(status_code=200))
    exc = MagicMock()
    exc.status_code = 403
    sample_kwargs["exception"] = exc

    await logger.async_log_failure_event(
        kwargs=sample_kwargs, response_obj=None, start_time=None, end_time=None
    )

    payload = logger.async_http_handler.post.call_args.kwargs["json"]
    assert payload["statusCode"] == "403"


# ── Client IP extraction ──


def test_extract_client_ip_forwarded():
    assert (
        AktoLogger.extract_client_ip(
            {
                "proxy_server_request": {
                    "headers": {"x-forwarded-for": "1.2.3.4, 5.6.7.8"}
                }
            }
        )
        == "1.2.3.4"
    )


def test_extract_client_ip_real_ip():
    assert (
        AktoLogger.extract_client_ip(
            {"proxy_server_request": {"headers": {"x-real-ip": "10.0.0.1"}}}
        )
        == "10.0.0.1"
    )


def test_extract_client_ip_fallback():
    assert AktoLogger.extract_client_ip({}) == "0.0.0.0"


# ── Request path extraction ──


def test_extract_request_path_from_metadata():
    assert (
        AktoLogger.extract_request_path(
            {"metadata": {"user_api_key_request_route": "/v1/embeddings"}}
        )
        == "/v1/embeddings"
    )


def test_extract_request_path_fallback():
    assert AktoLogger.extract_request_path({}) == "/v1/chat/completions"


# ── Default host fallback ──


def test_build_request_headers_default_host():
    headers = AktoLogger.build_request_headers({})
    assert headers["host"] == "litellm.ai"
    assert headers["content-type"] == "application/json"


# ── Error handling ──


@pytest.mark.asyncio
async def test_async_log_success_swallows_errors(logger):
    logger.async_http_handler.post = AsyncMock(
        side_effect=httpx.ConnectError("refused")
    )
    # Should not raise
    await logger.async_log_success_event(
        kwargs={"messages": [], "model": "m", "litellm_params": {}},
        response_obj=None,
        start_time=None,
        end_time=None,
    )


@pytest.mark.asyncio
async def test_async_log_failure_swallows_errors(logger):
    logger.async_http_handler.post = AsyncMock(
        side_effect=httpx.ConnectError("refused")
    )
    # Should not raise
    await logger.async_log_failure_event(
        kwargs={"messages": [], "model": "m", "litellm_params": {}},
        response_obj=None,
        start_time=None,
        end_time=None,
    )


# ── Health check ──


@pytest.mark.asyncio
async def test_async_health_check_healthy(logger):
    logger.async_http_handler.get = AsyncMock(return_value=MagicMock(status_code=200))
    result = await logger.async_health_check()
    assert result["status"] == "healthy"
    assert result["error_message"] is None


@pytest.mark.asyncio
async def test_async_health_check_unhealthy(logger):
    logger.async_http_handler.get = AsyncMock(return_value=MagicMock(status_code=503))
    result = await logger.async_health_check()
    assert result["status"] == "unhealthy"
    assert "503" in result["error_message"]


@pytest.mark.asyncio
async def test_async_health_check_exception(logger):
    logger.async_http_handler.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
    result = await logger.async_health_check()
    assert result["status"] == "unhealthy"
