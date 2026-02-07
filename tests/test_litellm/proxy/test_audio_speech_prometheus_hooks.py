"""
Regression tests: proxy /v1/audio/speech (TTS) must call proxy-level success/failure
hooks so Prometheus metrics (litellm_proxy_total_requests_metric, litellm_proxy_failed_requests_metric)
and other callbacks see TTS requests.
"""
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Import after path setup so proxy_server is loadable
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app, initialize


def _mock_user_api_key_auth():
    """Bypass auth for tests so /v1/audio/speech doesn't require a real key."""
    return MagicMock()


def _make_mock_tts_response():
    """Mock response for handler: llm_call = await route_request(), response = await llm_call, then _audio_speech_chunk_generator does await response.aiter_bytes() and async for chunk in it."""

    async def _chunks():
        yield b"\xff\xfb"

    def _aiter_bytes(chunk_size=8192):
        async def _wrapper():
            return _chunks()

        return _wrapper()

    inner = MagicMock()
    inner.aiter_bytes = _aiter_bytes
    inner._hidden_params = {}

    async def _resolver():
        return inner

    return _resolver()


@pytest.fixture
def client_no_auth():
    from litellm.proxy.proxy_server import cleanup_router_config_variables

    cleanup_router_config_variables()
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_fp = os.path.join(filepath, "test_configs", "test_config_no_auth.yaml")
    asyncio.run(initialize(config=config_fp, debug=True))
    return TestClient(app)


@pytest.mark.asyncio
@pytest.mark.retry(retries=0)
async def test_audio_speech_success_calls_post_call_success_hook(client_no_auth):
    """TTS success path must call proxy_logging_obj.post_call_success_hook (Prometheus total requests)."""
    mock_success_hook = AsyncMock()
    mock_failure_hook = AsyncMock()
    mock_pre_call = AsyncMock(side_effect=lambda *, data, **kw: data)
    mock_update_status = AsyncMock()

    mock_logging = MagicMock()
    mock_logging.post_call_success_hook = mock_success_hook
    mock_logging.post_call_failure_hook = mock_failure_hook
    mock_logging.pre_call_hook = mock_pre_call
    mock_logging.update_request_status = mock_update_status

    async def _mock_route_request(*, data, route_type, llm_router, user_model):
        assert route_type == "aspeech"
        return _make_mock_tts_response()

    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[user_api_key_auth] = _mock_user_api_key_auth
    try:
        with (
            patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_logging),
            patch(
                "litellm.proxy.proxy_server.route_request",
                side_effect=_mock_route_request,
            ),
        ):
            response = client_no_auth.post(
                "/v1/audio/speech",
                json={"model": "tts-1", "input": "hello"},
                headers={"Content-Type": "application/json"},
            )
    finally:
        app.dependency_overrides = original_overrides

    assert response.status_code == 200
    mock_success_hook.assert_awaited_once()
    mock_failure_hook.assert_not_called()
    # Ensure we passed through the right call type
    call_kw = mock_success_hook.call_args.kwargs
    assert "data" in call_kw and "user_api_key_dict" in call_kw


@pytest.mark.asyncio
@pytest.mark.retry(retries=0)
async def test_audio_speech_failure_calls_post_call_failure_hook(client_no_auth):
    """TTS failure path must call proxy_logging_obj.post_call_failure_hook (Prometheus failed requests)."""
    mock_success_hook = AsyncMock()
    mock_failure_hook = AsyncMock()
    mock_pre_call = AsyncMock(side_effect=lambda *, data, **kw: data)

    mock_logging = MagicMock()
    mock_logging.post_call_success_hook = mock_success_hook
    mock_logging.post_call_failure_hook = mock_failure_hook
    mock_logging.pre_call_hook = mock_pre_call

    async def _mock_route_request_raise(*, data, route_type, llm_router, user_model):
        raise ValueError("mock rate limit")

    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[user_api_key_auth] = _mock_user_api_key_auth
    # Don't re-raise server exceptions so we get the 500 response instead of ValueError
    client = TestClient(app, raise_server_exceptions=False)
    try:
        with (
            patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_logging),
            patch(
                "litellm.proxy.proxy_server.route_request",
                side_effect=_mock_route_request_raise,
            ),
        ):
            response = client.post(
                "/v1/audio/speech",
                json={"model": "tts-1", "input": "hello"},
                headers={"Content-Type": "application/json"},
            )
    finally:
        app.dependency_overrides = original_overrides

    assert response.status_code == 500
    mock_failure_hook.assert_awaited_once()
    mock_success_hook.assert_not_called()
    call_kw = mock_failure_hook.call_args.kwargs
    assert "user_api_key_dict" in call_kw and "original_exception" in call_kw
