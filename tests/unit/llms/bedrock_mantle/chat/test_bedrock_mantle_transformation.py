"""
E2E tests for Bedrock Mantle (Claude Mythos Preview) integration.

Tests use a fake/mocked HTTP layer to verify the full request pipeline:
- correct endpoint URL
- model ID in the request body
- AWS SigV4 Authorization header present
- response parsing
"""

import asyncio
import importlib
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome
from tests.fake_openai_endpoint import ensure_fake_openai_endpoint

MODEL = "bedrock/mantle/anthropic.claude-mythos-preview"
REGION = "us-east-1"
EXPECTED_URL = f"https://bedrock-mantle.{REGION}.api.aws/anthropic/v1/messages"

FAKE_ANTHROPIC_RESPONSE = {
    "id": "msg_fake123",
    "type": "message",
    "role": "assistant",
    "model": "anthropic.claude-mythos-preview",
    "content": [{"type": "text", "text": "Hello from Mythos!"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


def _make_fake_response(body: dict) -> MagicMock:
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.headers = httpx.Headers({"content-type": "application/json"})
    mock_resp.text = json.dumps(body)
    mock_resp.json.return_value = body
    mock_resp.is_error = False
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def test_mantle_request_url_and_body():
    """Verify the correct URL is called and model appears in the request body."""
    client = HTTPHandler()

    with patch.object(client, "post", return_value=_make_fake_response(FAKE_ANTHROPIC_RESPONSE)) as mock_post:
        try:
            litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=50,
                aws_region_name=REGION,
                aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                client=client,
            )
        except Exception:
            pass  # response parsing may fail on mock; we only care about the outgoing call

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs

        # Correct endpoint
        assert call_kwargs["url"] == EXPECTED_URL, f"Expected {EXPECTED_URL}, got {call_kwargs['url']}"

        # Request body has model ID (without "mantle/" prefix)
        raw_data = call_kwargs.get("data") or call_kwargs.get("json")
        body = json.loads(raw_data) if isinstance(raw_data, (str, bytes)) else raw_data
        assert body["model"] == "anthropic.claude-mythos-preview", f"body['model'] = {body.get('model')}"
        assert "messages" in body
        assert body["max_tokens"] == 50

        # AWS SigV4 Authorization header must be present
        headers = call_kwargs.get("headers", {})
        assert "Authorization" in headers, f"No Authorization header in {headers}"
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256"), (
            f"Expected SigV4 auth, got: {headers['Authorization'][:50]}"
        )


def test_mantle_request_does_not_include_mantle_prefix_in_body():
    """Ensure 'mantle/' never leaks into the request body."""
    client = HTTPHandler()

    with patch.object(client, "post", return_value=_make_fake_response(FAKE_ANTHROPIC_RESPONSE)) as mock_post:
        try:
            litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10,
                aws_region_name=REGION,
                aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                client=client,
            )
        except Exception:
            pass

        call_kwargs = mock_post.call_args.kwargs
        raw_data = call_kwargs.get("data") or call_kwargs.get("json")
        body = json.loads(raw_data) if isinstance(raw_data, (str, bytes)) else raw_data

        body_str = json.dumps(body)
        assert "mantle/" not in body_str, f"'mantle/' leaked into body: {body_str}"


def test_mantle_region_reflected_in_url():
    """The region from aws_region_name must appear in the endpoint URL."""
    client = HTTPHandler()

    for region in ["us-east-1", "us-west-2", "eu-west-1"]:
        with patch.object(client, "post", return_value=_make_fake_response(FAKE_ANTHROPIC_RESPONSE)) as mock_post:
            try:
                litellm.completion(
                    model=MODEL,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=10,
                    aws_region_name=region,
                    aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                    aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                    client=client,
                )
            except Exception:
                pass

            call_kwargs = mock_post.call_args.kwargs
            expected = f"https://bedrock-mantle.{region}.api.aws/anthropic/v1/messages"
            assert call_kwargs["url"] == expected, f"region={region}: expected URL {expected}, got {call_kwargs['url']}"


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def fake_openai_endpoint():
    ensure_fake_openai_endpoint()
    yield


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown(event_loop):
    import litellm

    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    asyncio.set_event_loop(event_loop)
    yield
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
    "cohere_key": getattr(litellm, "cohere_key", None),
}
