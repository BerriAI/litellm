"""
Test DynamoAI Guardrails integration
"""

import importlib
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.dynamoai import DynamoAIGuardrails
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


@pytest.mark.asyncio
async def test_dynamoai_blocks_content_with_block_action():
    """
    Test that DynamoAI guardrail blocks content when finalAction is BLOCK.
    """
    # Create guardrail instance
    guardrail = DynamoAIGuardrails(
        guardrail_name="test-dynamoai",
        api_key="test-api-key",
        api_base="https://api.dynamo.ai",
    )

    # Mock the DynamoAI API response with BLOCK action
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "This is harmful content",
        "textType": "MODEL_INPUT",
        "finalAction": "BLOCK",
        "appliedPolicies": [
            {
                "policy": {
                    "id": "policy-123",
                    "name": "Toxicity Policy",
                    "description": "Blocks toxic content",
                    "method": "TOXICITY",
                },
                "outputs": {
                    "action": "BLOCK",
                    "message": "Content contains toxic language",
                },
            }
        ],
    }
    mock_response.raise_for_status = MagicMock()
    with patch.object(guardrail.async_handler, "post", AsyncMock(return_value=mock_response)):
        request_data = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "This is harmful content"}],
        }

        # Mock should_run_guardrail to return True
        guardrail.should_run_guardrail = MagicMock(return_value=True)

        # Test that the guardrail raises ValueError for blocked content
        with pytest.raises(ValueError, match="violation\\(s\\) detected") as exc_info:
            await guardrail.async_pre_call_hook(
                data=request_data,
                user_api_key_dict=UserAPIKeyAuth(),
                call_type="completion",
                cache=MagicMock(spec=DualCache),
            )

    # Verify the error message contains policy information
    error_message = str(exc_info.value)
    assert "Guardrail failed" in error_message
    assert "TOXICITY POLICY" in error_message.upper()
    assert "BLOCK" in error_message.upper()


@pytest.mark.asyncio
async def test_dynamoai_allows_content_with_none_action():
    """
    Test that DynamoAI guardrail allows content when finalAction is NONE.
    """
    # Create guardrail instance
    guardrail = DynamoAIGuardrails(
        guardrail_name="test-dynamoai",
        api_key="test-api-key",
        api_base="https://api.dynamo.ai",
    )

    # Mock the DynamoAI API response with NONE action (no violations)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Hello, how are you?",
        "textType": "MODEL_INPUT",
        "finalAction": "NONE",
        "appliedPolicies": [],
    }
    mock_response.raise_for_status = MagicMock()
    with patch.object(guardrail.async_handler, "post", AsyncMock(return_value=mock_response)):
        request_data = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
        }

        # Mock should_run_guardrail to return True
        guardrail.should_run_guardrail = MagicMock(return_value=True)

        # Test that the guardrail allows the content (no exception raised)
        result = await guardrail.async_pre_call_hook(
            data=request_data,
            user_api_key_dict=UserAPIKeyAuth(),
            call_type="completion",
            cache=MagicMock(spec=DualCache),
        )

    # Should return the request data unchanged
    assert result == request_data


_ANALYZE_URL = "https://api.dynamo.ai/v1/moderation/analyze/"
_BLOCK_VERDICT = {
    "finalAction": "BLOCK",
    "appliedPolicies": [
        {
            "policy": {"id": "policy-123", "name": "Toxicity Policy", "method": "TOXICITY"},
            "outputs": {"action": "BLOCK", "message": "Content contains toxic language"},
        }
    ],
}


def _post_call_guardrail() -> DynamoAIGuardrails:
    return DynamoAIGuardrails(
        guardrail_name="test-dynamoai",
        api_key="test-api-key",
        api_base="https://api.dynamo.ai",
        event_hook="post_call",
        default_on=True,
    )


def _assistant_reply(content: str | None) -> litellm.ModelResponse:
    return litellm.ModelResponse(
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}]
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("httpx_transport")
async def test_post_call_hook_sends_the_assistant_text_to_dynamoai_and_allows_a_clean_reply(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(_ANALYZE_URL).respond(json={"finalAction": "NONE", "appliedPolicies": []})
    request_data = {"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]}

    result = await _post_call_guardrail().async_post_call_success_hook(
        data=request_data, user_api_key_dict=UserAPIKeyAuth(), response=_assistant_reply("Hello, how are you?")
    )

    assert result is None
    sent = route.calls.last.request
    assert json.loads(sent.content) == {"messages": [{"role": "assistant", "content": "Hello, how are you?"}]}
    assert sent.headers["Authorization"] == "Bearer test-api-key"


@pytest.mark.asyncio
@pytest.mark.usefixtures("httpx_transport")
async def test_post_call_hook_raises_when_dynamoai_blocks_the_assistant_text(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(_ANALYZE_URL).respond(json=_BLOCK_VERDICT)

    with pytest.raises(ValueError, match="Guardrail failed") as blocked:
        await _post_call_guardrail().async_post_call_success_hook(
            data={"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
            user_api_key_dict=UserAPIKeyAuth(),
            response=_assistant_reply("This is harmful content"),
        )

    assert str(blocked.value) == (
        "Guardrail failed: 1 violation(s) detected\n\n"
        "- TOXICITY POLICY:\n"
        "  Action: BLOCK\n"
        "  Method: TOXICITY\n"
        "  Message: Content contains toxic language\n"
        "  Policy ID: policy-123"
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("httpx_transport")
async def test_post_call_hook_skips_dynamoai_when_the_reply_has_no_text(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(_ANALYZE_URL).respond(json=_BLOCK_VERDICT)

    result = await _post_call_guardrail().async_post_call_success_hook(
        data={"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
        user_api_key_dict=UserAPIKeyAuth(),
        response=_assistant_reply(None),
    )

    assert result is None
    assert route.called is False


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="function", autouse=True)
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Saves and restores litellm callback/global state so tests don't leak
    side effects. Works safely under pytest-xdist parallel execution.
    """
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
    for attr in ("set_verbose", "cache", "num_retries"):
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr in ("success_callback", "failure_callback", "_async_success_callback", "_async_failure_callback"):
        if hasattr(litellm, attr):
            setattr(litellm, attr, [])
    yield
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception:
            pass
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield
