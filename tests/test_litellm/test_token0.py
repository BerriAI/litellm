"""Tests for the Token0 LiteLLM CustomLogger integration.

These tests verify the Token0Hook contract without making real API calls.
Token0 is installed separately: pip install token0
"""

import pytest
from unittest.mock import patch


def _make_image_message(url: str = "data:image/jpeg;base64,/9j/fake") -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }


# ---------------------------------------------------------------------------
# Import guard — skip entire module if token0 is not installed
# ---------------------------------------------------------------------------

token0 = pytest.importorskip("token0", reason="token0 not installed")


# ---------------------------------------------------------------------------
# Hook contract tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token0_hook_passthrough_for_non_completion():
    """Hook must return data unchanged when call_type != 'completion'."""
    from token0.litellm_hook import Token0Hook

    hook = Token0Hook()
    data = {"messages": [_make_image_message()], "model": "gpt-4o"}
    result = await hook.async_pre_call_hook(
        user_api_key_dict={}, cache=None, data=data, call_type="embedding"
    )
    assert result is data


@pytest.mark.asyncio
async def test_token0_hook_passthrough_for_empty_messages():
    """Hook must return data unchanged when messages is empty."""
    from token0.litellm_hook import Token0Hook

    hook = Token0Hook()
    data = {"messages": [], "model": "gpt-4o"}
    result = await hook.async_pre_call_hook(
        user_api_key_dict={}, cache=None, data=data, call_type="completion"
    )
    assert result is data


@pytest.mark.asyncio
async def test_token0_hook_text_only_passthrough():
    """Text-only messages must pass through with zero overhead."""
    from token0.litellm_hook import Token0Hook

    hook = Token0Hook()
    original_messages = [{"role": "user", "content": "Hello, what is 2+2?"}]
    data = {"messages": original_messages, "model": "gpt-4o"}

    result = await hook.async_pre_call_hook(
        user_api_key_dict={}, cache=None, data=data, call_type="completion"
    )

    assert result["messages"] == original_messages


@pytest.mark.asyncio
async def test_token0_hook_attaches_stats_metadata():
    """Hook must attach token0 stats to data['metadata']['token0']."""
    from token0.litellm_hook import Token0Hook

    hook = Token0Hook()
    messages = [_make_image_message()]
    data = {"messages": messages, "model": "gpt-4o"}

    mock_stats = {
        "tokens_before": 765,
        "tokens_after": 85,
        "tokens_saved": 680,
        "optimizations": ["prompt-aware→low detail"],
        "recommended_model": None,
    }

    with patch(
        "token0.litellm_hook.optimize_messages",
        return_value=(messages, mock_stats),
    ):
        result = await hook.async_pre_call_hook(
            user_api_key_dict={}, cache=None, data=data, call_type="completion"
        )

    assert "metadata" in result
    assert "token0" in result["metadata"]
    assert result["metadata"]["token0"]["tokens_saved"] == 680


@pytest.mark.asyncio
async def test_token0_hook_remote_url_passthrough():
    """Images with remote http/https URLs must not be modified."""
    from token0.litellm_hook import Token0Hook

    hook = Token0Hook()
    remote_url = "https://example.com/photo.jpg"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": remote_url}},
            ],
        }
    ]
    data = {"messages": messages, "model": "gpt-4o"}

    mock_stats = {
        "tokens_before": 0,
        "tokens_after": 0,
        "tokens_saved": 0,
        "optimizations": [],
        "recommended_model": None,
    }

    with patch(
        "token0.litellm_hook.optimize_messages",
        return_value=(messages, mock_stats),
    ):
        result = await hook.async_pre_call_hook(
            user_api_key_dict={}, cache=None, data=data, call_type="completion"
        )

    content = result["messages"][0]["content"]
    image_parts = [p for p in content if p.get("type") == "image_url"]
    assert image_parts[0]["image_url"]["url"] == remote_url
