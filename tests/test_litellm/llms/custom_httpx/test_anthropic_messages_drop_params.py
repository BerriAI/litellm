"""
Regression tests for issue #25931.

`async_anthropic_messages_handler` (the /v1/messages Bedrock/Anthropic path)
must honor `additional_drop_params` for both top-level and nested field names.

Before the fix, only nested paths (containing `.` or `[`) were dropped, so
top-level fields like `context_management` — injected automatically by
Claude Code v2.1+ via the `context-management-2025-06-27` Anthropic beta —
leaked through to Bedrock and produced:

    {"message": "context_management: Extra inputs are not permitted"}
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler


class _StopAfterDrop(Exception):
    """Sentinel raised by the mocked transform to halt the handler after drop."""


def _build_provider_config_mock(captured: dict) -> MagicMock:
    provider_config = MagicMock()
    provider_config.validate_anthropic_messages_environment.return_value = (
        {},
        "http://test-base",
    )

    def _capture_transform(**kwargs):
        captured["optional_params"] = kwargs[
            "anthropic_messages_optional_request_params"
        ]
        raise _StopAfterDrop

    provider_config.transform_anthropic_messages_request.side_effect = (
        _capture_transform
    )
    return provider_config


def _build_logging_obj_mock() -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    return logging_obj


@pytest.mark.asyncio
async def test_top_level_additional_drop_params_applied_on_messages_path():
    """Top-level keys (e.g. `context_management`) must be dropped from
    the params forwarded to the provider transform.

    Regression for https://github.com/BerriAI/litellm/issues/25931."""
    handler = BaseLLMHTTPHandler()
    captured: dict = {}

    params = {
        "context_management": {
            "edits": [{"type": "clear_thinking_20251015"}],
        },
        "max_tokens": 1024,
    }
    litellm_params = {"additional_drop_params": ["context_management"]}

    with pytest.raises(_StopAfterDrop):
        await handler.async_anthropic_messages_handler(
            model="us.anthropic.claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            anthropic_messages_provider_config=_build_provider_config_mock(captured),
            anthropic_messages_optional_request_params=params,
            custom_llm_provider="bedrock",
            litellm_params=litellm_params,
            logging_obj=_build_logging_obj_mock(),
        )

    forwarded = captured["optional_params"]
    assert "context_management" not in forwarded, (
        "top-level `context_management` should be stripped from params before "
        "transform_anthropic_messages_request — otherwise Bedrock returns 400"
    )
    assert forwarded["max_tokens"] == 1024, "unrelated fields must be preserved"


@pytest.mark.asyncio
async def test_nested_additional_drop_params_still_work():
    """Nested JSONPath entries must continue to work alongside top-level drops."""
    handler = BaseLLMHTTPHandler()
    captured: dict = {}

    params = {
        "tools": [
            {"name": "t0", "input_examples": ["ex0"]},
            {"name": "t1", "input_examples": ["ex1"]},
        ],
        "max_tokens": 1024,
    }
    litellm_params = {"additional_drop_params": ["tools[*].input_examples"]}

    with pytest.raises(_StopAfterDrop):
        await handler.async_anthropic_messages_handler(
            model="us.anthropic.claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            anthropic_messages_provider_config=_build_provider_config_mock(captured),
            anthropic_messages_optional_request_params=params,
            custom_llm_provider="bedrock",
            litellm_params=litellm_params,
            logging_obj=_build_logging_obj_mock(),
        )

    forwarded = captured["optional_params"]
    assert forwarded["max_tokens"] == 1024
    assert [t["name"] for t in forwarded["tools"]] == ["t0", "t1"]
    for tool in forwarded["tools"]:
        assert "input_examples" not in tool


@pytest.mark.asyncio
async def test_mixed_top_level_and_nested_drop_params():
    """Top-level and nested drops specified in the same list should both apply."""
    handler = BaseLLMHTTPHandler()
    captured: dict = {}

    params = {
        "context_management": {"edits": []},
        "tools": [{"name": "t", "input_examples": ["ex"]}],
        "max_tokens": 1024,
    }
    litellm_params = {
        "additional_drop_params": [
            "context_management",
            "tools[*].input_examples",
        ],
    }

    with pytest.raises(_StopAfterDrop):
        await handler.async_anthropic_messages_handler(
            model="us.anthropic.claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            anthropic_messages_provider_config=_build_provider_config_mock(captured),
            anthropic_messages_optional_request_params=params,
            custom_llm_provider="bedrock",
            litellm_params=litellm_params,
            logging_obj=_build_logging_obj_mock(),
        )

    forwarded = captured["optional_params"]
    assert "context_management" not in forwarded
    assert "input_examples" not in forwarded["tools"][0]
    assert forwarded["max_tokens"] == 1024
