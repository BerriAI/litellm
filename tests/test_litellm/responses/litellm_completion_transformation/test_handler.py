"""Regression tests for the responses -> completion fallback bridge guard.

When the Responses API falls back to chat completions (no native responses
config), it must tag the forwarded ``litellm.completion`` / ``litellm.acompletion``
call with ``_skip_responses_api_bridge=True`` so ``completion()`` does not bridge
the request straight back to the Responses API and mutually recurse forever.

Both fallback paths are covered: the sync ``response_api_handler`` (``_is_async``
False) and the async ``async_response_api_handler`` (``_is_async`` True). The
module-level ``litellm.completion`` / ``litellm.acompletion`` are patched to
capture the forwarded kwargs; if the flag-setting line is removed the captured
kwargs lack the flag and these tests fail.

The same capture also covers which kwargs the bridge forwards. Responses API
request params it cannot translate must be dropped, because unrecognised keys
are swept into the provider request downstream; LiteLLM-level and
provider-specific kwargs must still pass through untouched.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)


class _StopForwarding(Exception):
    """Raised by the mocked (a)completion once the forwarded kwargs are captured."""


def test_sync_fallback_tags_skip_responses_api_bridge():
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.completion", fake_completion):
        with pytest.raises(_StopForwarding):
            handler.response_api_handler(
                model="gpt-4o",
                input="hello",
                responses_api_request={},
                custom_llm_provider="openai",
                _is_async=False,
            )

    assert captured.get("_skip_responses_api_bridge") is True


@pytest.mark.asyncio
async def test_async_fallback_tags_skip_responses_api_bridge():
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.acompletion", fake_acompletion):
        coro = handler.response_api_handler(
            model="gpt-4o",
            input="hello",
            responses_api_request={},
            custom_llm_provider="openai",
            _is_async=True,
        )
        with pytest.raises(_StopForwarding):
            await coro

    assert captured.get("_skip_responses_api_bridge") is True


UNTRANSLATED_RESPONSES_PARAMS: dict[str, object] = {
    "prompt_cache_key": "pck-1",
    "prompt_cache_retention": "24h",
    "max_tool_calls": 3,
    "partial_images": 2,
    "top_logprobs": 2,
}

PRESERVED_KWARGS: dict[str, object] = {
    "litellm_metadata": {"model_group": "claude"},
    "aws_region_name": "us-west-2",
    "stream_chunk_size": 1,
    "guided_json": {"type": "object"},
}


def _capture_sync_forwarded_kwargs(**handler_kwargs: object) -> dict[str, object]:
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict[str, object] = {}

    def fake_completion(**kwargs: object) -> None:
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.completion", fake_completion):
        with pytest.raises(_StopForwarding):
            handler.response_api_handler(
                model="gpt-4o",
                input="hello",
                responses_api_request={},
                custom_llm_provider="openai",
                _is_async=False,
                **handler_kwargs,
            )
    return captured


async def _capture_async_forwarded_kwargs(**handler_kwargs: object) -> dict[str, object]:
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> None:
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.acompletion", fake_acompletion):
        coro = handler.response_api_handler(
            model="gpt-4o",
            input="hello",
            responses_api_request={},
            custom_llm_provider="openai",
            _is_async=True,
            **handler_kwargs,
        )
        with pytest.raises(_StopForwarding):
            await coro
    return captured


def test_sync_fallback_drops_untranslated_responses_params():
    """Responses params the bridge cannot translate must not reach litellm.completion."""
    captured = _capture_sync_forwarded_kwargs(**UNTRANSLATED_RESPONSES_PARAMS)

    for param in UNTRANSLATED_RESPONSES_PARAMS:
        assert param not in captured, f"{param} was forwarded to the provider"


@pytest.mark.asyncio
async def test_async_fallback_drops_untranslated_responses_params():
    """The async handler merges kwargs separately, so it needs the same guard."""
    captured = await _capture_async_forwarded_kwargs(**UNTRANSLATED_RESPONSES_PARAMS)

    for param in UNTRANSLATED_RESPONSES_PARAMS:
        assert param not in captured, f"{param} was forwarded to the provider"


def test_sync_fallback_preserves_litellm_and_provider_kwargs():
    """Filtering must not turn into an allowlist: LiteLLM and deployment kwargs still pass."""
    captured = _capture_sync_forwarded_kwargs(metadata={"user": "u1"}, temperature=0.5, **PRESERVED_KWARGS)

    for param, value in PRESERVED_KWARGS.items():
        assert captured.get(param) == value, f"{param} was dropped"
    assert captured.get("metadata") == {"user": "u1"}
    assert captured.get("temperature") == 0.5


@pytest.mark.asyncio
async def test_async_fallback_preserves_litellm_and_provider_kwargs():
    """Same guard on the async path."""
    captured = await _capture_async_forwarded_kwargs(metadata={"user": "u1"}, temperature=0.5, **PRESERVED_KWARGS)

    for param, value in PRESERVED_KWARGS.items():
        assert captured.get(param) == value, f"{param} was dropped"
    assert captured.get("metadata") == {"user": "u1"}
    assert captured.get("temperature") == 0.5


def test_supported_responses_params_are_never_dropped():
    """The drop set is derived, so a param the bridge does translate must survive it."""
    from litellm.responses.litellm_completion_transformation.handler import (
        _drop_untranslated_responses_params,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    supported = LiteLLMCompletionResponsesConfig.get_supported_openai_params("gpt-4o")
    kwargs = {param: "value" for param in supported}

    assert _drop_untranslated_responses_params(kwargs, "gpt-4o") == kwargs
