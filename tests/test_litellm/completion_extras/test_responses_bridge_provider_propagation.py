"""
Regression test for https://github.com/BerriAI/litellm/issues/28505 -
the Responses API bridge double-strips the provider prefix from the
model name when a Chat Completions request has both `tools` and
`reasoning_effort`.

Root cause: the bridge handler called `litellm.responses()` /
`litellm.aresponses()` without passing the already-resolved
`custom_llm_provider`. The downstream call then re-invoked
`get_llm_provider()` with `custom_llm_provider=None`, which stripped
a second provider prefix from a `provider/provider/model` deployment
string.

This test pins both the sync and async bridge handler call sites:
the resolved `custom_llm_provider` must be forwarded to the underlying
`responses` / `aresponses` call so the provider isn't re-detected.
"""

from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.completion_extras.litellm_responses_transformation.handler import (
    ResponsesToCompletionBridgeHandler,
)

SYNC_RESPONSES_HANDLER = "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.response_api_handler"
ASYNC_RESPONSES_HANDLER = "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler"


def _bridge_tools():
    return [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
        }
    ]


def _validated_kwargs():
    return {
        "model": "openai/openai/openai/gpt-5.5",
        "messages": [{"role": "user", "content": "hi"}],
        "optional_params": {},
        "litellm_params": {},
        "headers": {},
        "model_response": MagicMock(),
        "logging_obj": MagicMock(),
        "custom_llm_provider": "openai",
    }


def test_sync_completion_forwards_custom_llm_provider():
    handler = ResponsesToCompletionBridgeHandler()
    handler.transformation_handler = MagicMock()
    handler.transformation_handler.transform_request.return_value = {
        "model": "openai/openai/openai/gpt-5.5",
        "input": [],
        # `_build_sanitized_litellm_params` spreads `custom_llm_provider` from
        # `litellm_params` into request_data on the real bridge path.  Seed
        # it here so the test exercises the overwrite (not an explicit kwarg
        # that would TypeError against an already-present key).
        "custom_llm_provider": "should-be-overwritten",
    }
    handler.transformation_handler.transform_response.return_value = (
        _validated_kwargs()["model_response"]
    )
    with (
        patch.object(
            handler, "validate_input_kwargs", return_value=_validated_kwargs()
        ),
        patch(
            "litellm.responses",
            return_value=MagicMock(spec=[]),
        ) as mock_responses,
    ):
        # The handler routes ResponsesAPIResponse through transform_response.
        # We just want to verify the kwargs going INTO responses().
        try:
            handler.completion(acompletion=False)
        except Exception:
            # Downstream handling (transform_response, type checks) is not
            # the subject of this test.
            pass
        assert mock_responses.called
        kwargs = mock_responses.call_args.kwargs
        assert kwargs.get("custom_llm_provider") == "openai", (
            "sync bridge must forward custom_llm_provider to litellm.responses() "
            "so the downstream get_llm_provider() call does not re-strip the "
            "provider prefix on a provider/provider/model deployment string"
        )


@pytest.mark.asyncio
async def test_async_completion_forwards_custom_llm_provider():
    handler = ResponsesToCompletionBridgeHandler()
    handler.transformation_handler = MagicMock()
    handler.transformation_handler.transform_request.return_value = {
        "model": "openai/openai/openai/gpt-5.5",
        "input": [],
        # `_build_sanitized_litellm_params` spreads `custom_llm_provider` from
        # `litellm_params` into request_data on the real bridge path.  Seed
        # it here so the test exercises the overwrite (not an explicit kwarg
        # that would TypeError against an already-present key).
        "custom_llm_provider": "should-be-overwritten",
    }

    async def _fake_aresponses(**kwargs):
        _fake_aresponses.kwargs = kwargs
        return MagicMock(spec=[])

    _fake_aresponses.kwargs = {}

    with (
        patch.object(
            handler, "validate_input_kwargs", return_value=_validated_kwargs()
        ),
        patch("litellm.aresponses", _fake_aresponses),
    ):
        try:
            await handler.acompletion()
        except Exception:
            pass
        assert _fake_aresponses.kwargs.get("custom_llm_provider") == "openai", (
            "async bridge must forward custom_llm_provider to litellm.aresponses() "
            "so the downstream get_llm_provider() call does not re-strip the "
            "provider prefix on a provider/provider/model deployment string"
        )


@pytest.mark.asyncio
async def test_async_completion_forwards_aws_region_name():
    handler = ResponsesToCompletionBridgeHandler()
    handler.transformation_handler = MagicMock()
    handler.transformation_handler.transform_request.return_value = {
        "model": "openai.gpt-5.5",
        "input": [],
        "aws_region_name": "us-east-2",
        "api_base": "https://bedrock-mantle.us-east-1.api.aws/v1",
        "custom_llm_provider": "bedrock_mantle",
    }

    async def _fake_aresponses(**kwargs):
        _fake_aresponses.kwargs = kwargs
        return MagicMock(spec=[])

    _fake_aresponses.kwargs = {}

    validated = _validated_kwargs()
    validated["custom_llm_provider"] = "bedrock_mantle"
    validated["litellm_params"] = {
        "aws_region_name": "us-east-2",
        "api_base": "https://bedrock-mantle.us-east-1.api.aws/v1",
        "custom_llm_provider": "bedrock_mantle",
    }

    with (
        patch.object(handler, "validate_input_kwargs", return_value=validated),
        patch("litellm.aresponses", _fake_aresponses),
    ):
        try:
            await handler.acompletion()
        except Exception:
            pass
        assert _fake_aresponses.kwargs.get("aws_region_name") == "us-east-2"


# Forwarding `custom_llm_provider` is necessary but not sufficient: `get_llm_provider()`
# strips a leading `provider/` off the model string whether or not a provider is passed
# in, so the bridge must also tell `responses()` not to resolve the model a second time.
# The tests below pin the model string that actually reaches the wire.


@pytest.mark.parametrize(
    "configured_model, expected_outgoing_model",
    [
        # litellm consumes one routing prefix; the remaining prefixes belong to the
        # upstream model name and must survive
        ("openai/openai/openai/gpt-5.5", "openai/openai/gpt-5.5"),
        ("openai/gpt-5.5", "gpt-5.5"),
        ("openai/azure/gpt-5.5", "azure/gpt-5.5"),
    ],
)
def test_sync_bridge_strips_exactly_one_provider_prefix(configured_model, expected_outgoing_model):
    captured = {}

    def _capture(*args, **kwargs):
        captured["model"] = kwargs.get("model")
        captured["custom_llm_provider"] = kwargs.get("custom_llm_provider")
        raise RuntimeError("stop before the network call")

    with patch(SYNC_RESPONSES_HANDLER, side_effect=_capture), pytest.raises(Exception):
        litellm.completion(
            model=configured_model,
            messages=[{"role": "user", "content": "hi"}],
            tools=_bridge_tools(),
            reasoning_effort="low",
            api_base="https://example.invalid",
            api_key="sk-fake",
        )

    assert captured["model"] == expected_outgoing_model
    assert captured["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_async_bridge_strips_exactly_one_provider_prefix():
    captured = {}

    async def _capture(*args, **kwargs):
        captured["model"] = kwargs.get("model")
        captured["custom_llm_provider"] = kwargs.get("custom_llm_provider")
        raise RuntimeError("stop before the network call")

    with patch(ASYNC_RESPONSES_HANDLER, side_effect=_capture), pytest.raises(Exception):
        await litellm.acompletion(
            model="openai/openai/openai/gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            tools=_bridge_tools(),
            reasoning_effort="low",
            api_base="https://example.invalid",
            api_key="sk-fake",
        )

    assert captured["model"] == "openai/openai/gpt-5.5"
    assert captured["custom_llm_provider"] == "openai"


def test_direct_responses_call_still_resolves_its_own_provider():
    captured = {}

    def _capture(*args, **kwargs):
        captured["model"] = kwargs.get("model")
        captured["custom_llm_provider"] = kwargs.get("custom_llm_provider")
        raise RuntimeError("stop before the network call")

    with patch(SYNC_RESPONSES_HANDLER, side_effect=_capture), pytest.raises(Exception):
        litellm.responses(
            model="openai/gpt-5.5",
            input="hi",
            api_base="https://example.invalid",
            api_key="sk-fake",
        )

    assert captured["model"] == "gpt-5.5"
    assert captured["custom_llm_provider"] == "openai"
