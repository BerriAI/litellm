from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


import litellm
from litellm.completion_extras.litellm_responses_transformation.handler import (
    ResponsesToCompletionBridgeHandler,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ModelResponse


def test_is_preformatted_cached_chat_stream_true():
    stream = MagicMock(spec=CustomStreamWrapper)
    stream.custom_llm_provider = "cached_response"
    assert (
        ResponsesToCompletionBridgeHandler._is_preformatted_cached_chat_stream(stream)
        is True
    )


def test_is_preformatted_cached_chat_stream_false_wrong_provider():
    stream = MagicMock(spec=CustomStreamWrapper)
    stream.custom_llm_provider = "openai"
    assert (
        ResponsesToCompletionBridgeHandler._is_preformatted_cached_chat_stream(stream)
        is False
    )


def test_is_preformatted_cached_chat_stream_false_wrong_type():
    assert (
        ResponsesToCompletionBridgeHandler._is_preformatted_cached_chat_stream(
            {"object": "chat.completion.chunk"}
        )
        is False
    )


def _bridge_kwargs(stream: bool):
    logging_obj = LiteLLMLogging(
        litellm_call_id="test-call",
        call_type="completion",
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        function_id="fn-id",
        stream=stream,
        start_time=datetime.now(),
    )
    return {
        "model": "gpt-5.4",
        "custom_llm_provider": "openai",
        "messages": [{"role": "user", "content": "hi"}],
        "optional_params": {"stream": stream},
        "litellm_params": {},
        "headers": {},
        "model_response": ModelResponse(),
        "logging_obj": logging_obj,
    }


def test_completion_returns_cached_model_response_directly():
    """Non-streaming bridge cache hit: responses() returns a ModelResponse -> bridge returns it as-is."""
    cached = ModelResponse(id="chatcmpl-cached-nonstream", model="gpt-5.4")
    bridge = ResponsesToCompletionBridgeHandler()

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ),
        patch("litellm.responses", return_value=cached),
    ):
        result = bridge.completion(**_bridge_kwargs(stream=False))

    assert result is cached


@pytest.mark.asyncio
async def test_acompletion_returns_cached_model_response_directly():
    cached = ModelResponse(id="chatcmpl-cached-nonstream-async", model="gpt-5.4")
    bridge = ResponsesToCompletionBridgeHandler()

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ),
        patch("litellm.aresponses", new=AsyncMock(return_value=cached)),
    ):
        result = await bridge.acompletion(**_bridge_kwargs(stream=False))

    assert result is cached


def test_completion_skips_rewrapping_preformatted_cached_chat_stream():
    """Streaming bridge cache hit returning CustomStreamWrapper(cached_response) -> bridge skips re-wrapping."""
    stream = MagicMock(spec=CustomStreamWrapper)
    stream.custom_llm_provider = "cached_response"
    bridge = ResponsesToCompletionBridgeHandler()

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ),
        patch("litellm.responses", return_value=stream),
        patch.object(
            bridge,
            "_apply_post_stream_processing",
            side_effect=lambda s, *a, **kw: s,
        ) as post,
    ):
        result = bridge.completion(**_bridge_kwargs(stream=True))

    post.assert_called_once()
    assert result is stream


def test_completion_preserves_top_level_stream_flag_in_responses_request():
    stream = MagicMock(spec=CustomStreamWrapper)
    stream.custom_llm_provider = "cached_response"
    bridge = ResponsesToCompletionBridgeHandler()
    kwargs = _bridge_kwargs(stream=False)
    kwargs["stream"] = True
    kwargs["optional_params"].pop("stream")

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ) as transform_request,
        patch("litellm.responses", return_value=stream),
        patch.object(
            bridge,
            "_apply_post_stream_processing",
            side_effect=lambda s, *a, **kw: s,
        ),
    ):
        result = bridge.completion(**kwargs)

    assert result is stream
    assert transform_request.call_args.kwargs["optional_params"]["stream"] is True


@pytest.mark.asyncio
async def test_acompletion_skips_rewrapping_preformatted_cached_chat_stream():
    stream = MagicMock(spec=CustomStreamWrapper)
    stream.custom_llm_provider = "cached_response"
    bridge = ResponsesToCompletionBridgeHandler()

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ),
        patch("litellm.aresponses", new=AsyncMock(return_value=stream)),
        patch.object(
            bridge,
            "_apply_post_stream_processing",
            side_effect=lambda s, *a, **kw: s,
        ) as post,
    ):
        result = await bridge.acompletion(**_bridge_kwargs(stream=True))

    post.assert_called_once()
    assert result is stream


@pytest.mark.asyncio
async def test_acompletion_preserves_top_level_stream_flag_in_responses_request():
    stream = MagicMock(spec=CustomStreamWrapper)
    stream.custom_llm_provider = "cached_response"
    bridge = ResponsesToCompletionBridgeHandler()
    kwargs = _bridge_kwargs(stream=False)
    kwargs["stream"] = True
    kwargs["optional_params"].pop("stream")

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ) as transform_request,
        patch("litellm.aresponses", new=AsyncMock(return_value=stream)),
        patch.object(
            bridge,
            "_apply_post_stream_processing",
            side_effect=lambda s, *a, **kw: s,
        ),
    ):
        result = await bridge.acompletion(**kwargs)

    assert result is stream
    assert transform_request.call_args.kwargs["optional_params"]["stream"] is True


def _completed_chat_response() -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-completed",
        model="gpt-5.4",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "pong"},
                "finish_reason": "stop",
            }
        ],
    )


@pytest.mark.asyncio
async def test_acompletion_streams_completed_model_response():
    """A streaming request whose bridge call comes back already completed must still be
    handed back as an async-iterable stream. Returning the bare ModelResponse crashed the
    proxy's SSE generator with "'async for' requires an object with __aiter__ method".
    Regression for #33154."""
    completed = _completed_chat_response()
    bridge = ResponsesToCompletionBridgeHandler()

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ),
        patch("litellm.aresponses", new=AsyncMock(return_value=completed)),
    ):
        result = await bridge.acompletion(**_bridge_kwargs(stream=True))

    assert isinstance(result, CustomStreamWrapper), f"streaming request got {type(result)}"
    chunks = [chunk async for chunk in result]
    assert "".join(
        chunk.choices[0].delta.content or "" for chunk in chunks
    ) == "pong", f"completed response did not stream its content: {chunks}"
    assert [c for c in chunks if c.choices[0].finish_reason], "stream never emitted a finish_reason"


def test_completion_streams_completed_model_response():
    completed = _completed_chat_response()
    bridge = ResponsesToCompletionBridgeHandler()

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ),
        patch("litellm.responses", return_value=completed),
    ):
        result = bridge.completion(**_bridge_kwargs(stream=True))

    assert isinstance(result, CustomStreamWrapper), f"streaming request got {type(result)}"
    chunks = list(result)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "pong", (
        f"completed response did not stream its content: {chunks}"
    )


_PROVIDER_NATIVE_MODEL_CASES = [
    ("perplexity", "perplexity/kimi-k3", "perplexity/kimi-k3"),
    ("perplexity", "openai/gpt-5.2", "openai/gpt-5.2"),
    ("openai", "gpt-5.4", "gpt-5.4"),
]


def _upstream_model_for(handed_model: str, custom_llm_provider: str) -> str:
    upstream_model, _, _, _ = litellm.get_llm_provider(
        model=handed_model,
        litellm_params=GenericLiteLLMParams(custom_llm_provider=custom_llm_provider),
    )
    return upstream_model


@pytest.mark.parametrize(
    "custom_llm_provider, bridge_model, expected_upstream_model",
    _PROVIDER_NATIVE_MODEL_CASES,
)
def test_completion_keeps_provider_native_model_id_through_responses(
    custom_llm_provider, bridge_model, expected_upstream_model
):
    """responses() resolves the provider itself, so the bridge must not hand it an already-stripped model."""
    cached = ModelResponse(id="chatcmpl-cached", model=bridge_model)
    bridge = ResponsesToCompletionBridgeHandler()
    kwargs = _bridge_kwargs(stream=False)
    kwargs["model"] = bridge_model
    kwargs["custom_llm_provider"] = custom_llm_provider

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": bridge_model, "input": "hi"},
        ),
        patch("litellm.responses", return_value=cached) as responses_call,
    ):
        bridge.completion(**kwargs)

    handed_model = responses_call.call_args.kwargs["model"]
    assert _upstream_model_for(handed_model, custom_llm_provider) == expected_upstream_model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "custom_llm_provider, bridge_model, expected_upstream_model",
    _PROVIDER_NATIVE_MODEL_CASES,
)
async def test_acompletion_keeps_provider_native_model_id_through_responses(
    custom_llm_provider, bridge_model, expected_upstream_model
):
    cached = ModelResponse(id="chatcmpl-cached-async", model=bridge_model)
    bridge = ResponsesToCompletionBridgeHandler()
    kwargs = _bridge_kwargs(stream=False)
    kwargs["model"] = bridge_model
    kwargs["custom_llm_provider"] = custom_llm_provider

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": bridge_model, "input": "hi"},
        ),
        patch("litellm.aresponses", new=AsyncMock(return_value=cached)) as responses_call,
    ):
        await bridge.acompletion(**kwargs)

    handed_model = responses_call.call_args.kwargs["model"]
    assert _upstream_model_for(handed_model, custom_llm_provider) == expected_upstream_model
