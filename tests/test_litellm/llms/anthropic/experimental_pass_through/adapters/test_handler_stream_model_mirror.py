"""
Regression tests: streaming ``/v1/messages`` must report the client-requested
model, not the internal post-routing deployment identifier.

Background — what was broken:
* Non-streaming ``/v1/messages`` responses already report the model name the
  client asked for: the proxy restamps the response object after the call
  (``_override_openai_response_model`` in
  ``litellm/proxy/common_request_processing.py``).
* Streaming responses bypass that restamp entirely: the adapter serializes the
  ``message_start`` event to SSE bytes before the proxy-level restamp runs (a
  bytes async-generator has no ``model`` attribute to restamp), and
  ``AnthropicStreamWrapper`` is constructed with the POST-routing model the
  handler received from the router (e.g. ``hosted_vllm/<internal-name>``).
* Net effect: for aliased/wildcard ``model_list`` deployments, streaming and
  non-streaming responses to the same endpoint disagree, and the streaming
  path leaks internal deployment identifiers to Anthropic-protocol clients
  (agent harnesses read ``message_start.model``).

The fix threads the client-requested name — which the router already stamps
into ``litellm_metadata["model_group"]`` — into the streaming translation via
``_client_facing_model``. SDK callers that bypass the proxy carry no such
metadata and keep the existing behavior.
"""

import json
from unittest.mock import patch

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
    _client_facing_model,
)

MESSAGES = [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# Helper-level coverage
# ---------------------------------------------------------------------------


def test_client_facing_model_prefers_model_group():
    assert (
        _client_facing_model(
            "hosted_vllm/internal-served-name",
            {"model_group": "claude-opus-5"},
        )
        == "claude-opus-5"
    )


def test_client_facing_model_without_metadata_keeps_model():
    assert _client_facing_model("hosted_vllm/internal-served-name", None) == (
        "hosted_vllm/internal-served-name"
    )


@pytest.mark.parametrize("bad_group", [None, "", 42, {"nested": "dict"}])
def test_client_facing_model_ignores_unusable_model_group(bad_group):
    assert (
        _client_facing_model("deployment-model", {"model_group": bad_group})
        == "deployment-model"
    )


# ---------------------------------------------------------------------------
# End-to-end through the async handler: message_start carries the alias
# ---------------------------------------------------------------------------


class _EmptyAsyncStream:
    """Minimal async completion stream: ends immediately.

    ``AnthropicStreamWrapper`` emits ``message_start`` before consuming the
    underlying stream, which is all these tests need.
    """

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


async def _first_message_start(transformed_stream) -> dict:
    async for raw in transformed_stream:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data: "):
                event = json.loads(line[len("data: ") :])
                if event.get("type") == "message_start":
                    return event
    raise AssertionError("no message_start event emitted")


@pytest.mark.asyncio
async def test_streaming_message_start_mirrors_requested_alias():
    """Proxy-routed call: litellm_metadata.model_group wins over the
    post-routing deployment model."""
    with patch("litellm.acompletion", return_value=_EmptyAsyncStream()):  # test-quality-ok: the handler calls litellm.acompletion directly, no injection seam
        transformed_stream = await (
            LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
                max_tokens=16,
                messages=MESSAGES,
                model="hosted_vllm/internal-served-name",
                stream=True,
                litellm_metadata={"model_group": "claude-opus-5"},
            )
        )
        event = await _first_message_start(transformed_stream)
        assert event["message"]["model"] == "claude-opus-5"


@pytest.mark.asyncio
async def test_streaming_message_start_without_proxy_metadata_unchanged():
    """SDK-style call (no proxy metadata, no provider): the model passed to
    the handler is reported as-is."""
    with patch("litellm.acompletion", return_value=_EmptyAsyncStream()):  # test-quality-ok: the handler calls litellm.acompletion directly, no injection seam
        transformed_stream = await (
            LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
                max_tokens=16,
                messages=MESSAGES,
                model="hosted_vllm/internal-served-name",
                stream=True,
            )
        )
        event = await _first_message_start(transformed_stream)
        assert event["message"]["model"] == "hosted_vllm/internal-served-name"


@pytest.mark.asyncio
async def test_streaming_message_start_without_proxy_metadata_strips_provider_prefix():
    """SDK-style call with a resolved provider: falls back to the
    provider-local name (``local_model_name``), not the alias path."""
    with patch("litellm.acompletion", return_value=_EmptyAsyncStream()):  # test-quality-ok: the handler calls litellm.acompletion directly, no injection seam
        transformed_stream = await (
            LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
                max_tokens=16,
                messages=MESSAGES,
                model="hosted_vllm/internal-served-name",
                stream=True,
                custom_llm_provider="hosted_vllm",
            )
        )
        event = await _first_message_start(transformed_stream)
        assert event["message"]["model"] == "internal-served-name"
