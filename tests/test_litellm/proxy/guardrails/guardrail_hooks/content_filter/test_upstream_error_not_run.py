import openai
from unittest.mock import MagicMock

import pytest

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)
from litellm.types.guardrails import GuardrailEventHooks


@pytest.mark.asyncio
async def test_streaming_hook_upstream_error_is_not_run():
    """
    When the upstream stream raises a provider error (openai.OpenAIError
    subclass), the content filter never evaluates content. The error must
    propagate unchanged and be logged as "not_run", not as a guardrail
    failure. Regression test for issue #31004.
    """
    guardrail = ContentFilterGuardrail(
        guardrail_name="test-streaming-upstream-error",
        patterns=[],
        event_hook=GuardrailEventHooks.during_call,
    )

    async def mock_stream():
        raise openai.APIError("Provider returned error", request=MagicMock(), body=None)
        yield  # pragma: no cover - makes this an async generator

    user_api_key_dict = MagicMock()
    request_data: dict = {}

    with pytest.raises(openai.APIError):
        async for _ in guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=user_api_key_dict,
            response=mock_stream(),
            request_data=request_data,
        ):
            pass

    info = request_data["metadata"]["standard_logging_guardrail_information"][0]
    assert info["guardrail_status"] == "not_run"
