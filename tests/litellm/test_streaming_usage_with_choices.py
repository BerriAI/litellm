"""Regression tests for provider usage on non-empty streaming choices."""

from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

import litellm
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types import utils
from litellm.types.llms import openai
from litellm.types.router import GenericLiteLLMParams


class _LoggingObject:
    model_call_details = {"litellm_params": {}, "custom_llm_provider": "custom_openai"}
    optional_params = {}
    stream_options = {"include_usage": True}
    messages = [{"role": "user", "content": "hello"}]


def _rebuild_litellm_response_types() -> None:
    """Resolve forward references when the test environment uses lazy Pydantic models."""
    namespace = {**vars(utils), **vars(openai)}
    for model in (utils.Delta, utils.StreamingChoices, utils.ModelResponseStream, GenericLiteLLMParams):
        model.model_rebuild(_types_namespace=namespace, raise_errors=False)


def test_streaming_usage_is_preserved_when_choices_are_non_empty():
    _rebuild_litellm_response_types()
    chunk = litellm.ModelResponseStream(
        id="chatcmpl-usage-test",
        created=1,
        model="custom_openai/m",
        choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
    )
    chunk.usage = CompletionUsage(
        prompt_tokens=39779,
        completion_tokens=32,
        total_tokens=39811,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=39424),
    )

    wrapper = CustomStreamWrapper(
        completion_stream=iter([chunk]),
        model="custom_openai/m",
        logging_obj=_LoggingObject(),
        custom_llm_provider="custom_openai",
        stream_options={"include_usage": True},
    )

    processed = wrapper.chunk_creator(chunk)

    assert processed is not None
    assert isinstance(processed.usage, litellm.Usage)
    assert processed.usage.prompt_tokens == 39779
    assert processed.usage.prompt_tokens_details is not None
    assert processed.usage.prompt_tokens_details.cached_tokens == 39424
