"""
Tests for LiteLLMMessagesToCompletionTransformationHandler
"""

from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)


def test_prepare_completion_kwargs_excludes_output_config():
    """
    Verify that `output_config` (an Anthropic-only parameter) is stripped when
    translating an Anthropic Messages request into litellm.completion() kwargs.

    The Claude Agent SDK sends `output_config` in its request body. For
    non-Anthropic providers (e.g. Bedrock Nova), this parameter leaks through
    **kwargs → extra_kwargs → completion_kwargs, causing Bedrock to reject the
    request with "extraneous key [output_config] is not permitted".

    Regression test for: https://github.com/BerriAI/litellm/issues/22797
    """
    completion_kwargs, _ = (
        LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
            model="bedrock/us.amazon.nova-pro-v1:0",
            extra_kwargs={
                "output_config": {"type": "text"},
                "custom_llm_provider": "bedrock",
            },
        )
    )
    assert "output_config" not in completion_kwargs, (
        "output_config should be excluded from completion kwargs; "
        "it is an Anthropic-only param that Bedrock rejects."
    )
    # custom_llm_provider should still be passed through
    assert completion_kwargs.get("custom_llm_provider") == "bedrock"


def test_prepare_completion_kwargs_excludes_anthropic_messages():
    """
    Verify that `anthropic_messages` is also excluded from completion kwargs
    (pre-existing behavior).
    """
    completion_kwargs, _ = (
        LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
            model="bedrock/us.amazon.nova-pro-v1:0",
            extra_kwargs={
                "anthropic_messages": True,
                "custom_llm_provider": "bedrock",
            },
        )
    )
    assert "anthropic_messages" not in completion_kwargs
