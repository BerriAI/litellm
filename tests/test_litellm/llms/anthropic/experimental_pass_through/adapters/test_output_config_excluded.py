import os
import sys
from typing import Any, Dict

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)


def test_output_config_excluded_from_completion_kwargs():
    """
    Test that `output_config` is excluded when translating Anthropic messages
    to OpenAI completion format.

    When Claude Code sends requests via the Anthropic Messages API to a
    non-Anthropic provider (e.g., Azure OpenAI, Fireworks), the adapter
    translates Anthropic params to OpenAI format. `output_config` is an
    Anthropic-specific parameter that non-Anthropic providers reject with
    a 400 error ("Extra inputs are not permitted"). This test verifies
    that `output_config` is properly stripped during translation.

    Related: https://github.com/BerriAI/litellm/issues/22963
    """
    handler = LiteLLMMessagesToCompletionTransformationHandler()

    completion_kwargs, _ = handler._prepare_completion_kwargs(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
        model="azure/my-deployment",
        extra_kwargs={
            "output_config": {"effort": "standard"},
        },
    )

    # output_config should NOT be present in the completion kwargs
    assert "output_config" not in completion_kwargs, (
        "output_config should be excluded from completion kwargs when translating "
        "Anthropic messages to OpenAI format. Non-Anthropic providers reject this "
        "Anthropic-specific parameter."
    )


def test_anthropic_messages_excluded_from_completion_kwargs():
    """
    Verify that `anthropic_messages` is also excluded (existing behavior).
    """
    handler = LiteLLMMessagesToCompletionTransformationHandler()

    completion_kwargs, _ = handler._prepare_completion_kwargs(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
        model="azure/my-deployment",
        extra_kwargs={
            "anthropic_messages": [{"role": "user", "content": "test"}],
        },
    )

    assert "anthropic_messages" not in completion_kwargs


def test_valid_extra_kwargs_still_passed_through():
    """
    Verify that legitimate extra kwargs that are NOT in the excluded list
    are still passed through to completion kwargs.
    """
    handler = LiteLLMMessagesToCompletionTransformationHandler()

    completion_kwargs, _ = handler._prepare_completion_kwargs(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
        model="azure/my-deployment",
        extra_kwargs={
            "output_config": {"effort": "standard"},  # should be excluded
            "some_valid_param": "some_value",  # should pass through
        },
    )

    assert "output_config" not in completion_kwargs
    assert completion_kwargs.get("some_valid_param") == "some_value", (
        "Valid extra kwargs should still be passed through to completion kwargs"
    )
