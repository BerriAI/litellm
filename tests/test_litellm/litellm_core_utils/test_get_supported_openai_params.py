import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,
)

BEDROCK_REAL_MODEL = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_LABEL = "claude-haiku-4-5"


def test_base_model_label_does_not_strip_bedrock_tools():
    """Regression for #29618.

    A Bedrock deployment whose ``model_info.base_model`` is a friendly label
    (``claude-haiku-4-5``) must still advertise ``tools``/``tool_choice``. The label
    on its own resolves to no tool support, so before the fix it stripped the
    capability the real model id exposes, silently dropping function calling under
    ``drop_params``."""
    params = get_supported_openai_params(
        model=BEDROCK_REAL_MODEL,
        custom_llm_provider="bedrock",
        base_model=BEDROCK_LABEL,
    )

    assert params is not None
    assert "tools" in params
    assert "tool_choice" in params


def test_base_model_label_alone_lacks_bedrock_tools():
    """The label by itself does not advertise tools; this is what made the union
    necessary. Guards against the discrepancy disappearing (and the regression test
    above silently passing for the wrong reason)."""
    params = get_supported_openai_params(
        model=BEDROCK_LABEL, custom_llm_provider="bedrock"
    )

    assert params is not None
    assert "tools" not in params


def test_base_model_is_additive_not_replacement():
    """``base_model`` may only add capabilities, never remove ones the real model has.

    Bedrock: real id supports ``tools`` but not the label's reasoning hint; the union
    must contain the real model's ``tools`` regardless of the label being a subset."""
    real_only = set(
        get_supported_openai_params(
            model=BEDROCK_REAL_MODEL, custom_llm_provider="bedrock"
        )
    )
    label_only = set(
        get_supported_openai_params(model=BEDROCK_LABEL, custom_llm_provider="bedrock")
    )
    combined = set(
        get_supported_openai_params(
            model=BEDROCK_REAL_MODEL,
            custom_llm_provider="bedrock",
            base_model=BEDROCK_LABEL,
        )
    )

    assert combined == real_only | label_only
    assert real_only - label_only  # the label really is a strict subset here
    assert real_only <= combined


def test_base_model_adds_capabilities_the_real_model_lacks():
    """Regression for #27717 (the behavior the union must preserve).

    ``gemini-3.1-pro`` isn't in the cost map so it advertises no reasoning support,
    but the registered ``gemini-3.1-pro-preview`` base_model does. The hint must add
    ``reasoning_effort``/``thinking`` without the call erroring."""
    real_only = set(
        get_supported_openai_params(
            model="gemini-3.1-pro", custom_llm_provider="gemini"
        )
    )
    assert "reasoning_effort" not in real_only

    combined = set(
        get_supported_openai_params(
            model="gemini-3.1-pro",
            custom_llm_provider="gemini",
            base_model="gemini-3.1-pro-preview",
        )
    )
    assert "reasoning_effort" in combined
    assert "thinking" in combined


def test_no_base_model_is_unchanged():
    """Omitting ``base_model`` must resolve purely from ``model``."""
    with_none = get_supported_openai_params(
        model=BEDROCK_REAL_MODEL, custom_llm_provider="bedrock", base_model=None
    )
    plain = get_supported_openai_params(
        model=BEDROCK_REAL_MODEL, custom_llm_provider="bedrock"
    )

    assert with_none == plain


def test_base_model_equal_to_model_is_unchanged():
    """A ``base_model`` identical to ``model`` must not double-resolve or reorder."""
    plain = get_supported_openai_params(
        model=BEDROCK_REAL_MODEL, custom_llm_provider="bedrock"
    )
    same = get_supported_openai_params(
        model=BEDROCK_REAL_MODEL,
        custom_llm_provider="bedrock",
        base_model=BEDROCK_REAL_MODEL,
    )

    assert same == plain


def test_azure_base_model_detection_preserved():
    """Azure relies on ``base_model`` for model-type detection when the deployment
    name is opaque; the union must keep advertising the gpt-5 capabilities."""
    params = get_supported_openai_params(
        model="my-opaque-deployment",
        custom_llm_provider="azure",
        base_model="azure/gpt-5.2",
    )

    assert params is not None
    assert "reasoning_effort" in params
    assert "tools" in params
