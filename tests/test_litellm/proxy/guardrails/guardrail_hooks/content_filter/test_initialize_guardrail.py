"""Regression for #30008 — keyword_redaction_tag / pattern_redaction_format must be
propagated from LitellmParams through initialize_guardrail to ContentFilterGuardrail."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../../"))

import litellm
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter import (
    initialize_guardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)


def _make_litellm_params(**overrides):
    """Build a permissive stand-in for LitellmParams — fields are read via getattr."""
    defaults = dict(
        patterns=None,
        blocked_words=None,
        blocked_words_file=None,
        mode="pre_call",
        default_on=False,
        categories=None,
        severity_threshold="medium",
        image_model=None,
        competitor_intent_config=None,
        end_session_after_n_fails=None,
        on_violation=None,
        realtime_violation_message=None,
        keyword_redaction_tag=None,
        pattern_redaction_format=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def _quiet_callback_register():
    """initialize_guardrail registers a callback on success — keep test side-effect-free."""
    with patch.object(litellm.logging_callback_manager, "add_litellm_callback", lambda _: None):
        yield


def test_initialize_guardrail_propagates_custom_redaction_tags():
    """Custom keyword_redaction_tag + pattern_redaction_format must reach the filter."""
    params = _make_litellm_params(
        keyword_redaction_tag="***REDACTED***",
        pattern_redaction_format="***{pattern_name}***",
    )
    guardrail = {"guardrail_name": "test_filter"}

    filter_obj = initialize_guardrail(litellm_params=params, guardrail=guardrail)

    assert isinstance(filter_obj, ContentFilterGuardrail)
    assert filter_obj.keyword_redaction_tag == "***REDACTED***"
    assert filter_obj.pattern_redaction_format == "***{pattern_name}***"


def test_initialize_guardrail_defaults_when_redaction_tags_missing():
    """Omitting the redaction tags falls back to the class defaults (current behavior)."""
    params = _make_litellm_params()
    guardrail = {"guardrail_name": "test_filter_defaults"}

    filter_obj = initialize_guardrail(litellm_params=params, guardrail=guardrail)

    assert filter_obj.keyword_redaction_tag == ContentFilterGuardrail.KEYWORD_REDACTION_STR
    assert filter_obj.pattern_redaction_format == ContentFilterGuardrail.PATTERN_REDACTION_FORMAT
