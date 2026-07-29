"""
Regression tests for output_config passthrough through the Anthropic
``/v1/messages`` → ``/chat/completions`` adapter.

Background — what was broken:
* When a client sent ``output_config`` to ``/v1/messages`` and the request
  was routed to a non-Anthropic backend (Azure OpenAI, Fireworks, Bedrock
  Nova, etc.), the adapter forwarded the raw Anthropic-shaped ``output_config``
  field as-is into the OpenAI-format ``completion_kwargs``. The non-Anthropic
  backend then rejected the request with 400 "Extra inputs are not permitted".
* The translator above the re-merge already extracts the meaningful parts of
  ``output_config`` (``format`` → ``response_format``, ``effort`` →
  ``reasoning_effort`` for non-Claude targets), so re-adding the raw key was
  always either redundant (Anthropic-family) or harmful (non-Anthropic).

Tests cover (consolidating PRs #23706 and #22727):
1. ``output_config`` is excluded from the post-translation re-merge.
2. ``ANTHROPIC_ONLY_REQUEST_KEYS`` constant is exported and contains
   ``output_config`` so future maintainers know where to extend it.
3. The translator-extracted fields (``response_format`` / ``reasoning_effort``)
   are still present after the strip — the strip removes only the raw
   Anthropic-shaped duplicate.
4. Helper-level coverage for empty ``extra_kwargs`` (PR #22727 Greptile P2 —
   the original ``or {}`` pattern silently substituted a default and prevented
   the fallback inference path from being exercised).
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Anchor sys.path to this file's location — not the working-directory-relative
# pattern Greptile flagged on PR #23706. Resolves correctly regardless of
# where pytest is invoked from.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
)

from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    ANTHROPIC_ONLY_REQUEST_KEYS,
    LiteLLMMessagesToCompletionTransformationHandler,
)

MESSAGES = [{"role": "user", "content": "hello"}]


def _call_prepare(extra_kwargs, model="gpt-4o", output_format=None, **overrides):
    """
    Drive ``_prepare_completion_kwargs`` with the minimum scaffolding needed.

    ``output_format`` is a top-level parameter on the function, so callers
    pass it explicitly here rather than tucking it into ``extra_kwargs``.

    Uses an explicit-None check on ``extra_kwargs`` so callers can test the
    falsy-empty-dict path. The fallback ``or {}`` pattern PR #22727 used here
    masked the no-extra-kwargs case from ever exercising the test's intent.
    """
    return LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
        max_tokens=overrides.get("max_tokens", 1024),
        messages=overrides.get("messages", MESSAGES),
        model=model,
        metadata=None,
        stop_sequences=None,
        stream=False,
        system=None,
        temperature=None,
        thinking=None,
        tool_choice=None,
        tools=None,
        top_k=None,
        top_p=None,
        output_format=output_format,
        extra_kwargs=extra_kwargs,
    )


class TestAnthropicOnlyRequestKeysExport:
    """The exclusion list must be a public, named constant for maintainability —
    Greptile P2 on PR #23706: ``excluded_keys`` was silently growing as a
    point-fix pattern. A named module-level constant gives reviewers a single
    grep target when extending Anthropic-only fields."""

    def test_constant_exposed(self):
        assert isinstance(ANTHROPIC_ONLY_REQUEST_KEYS, frozenset)

    def test_contains_output_config(self):
        assert "output_config" in ANTHROPIC_ONLY_REQUEST_KEYS


class TestOutputConfigStrippedFromCompletionKwargs:
    """``output_config`` must not survive the post-translation re-merge into
    ``completion_kwargs`` regardless of the target provider — the translator
    has already consumed its meaningful parts."""

    def test_output_config_with_effort_is_stripped(self):
        extra_kwargs = {
            "custom_llm_provider": "azure",
            "output_config": {"effort": "high"},
        }

        result = _call_prepare(extra_kwargs=extra_kwargs)

        # Returns (completion_kwargs, original_messages, ...) — first element
        # is the dict we care about.
        completion_kwargs = result[0] if isinstance(result, tuple) else result
        assert "output_config" not in completion_kwargs, (
            "Raw output_config must not be forwarded — non-Anthropic backends "
            "reject it with 400 'Extra inputs are not permitted'"
        )

    def test_output_config_format_translated_to_response_format(self):
        """When ``output_config`` carries structured-output ``format``, the
        translator now maps it to OpenAI's ``response_format`` so non-Anthropic
        backends see the schema in their native shape. The raw
        ``output_config`` key is still stripped from ``completion_kwargs`` —
        only the translated ``response_format`` survives.

        Before this PR, only the legacy top-level ``output_format`` was
        translated; ``output_config.format`` was silently dropped on the
        adapter path even when the schema was correctly supplied (issue
        flagged by Greptile review of the initial fix).
        """
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"name": {"type": "string"}},
        }
        extra_kwargs = {
            "custom_llm_provider": "azure",
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }

        result = _call_prepare(extra_kwargs=extra_kwargs)
        completion_kwargs = result[0] if isinstance(result, tuple) else result

        # Raw Anthropic-shaped key is gone (would 400 on non-Anthropic backends).
        assert "output_config" not in completion_kwargs
        # Translated OpenAI-shaped key is present so the schema actually
        # reaches the downstream backend.
        assert "response_format" in completion_kwargs, (
            "output_config.format must be translated to response_format — "
            "without this, structured-output schemas are silently dropped on "
            "the adapter path"
        )

    def test_output_format_top_level_still_translates(self):
        """Regression guard: the legacy top-level ``output_format`` field must
        continue to translate to ``response_format``. The new
        ``output_config.format`` path must not break this existing behavior."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = _call_prepare(
            extra_kwargs={"custom_llm_provider": "azure"},
            output_format={"type": "json_schema", "schema": schema},
        )
        completion_kwargs = result[0] if isinstance(result, tuple) else result

        assert "response_format" in completion_kwargs

    def test_output_format_takes_precedence_over_output_config_format(self):
        """When both top-level ``output_format`` and ``output_config.format``
        are present, the legacy top-level ``output_format`` wins. Documents
        which one the translator picks rather than leaving it implementation-
        defined."""
        winning_schema = {
            "type": "object",
            "properties": {"top_level": {"type": "string"}},
        }
        losing_schema = {
            "type": "object",
            "properties": {"nested": {"type": "string"}},
        }
        result = _call_prepare(
            extra_kwargs={
                "custom_llm_provider": "azure",
                "output_config": {
                    "format": {"type": "json_schema", "schema": losing_schema}
                },
            },
            output_format={"type": "json_schema", "schema": winning_schema},
        )
        completion_kwargs = result[0] if isinstance(result, tuple) else result

        assert "response_format" in completion_kwargs
        # Verify the winning_schema (top-level output_format) was used,
        # not the losing one nested under output_config.
        rendered = str(completion_kwargs["response_format"])
        assert "top_level" in rendered
        assert "nested" not in rendered

    def test_other_extra_kwargs_still_passed_through(self):
        """Regression guard: the strip must be narrow. Unrelated fields like
        ``api_key`` / ``timeout`` continue to flow through."""
        extra_kwargs = {
            "custom_llm_provider": "azure",
            "output_config": {"effort": "high"},
            "timeout": 30,
            "user": "end-user-123",
        }

        result = _call_prepare(extra_kwargs=extra_kwargs)
        completion_kwargs = result[0] if isinstance(result, tuple) else result

        assert "output_config" not in completion_kwargs
        assert completion_kwargs.get("timeout") == 30
        assert completion_kwargs.get("user") == "end-user-123"


class TestEmptyExtraKwargsPath:
    """Greptile P2 on PR #22727: ``extra_kwargs or {default}`` substitutes a
    default for an explicitly-passed empty dict, hiding the no-extra-kwargs
    path. The new explicit-None pattern lets ``extra_kwargs={}`` reach the
    code under test as written."""

    def test_explicit_empty_dict_does_not_substitute_default(self):
        # Explicit empty dict must be honored — not silently replaced with a
        # default that adds back a custom_llm_provider this test wants absent.
        result = _call_prepare(extra_kwargs={})
        completion_kwargs = result[0] if isinstance(result, tuple) else result

        # No output_config because nothing supplied it.
        assert "output_config" not in completion_kwargs

    def test_none_extra_kwargs_handled_safely(self):
        """The signature documents ``extra_kwargs: Optional[Dict] = None``;
        passing None must not crash with KeyError or AttributeError."""
        result = _call_prepare(extra_kwargs=None)
        # Just exercising the path; assert no exception and we get back a
        # dict-like result.
        completion_kwargs = result[0] if isinstance(result, tuple) else result
        assert isinstance(completion_kwargs, dict)
