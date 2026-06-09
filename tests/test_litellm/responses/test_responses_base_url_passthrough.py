"""Regression for #30026 — proxy YAML's ``base_url`` (OpenAI-SDK convention) must reach
the OpenAI client as ``api_base``. Before the fix, ``litellm.responses()`` constructed
``GenericLiteLLMParams(**kwargs)`` without first mapping ``base_url`` to ``api_base``,
so requests went to ``api.openai.com`` regardless of the configured base_url."""

import importlib
from unittest.mock import patch

import pytest

import litellm
from litellm.types.router import GenericLiteLLMParams

# Import the responses package's main module directly; `litellm.responses` is a
# callable in the public namespace, so we can't dotted-import through it.
responses_main = importlib.import_module("litellm.responses.main")


def _patched_params_capture(monkeypatch):
    """Return a (seen-dict, patched-class) pair that records the kwargs handed to
    GenericLiteLLMParams inside responses()."""
    seen = {}

    class _Spy(GenericLiteLLMParams):
        def __init__(self, **data):
            seen.update(data)
            super().__init__(**data)

    monkeypatch.setattr(responses_main, "GenericLiteLLMParams", _Spy)
    return seen


def test_base_url_in_kwargs_is_promoted_to_api_base(monkeypatch):
    seen = _patched_params_capture(monkeypatch)
    # mock_response short-circuits in responses() right after litellm_params is built,
    # so we reach the promotion logic without making an HTTP call.
    litellm.responses(
        input="hello",
        model="openai/gpt-4o-mini",
        base_url="https://eu.api.openai.com/v1",
        mock_response="ok",
    )
    assert seen.get("api_base") == "https://eu.api.openai.com/v1"
    assert "base_url" not in seen


def test_existing_api_base_takes_precedence(monkeypatch):
    seen = _patched_params_capture(monkeypatch)
    litellm.responses(
        input="hi",
        model="openai/gpt-4o-mini",
        base_url="https://wrong.example.com/v1",
        api_base="https://right.example.com/v1",
        mock_response="ok",
    )
    assert seen.get("api_base") == "https://right.example.com/v1"


def test_no_base_url_leaves_api_base_unset(monkeypatch):
    seen = _patched_params_capture(monkeypatch)
    litellm.responses(
        input="hi",
        model="openai/gpt-4o-mini",
        mock_response="ok",
    )
    # Neither key present → no promotion, api_base stays absent from the kwargs.
    assert "api_base" not in seen
    assert "base_url" not in seen
