"""Unit tests for litellm.main._resolve_completion_timeout (completion() timeout chain)."""

import os
import sys

import httpx

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
)

import litellm
from litellm.main import _resolve_completion_timeout


def test_explicit_timeout_wins():
    assert (
        _resolve_completion_timeout(
            timeout=12.5,
            kwargs={"timeout": 99.0, "request_timeout": 88.0},
            custom_llm_provider="openai",
        )
        == 12.5
    )


def test_kwargs_timeout_when_param_none():
    assert (
        _resolve_completion_timeout(
            timeout=None,
            kwargs={"timeout": 21.0},
            custom_llm_provider="azure_ai",
        )
        == 21.0
    )


def test_request_timeout_alias_in_kwargs():
    assert (
        _resolve_completion_timeout(
            timeout=None,
            kwargs={"request_timeout": 33.0},
            custom_llm_provider="bedrock",
        )
        == 33.0
    )


def test_litellm_module_request_timeout(monkeypatch):
    monkeypatch.setattr(litellm, "request_timeout", 360.0)
    assert (
        _resolve_completion_timeout(
            timeout=None,
            kwargs={},
            custom_llm_provider="vertex_ai",
        )
        == 360.0
    )


def test_fallback_600_when_no_timeout_anywhere(monkeypatch):
    """600 applies only when named, kwargs, and litellm.request_timeout are all unset."""
    monkeypatch.setattr(litellm, "request_timeout", None)
    assert (
        _resolve_completion_timeout(
            timeout=None,
            kwargs={},
            custom_llm_provider="azure_ai",
        )
        == 600.0
    )


def test_httpx_timeout_coerced_for_provider_without_httpx_timeout_support():
    t = httpx.Timeout(50.0, connect=2.0)
    out = _resolve_completion_timeout(
        timeout=t,
        kwargs={},
        custom_llm_provider="azure_ai",
    )
    assert out == 50.0
    assert not isinstance(out, httpx.Timeout)


def test_httpx_timeout_preserved_for_openai():
    t = httpx.Timeout(40.0, connect=5.0)
    out = _resolve_completion_timeout(
        timeout=t,
        kwargs={},
        custom_llm_provider="openai",
    )
    assert out is t
    assert isinstance(out, httpx.Timeout)
