"""Unit tests for litellm.litellm_core_utils.completion_timeout.CompletionTimeout."""

import os
import sys
from typing import Final, Literal

import httpx
import pytest
from openai import Timeout as SDKTimeout

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from litellm.litellm_core_utils.completion_timeout import CompletionTimeout
from litellm.utils import supports_httpx_timeout


def test_explicit_timeout_wins():
    assert (
        CompletionTimeout.resolve(
            12.5,
            {"timeout": 99.0, "request_timeout": 88.0},
            "openai",
            global_timeout=None,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 12.5
    )


def test_kwargs_timeout_when_param_none():
    assert (
        CompletionTimeout.resolve(
            None,
            {"timeout": 21.0},
            "azure_ai",
            global_timeout=None,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 21.0
    )


def test_request_timeout_alias_in_kwargs():
    assert (
        CompletionTimeout.resolve(
            None,
            {"request_timeout": 33.0},
            "bedrock",
            global_timeout=None,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 33.0
    )


def test_global_timeout_from_litellm_settings():
    assert (
        CompletionTimeout.resolve(
            None,
            {},
            "vertex_ai",
            global_timeout=360.0,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 360.0
    )


def test_explicit_global_timeout_6000_is_preserved():
    """The caller passes the explicitly-configured value (or None); an explicit
    6000 must be honored, not silently coerced to 600."""
    assert (
        CompletionTimeout.resolve(
            None,
            {},
            "openai",
            global_timeout=6000.0,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 6000.0
    )


def test_explicit_request_timeout_6000_preserved():
    """Explicit deployment/request timeout must not be truncated by the package sentinel."""
    assert (
        CompletionTimeout.resolve(
            None,
            {"request_timeout": 6000.0},
            "openai",
            global_timeout=None,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 6000.0
    )


def test_explicit_model_timeout_6000_preserved():
    assert (
        CompletionTimeout.resolve(
            6000.0,
            {"timeout": 1.0, "request_timeout": 2.0},
            "openai",
            global_timeout=None,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 6000.0
    )


def test_fallback_600_when_no_global_timeout():
    assert (
        CompletionTimeout.resolve(
            None,
            {},
            "azure_ai",
            global_timeout=None,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == 600.0
    )


def test_httpx_timeout_coerced_for_provider_without_httpx_timeout_support():
    t = httpx.Timeout(50.0, connect=2.0)
    out = CompletionTimeout.resolve(
        t,
        {},
        "azure_ai",
        global_timeout=None,
        supports_httpx_timeout=supports_httpx_timeout,
    )
    assert out == 50.0
    assert not isinstance(out, httpx.Timeout)


def test_httpx_timeout_preserved_for_openai():
    t = httpx.Timeout(40.0, connect=5.0)
    out = CompletionTimeout.resolve(
        t,
        {},
        "openai",
        global_timeout=None,
        supports_httpx_timeout=supports_httpx_timeout,
    )
    assert out is t
    assert isinstance(out, httpx.Timeout)


@pytest.mark.parametrize("source", ("model", "timeout", "request_timeout"))
def test_sdk_timeout_preserves_components_for_httpx_providers(
    source: Literal["model", "timeout", "request_timeout"],
) -> None:
    timeout: Final = SDKTimeout(connect=2.0, read=None, write=5.0, pool=7.0)
    resolved: Final = CompletionTimeout.resolve(
        timeout if source == "model" else None,
        {} if source == "model" else {source: timeout},
        "openai",
        global_timeout=None,
        supports_httpx_timeout=supports_httpx_timeout,
    )
    assert isinstance(resolved, httpx.Timeout)
    assert resolved.as_dict() == timeout.as_dict()


@pytest.mark.parametrize("read_timeout, expected", ((23.0, 23.0), (None, 600.0)))
def test_sdk_timeout_coerces_read_timeout_for_providers_without_httpx_support(
    read_timeout: float | None, expected: float
) -> None:
    assert (
        CompletionTimeout.resolve(
            SDKTimeout(connect=2.0, read=read_timeout, write=5.0, pool=7.0),
            {},
            "azure_ai",
            global_timeout=None,
            supports_httpx_timeout=supports_httpx_timeout,
        )
        == expected
    )
