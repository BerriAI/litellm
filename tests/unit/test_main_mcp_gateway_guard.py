"""Regression test for #46045: the MCP gateway's import chain needs the
proxy-extra `mcp` package and dies both when `mcp` is absent (base installs)
and on mcp 1.x (no CacheableResult); an ordinary tool call must skip the
gateway instead of dying on its import chain (ModuleNotFoundError)."""

import sys
from unittest.mock import patch

import pytest

import litellm
from litellm.litellm_core_utils.mcp_availability import (
    mcp_gateway_importable,
    reset_mcp_gateway_importable_cache,
)


@pytest.fixture(autouse=True)
def fresh_importable_cache():
    reset_mcp_gateway_importable_cache()
    yield
    reset_mcp_gateway_importable_cache()


def test_mcp_gateway_importable_false_when_import_chain_dies():
    # A None entry in sys.modules makes the import machinery raise ImportError,
    # simulating the chain failing (absent or incompatible `mcp` package).
    with patch.dict(sys.modules, {"litellm.responses.mcp.chat_completions_handler": None}):
        assert mcp_gateway_importable() is False


def test_mcp_gateway_importable_true_when_chain_works():
    assert mcp_gateway_importable() is True


def test_mcp_gateway_importable_caches_the_verdict():
    # A failed import leaves nothing in sys.modules, so an uncached probe would
    # rerun the failing chain on every tool call; the cached verdict must win
    # even once the chain would import again in the same process.
    with patch.dict(sys.modules, {"litellm.responses.mcp.chat_completions_handler": None}):
        assert mcp_gateway_importable() is False
    assert mcp_gateway_importable() is False
    reset_mcp_gateway_importable_cache()
    assert mcp_gateway_importable() is True


def test_completion_skips_mcp_gateway_when_import_chain_dies():
    # litellm's mock_response short-circuits the provider call after the gateway
    # check, so the call never touches the network; if the gateway were NOT
    # skipped, the dead import chain would raise ImportError inside completion()
    # before the request ever reached the provider path.
    with patch.dict(sys.modules, {"litellm.responses.mcp.chat_completions_handler": None}):
        response = litellm.completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}},
                }
            ],
            mock_response="Hello!",
        )
    assert isinstance(response, litellm.ModelResponse)
