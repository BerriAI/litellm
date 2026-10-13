"""Regression test for #46045: the MCP gateway's import chain needs the
proxy-extra `mcp` package and dies both when `mcp` is absent (base installs)
and on mcp 1.x (no CacheableResult); an ordinary tool call must skip the
gateway instead of dying on its import chain (ModuleNotFoundError)."""

import sys
from unittest.mock import patch

from openai import OpenAI

import litellm
from litellm.litellm_core_utils.mcp_availability import mcp_gateway_importable


def test_mcp_gateway_importable_false_when_import_chain_dies():
    # A None entry in sys.modules makes the import machinery raise ImportError,
    # simulating the chain failing (absent or incompatible `mcp` package).
    with patch.dict(sys.modules, {"litellm.responses.mcp.chat_completions_handler": None}):
        assert mcp_gateway_importable() is False


def test_mcp_gateway_importable_true_when_chain_works():
    assert mcp_gateway_importable() is True


def test_completion_skips_mcp_gateway_when_import_chain_dies():
    client = OpenAI(api_key="fake-api-key")

    with (
        patch.dict(sys.modules, {"litellm.responses.mcp.chat_completions_handler": None}),
        patch.object(client.chat.completions.with_raw_response, "create") as mock_client,
    ):
        try:
            litellm.completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}},
                    }
                ],
                client=client,
            )
        except Exception as e:
            print(e)

        mock_client.assert_called_once()
