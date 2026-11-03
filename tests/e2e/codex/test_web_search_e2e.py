from __future__ import annotations

from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import ModelSetup, with_expected_failures

pytestmark = pytest.mark.e2e

LIVE_WEB_SEARCH: Final = ("--config", 'web_search="live"')
CODE_MODE_GPT_GAP: Final = (
    "Codex sends no web search tool for this code-mode GPT model through a custom provider,"
    " though it searches when it talks to OpenAI directly."
)
IDENTITY_MAY_NOT_SEARCH: Final = "web search is not authorized for this identity"
SETUPS_WITHOUT_WEB_SEARCH: Final = with_expected_failures(
    {
        "openai": CODE_MODE_GPT_GAP,
        "azure_openai": CODE_MODE_GPT_GAP,
        "bedrock_runtime": (
            "bedrock-runtime runs no server-side tools, so LiteLLM drops Codex's web search tool before the request"
            " goes out."
        ),
        "claude_bedrock": (
            "LiteLLM's Responses to chat bridge drops web search for Bedrock Converse, which supports it only on Nova."
        ),
        "vertex_ai": "LiteLLM drops Gemini's search tool from any request that also carries function tools.",
    }
)


@pytest.mark.parametrize("deployment", SETUPS_WITHOUT_WEB_SEARCH)
class TestCodexWebSearch:
    def test_searches_the_web_through_the_gateway(self, codex: CodexHarness, setup: ModelSetup) -> None:
        run: Final = codex.exec_with_flags(
            "Use your web search tool, not the shell, to find the newest release of the litellm package on PyPI."
            " Reply with its version number only.",
            model=setup.model,
            cwd=codex.new_workspace(files={}),
            flags=LIVE_WEB_SEARCH,
        )
        if any(IDENTITY_MAY_NOT_SEARCH in error.message for error in run.errors):
            pytest.skip("The gateway's AWS identity lacks the bedrock-websearch:InvokeSearch permission.")
        assert run.exit_code == 0, run
        assert run.web_searches > 0, run
        assert run.answer.strip(), run
