"""Live e2e: the Anthropic web_search server tool over Bedrock Invoke.

Bedrock hosts none of Anthropic's ``web_search_*`` server tools, so a
``/v1/messages`` request carrying one is rejected outright with
400 "The provided request is not valid" if it reaches AWS unchanged. What makes
it work is web-search interception: the hooks rewrite the native tool into
LiteLLM's own search tool before the upstream call, Bedrock calls that tool, the
gateway runs the search, and the agentic loop feeds the results back for the
model to synthesize. The response is then rebuilt in the native shape, so a
client's citations panel sees ``server_tool_use`` and ``web_search_tool_result``
exactly as it would from Anthropic direct.

This cell pins that whole path. Nothing else covers it: the ``web_search`` cells
in the Claude Code compat matrix drive the CLI's *client-side* ``WebSearch``
tool, an ordinary custom tool the CLI executes and feeds back as a
``tool_result``, and the CLI never emits a ``web_search_20250305`` definition.

Prerequisites beyond AWS credentials: the proxy config must switch interception
on and declare a search backend. The callback entry is load-bearing; the params
block alone does not activate it.

    litellm_settings:
      callbacks: ["websearch_interception"]
      websearch_interception_params:
        enabled_providers: ["bedrock"]
        search_tool_name: e2e-search
    search_tools:
      - search_tool_name: e2e-search
        litellm_params:
          search_provider: searxng
          api_base: http://127.0.0.1:8391
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import unwrap
from endpoints_client import EndpointsClient
from lifecycle import ResourceManager
from models import (
    AnthropicMessagesBody,
    AnthropicWebSearchTool,
    ChatMessage,
    LiteLLMParamsBody,
)

pytestmark = pytest.mark.e2e

BEDROCK_INVOKE_BACKEND = "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0"

WEB_SEARCH_TOOL = AnthropicWebSearchTool(
    type="web_search_20250305",
    name="web_search",
    max_uses=3,
)

SEARCH_PROMPT = "Use web search to tell me one recent news headline about Anthropic."


class TestBedrockWebSearchServerTool:
    @pytest.mark.skip(
        reason="stage red: environment gap, the e2e stack neither enables the "
        "websearch_interception callback nor declares a search backend, so the request "
        "reaches bedrock's transformation and takes its by-design 400. Unskip once the "
        "ephemeral stack ships the config in this module's docstring."
    )
    @pytest.mark.covers("llm.messages.bedrock_invoke.web_search_server_tool.nonstream.works")
    def test_web_search_server_tool_is_served(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        """A bedrock deployment must answer a web_search server-tool request
        instead of handing the tool to AWS and returning its 400."""
        model = f"e2e-bedrock-websearch-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model=BEDROCK_INVOKE_BACKEND,
                aws_region_name="us-east-1",
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            endpoints_client.proxy.messages(
                key,
                AnthropicMessagesBody(
                    model=model,
                    max_tokens=512,
                    tools=[WEB_SEARCH_TOOL],
                    messages=[ChatMessage(role="user", content=SEARCH_PROMPT)],
                ),
            )
        )

        assert response.content, f"no content blocks in response: {response}"
        block_types = [block.type for block in response.content]
        assert "web_search_tool_result" in block_types, (
            "the answer carries no web_search_tool_result block, so the search "
            "either never ran or its results were not returned in the native shape "
            f"a citations panel reads. blocks={block_types}"
        )
        assert "text" in block_types, (
            "the model never synthesized an answer over the search results, so the "
            f"agentic loop stopped early. blocks={block_types}"
        )
