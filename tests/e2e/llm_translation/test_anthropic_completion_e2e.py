from __future__ import annotations

import os
from typing import Final, Literal

import pytest
from e2e_config import unique_marker
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import AnthropicExtraBody, AnthropicToolBody, LiteLLMParamsBody
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

ANTHROPIC_BACKEND: Final = "anthropic/claude-sonnet-5-5"
COMPUTER_20250124_BACKEND: Final = "anthropic/claude-sonnet-4-5-20250929"
MCP_SERVER_URL: Final = "https://mcp.zapier.com/api/mcp/mcp"


class _ServerToolUse(BaseModel):
    model_config = ConfigDict(frozen=True)

    web_search_requests: int = 0


class _AnthropicUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    server_tool_use: _ServerToolUse | None = None


class _ServerToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str


class _ProviderSpecificFields(BaseModel):
    model_config = ConfigDict(frozen=True)

    web_search_results: tuple[_ServerToolResult, ...] = ()


class _MessageExtras(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_specific_fields: _ProviderSpecificFields | None = None


def _register_anthropic(
    proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, prefix: str, backend: str = ANTHROPIC_BACKEND
) -> tuple[str, OpenAI]:
    model: Final = f"{prefix}-{unique_marker()}"
    model_id: Final = proxy.create_model(
        model,
        LiteLLMParamsBody(model=backend, api_key="os.environ/ANTHROPIC_API_KEY"),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, sdk.openai(resources.key())


class TestAnthropicCompletion:
    @pytest.mark.covers("llm.chat_completions.anthropic.tool_use.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_web_search_reports_server_tool_use(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, client = _register_anthropic(proxy, resources, sdk, "e2e-anthropic-web-search")
        response: Final = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Search the web for a current weather report in Tokyo. {unique_marker()}",
                }
            ],
            extra_body={**NO_PROXY_CACHE, **AnthropicExtraBody(web_search_options={}).model_dump(exclude_none=True)},
        )
        assert response.choices and response.choices[0].message.content
        assert response.usage is not None, f"response omitted usage: {response!r}"
        usage: Final = TypeAdapter(_AnthropicUsage).validate_python(response.usage.model_dump())
        assert usage.server_tool_use is not None, f"Anthropic did not report server tool use: {usage!r}"
        assert usage.server_tool_use.web_search_requests > 0, f"no web search reported: {usage!r}"

    @pytest.mark.covers("llm.chat_completions.anthropic.tool_use.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC,),
            models=(COMPUTER_20250124_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_computer_tool_returns_computer_tool_call(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, client = _register_anthropic(
            proxy, resources, sdk, "e2e-anthropic-computer", backend=COMPUTER_20250124_BACKEND
        )
        response: Final = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Take a screenshot of the screen. {unique_marker()}"}],
            extra_body={
                **NO_PROXY_CACHE,
                **AnthropicExtraBody(
                    tools=(
                        AnthropicToolBody(
                            type="computer_20250124",
                            function={
                                "name": "computer",
                                "parameters": {
                                    "display_height_px": 768,
                                    "display_width_px": 1024,
                                    "display_number": 1,
                                },
                            },
                        ),
                    )
                ).model_dump(exclude_none=True),
            },
        )
        choice: Final = response.choices[0] if response.choices else None
        assert choice is not None and choice.message.tool_calls, f"no computer tool call: {response!r}"
        tool_call: Final = choice.message.tool_calls[0]
        assert tool_call.type == "function", f"not a function tool call: {tool_call!r}"
        assert tool_call.function.name == "computer"

    @pytest.mark.covers("llm.chat_completions.anthropic.tool_use.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_web_fetch_tool_result_is_surfaced(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, client = _register_anthropic(proxy, resources, sdk, "e2e-anthropic-web-fetch")
        response: Final = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Use web_fetch on https://example.com and tell me the page title. {unique_marker()}",
                }
            ],
            extra_body={
                **NO_PROXY_CACHE,
                **AnthropicExtraBody(
                    tools=(AnthropicToolBody(type="web_fetch_20250910", name="web_fetch", max_uses=5),)
                ).model_dump(exclude_none=True),
            },
        )
        assert response.choices, f"no choices: {response!r}"
        extras: Final = _MessageExtras.model_validate(response.choices[0].message.model_extra or {})
        assert extras.provider_specific_fields is not None, f"no provider fields: {response!r}"
        assert "web_fetch_tool_result" in {r.type for r in extras.provider_specific_fields.web_search_results}

    @pytest.mark.covers("llm.chat_completions.anthropic.tool_use.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_text_editor_returns_tool_call(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model_and_client: Final = _register_anthropic(proxy, resources, sdk, "e2e-anthropic-text-editor")
        model: Final = model_and_client[0]
        client: Final = model_and_client[1]
        response: Final = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Use the editor tool to fix the syntax error in primes.py. {unique_marker()}",
                }
            ],
            extra_body={
                **NO_PROXY_CACHE,
                **AnthropicExtraBody(
                    tools=(AnthropicToolBody(type="text_editor_20250728", name="str_replace_based_edit_tool"),)
                ).model_dump(exclude_none=True),
            },
        )
        choice = response.choices[0] if response.choices else None
        assert choice is not None and choice.message.tool_calls, f"no editor tool call: {response!r}"
        tool_call: Final = choice.message.tool_calls[0]
        assert tool_call.type == "function", f"not a function tool call: {tool_call!r}"
        assert tool_call.function.name == "str_replace_based_edit_tool"

    @pytest.mark.covers("llm.chat_completions.anthropic.tool_use.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    @pytest.mark.parametrize("spec", ["anthropic", "openai"])
    def test_mcp_tool_use_by_spec(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        sdk: SdkClients,
        spec: Literal["anthropic", "openai"],
    ) -> None:
        token: Final = os.getenv("ZAPIER_CI_CD_MCP_TOKEN")
        if not token:
            pytest.skip("ZAPIER_CI_CD_MCP_TOKEN is not available; Shared vault has no Zapier MCP credential")
        model_and_client: Final = _register_anthropic(proxy, resources, sdk, "e2e-anthropic-mcp-chat")
        model: Final = model_and_client[0]
        client: Final = model_and_client[1]
        tool: Final[AnthropicToolBody] = (
            AnthropicToolBody(
                type="url",
                url=MCP_SERVER_URL,
                name="zapier-mcp",
                authorization_token=token,
            )
            if spec == "anthropic"
            else AnthropicToolBody(
                type="mcp",
                server_label="zapier",
                server_url=MCP_SERVER_URL,
                require_approval="never",
                headers={"Authorization": f"Bearer {token}"},
            )
        )
        response: Final = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Who won the World Cup in 2022? {unique_marker()}"}],
            extra_body={
                **NO_PROXY_CACHE,
                **AnthropicExtraBody(tools=(tool,)).model_dump(exclude_none=True),
            },
        )
        choice: Final = response.choices[0] if response.choices else None
        assert choice is not None and (choice.message.tool_calls or choice.message.content), response

    @pytest.mark.covers("llm.responses.anthropic.tool_use.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_mcp_tool_use_via_responses(self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients) -> None:
        token = os.getenv("ZAPIER_CI_CD_MCP_TOKEN")
        if not token:
            pytest.skip("ZAPIER_CI_CD_MCP_TOKEN is not available; Shared vault has no Zapier MCP credential")
        model_and_client: Final = _register_anthropic(proxy, resources, sdk, "e2e-anthropic-mcp-responses")
        model: Final = model_and_client[0]
        client: Final = model_and_client[1]
        response: Final = client.responses.create(
            model=model,
            input=f"Find the answer using the MCP server. {unique_marker()}",
            max_output_tokens=100,
            extra_body={
                **NO_PROXY_CACHE,
                "tools": [
                    {
                        "type": "mcp",
                        "server_label": "zapier",
                        "server_url": MCP_SERVER_URL,
                        "require_approval": "never",
                        "headers": {"Authorization": f"Bearer {token}"},
                    }
                ],
            },
        )
        assert response.output, f"/responses returned no MCP result: {response!r}"
