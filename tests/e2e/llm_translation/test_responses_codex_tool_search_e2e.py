"""Live e2e: Codex's client tool_search over /v1/responses on Claude.

Codex holds MCP tools back behind its client-executed `tool_search` tool when the model catalog says
`supports_search_tool`. Over the completion bridge the proxy offers that search to Claude as a plain function,
hands Claude's call back as a `tool_search_call` Codex runs itself, and loads the tools Codex returns in
`tool_search_output` so Claude can call them on the next turn
"""

from __future__ import annotations

from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from openai.types.responses import (
    FunctionToolParam,
    Response,
    ResponseFunctionToolCall,
    ResponseInputParam,
    ResponseToolSearchCall,
    ToolParam,
    ToolSearchToolParam,
)
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

ANTHROPIC_BACKEND: Final = "anthropic/claude-haiku-5-5"
DEFERRED_TOOL_COUNT: Final = 50
CALENDAR_TOOL_NAME: Final = "mcp__codex_apps__calendar_create_event"

CODEX_TOOL_SEARCH: Final[ToolSearchToolParam] = {
    "type": "tool_search",
    "execution": "client",
    "description": (
        "# Tool discovery\n\nSearches over deferred tool metadata with BM25 and exposes matching tools for the next "
        "model call. Some of the tools may not have been provided to you upfront, and you should use this tool "
        "(`tool_search`) to search for the required tools."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query for deferred tools."},
            "limit": {"type": "number", "description": "Maximum number of tools to return. Defaults to 8."},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

SHELL_TOOL: Final[FunctionToolParam] = {
    "type": "function",
    "name": "shell",
    "description": "Runs a shell command.",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
    "strict": False,
}

CALENDAR_TOOL: Final[FunctionToolParam] = {
    "type": "function",
    "name": CALENDAR_TOOL_NAME,
    "description": "Create a calendar event.",
    "parameters": {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
        "additionalProperties": False,
    },
    "strict": False,
    "defer_loading": True,
}

PROMPT: Final = "Create a calendar event titled Standup. Find the calendar tool with tool_search first, then call it."


class _CalendarArguments(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str


def _filler_tool(index: int) -> FunctionToolParam:
    return {
        "type": "function",
        "name": f"mcp__codex_apps__records_{index:02d}",
        "description": f"List, inspect, update and archive records of family {index} in the connected workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title of the record to act on"},
                "notes": {"type": "string", "description": "Free-form notes and context for the operation"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "priority": {"type": "integer", "minimum": 0, "maximum": 5},
            },
            "required": ["title"],
            "additionalProperties": False,
        },
        "strict": False,
        "defer_loading": True,
    }


DEFERRED_TOOLS: Final[tuple[FunctionToolParam, ...]] = (
    *(_filler_tool(index) for index in range(DEFERRED_TOOL_COUNT - 1)),
    CALENDAR_TOOL,
)


def _eager(tool: FunctionToolParam) -> ToolParam:
    return FunctionToolParam(
        type="function",
        name=tool["name"],
        description=tool.get("description"),
        parameters=tool.get("parameters"),
        strict=False,
    )


def _input_tokens(response: Response) -> int:
    assert response.usage is not None, f"response carried no usage: {response.id}"
    return response.usage.input_tokens


def _search_call(response: Response) -> ResponseToolSearchCall:
    calls: Final = tuple(item for item in response.output if isinstance(item, ResponseToolSearchCall))
    assert len(calls) == 1, f"expected one tool_search_call, got {response.output!r}"
    return calls[0]


class TestResponsesCodexToolSearchOverAnthropic:
    @pytest.mark.covers("llm.responses.anthropic.tool_search.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.TOOL_SEARCH, Capability.FUNCTION_CALLING),
            mode=Mode.NONSTREAM,
        )
    )
    def test_claude_finds_a_deferred_tool_through_codex_tool_search_and_calls_it(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = f"e2e-codex-tool-search-{unique_marker()}"
        model_id: Final = proxy.create_model(
            model, LiteLLMParamsBody(model=ANTHROPIC_BACKEND, api_key="os.environ/ANTHROPIC_API_KEY")
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        client: Final = sdk.openai(resources.key())
        searchable_tools: Final[list[ToolParam]] = [SHELL_TOOL, CODEX_TOOL_SEARCH]

        eager: Final = client.responses.create(
            model=model,
            input=PROMPT,
            tools=[SHELL_TOOL, *(_eager(tool) for tool in DEFERRED_TOOLS)],
            extra_body=NO_PROXY_CACHE,
        )
        first: Final = client.responses.create(
            model=model, input=PROMPT, tools=searchable_tools, extra_body=NO_PROXY_CACHE
        )

        search: Final = _search_call(first)
        assert search.execution == "client", f"tool_search ran on the wrong side: {search!r}"
        assert search.call_id, f"tool_search_call had no call_id: {search!r}"
        assert not any(isinstance(item, ResponseFunctionToolCall) for item in first.output), first.output
        assert _input_tokens(first) * 4 < _input_tokens(eager), (
            f"deferred first turn used {_input_tokens(first)} input tokens against {_input_tokens(eager)} eager"
        )

        follow_up: Final[ResponseInputParam] = [
            {"role": "user", "content": PROMPT},
            {
                "type": "tool_search_call",
                "call_id": search.call_id,
                "execution": "client",
                "arguments": search.arguments,
            },
            {
                "type": "tool_search_output",
                "call_id": search.call_id,
                "status": "completed",
                "execution": "client",
                "tools": [CALENDAR_TOOL],
            },
        ]
        second: Final = client.responses.create(
            model=model, input=follow_up, tools=searchable_tools, extra_body=NO_PROXY_CACHE
        )

        calendar_call: Final = next(
            (
                item
                for item in second.output
                if isinstance(item, ResponseFunctionToolCall) and item.name == CALENDAR_TOOL_NAME
            ),
            None,
        )
        assert calendar_call is not None, f"Claude did not call the loaded tool: {second.output!r}"
        assert "standup" in _CalendarArguments.model_validate_json(calendar_call.arguments).title.lower()
