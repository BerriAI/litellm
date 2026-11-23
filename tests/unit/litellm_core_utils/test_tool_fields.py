import datetime
import json
import logging
from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm import Router
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.tool_fields import (
    tool_params_without_hosted_mcp,
    tools_without_hosted_mcp,
    tools_without_mcp_servers,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, async_pre_call_deployment_hook, get_optional_params

_GOOD_TOOL: Final = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
_MALFORMED_TOOLS: Final = (
    5,
    "get_weather",
    ["get_weather"],
    {"type": "function"},
    [{"type": "function", "function": {"name": "get_weather", "parameters": "object"}}],
)


def _deployment(name: str, **params: object) -> Mapping[str, object]:
    return {
        "model_name": name,
        "litellm_params": {"model": "openai/gpt-5.2", "api_key": "sk-test", "mock_response": f"served by {name}", **params},
    }


def _proxy_request(model: str, route: str = "acompletion", **fields: object) -> Mapping[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "proxy_server_request": {"url": "/v1/chat/completions", "body": {}},
        "litellm_logging_obj": Logging(
            model=model,
            messages=[],
            stream=False,
            call_type=route,
            start_time=datetime.datetime.now(),
            litellm_call_id="call-1",
            function_id="fn-1",
        ),
        "num_retries": 0,
        **fields,
    }


@pytest.mark.parametrize("tools", _MALFORMED_TOOLS)
@pytest.mark.asyncio
async def test_malformed_tools_fall_back_to_a_deployment_that_drops_them(tools: object) -> None:
    router: Final = Router(
        model_list=[_deployment("primary"), _deployment("backup", additional_drop_params=["tools"])],
        fallbacks=[{"primary": ["backup"]}],
    )

    response: Final = await router.acompletion(**_proxy_request("primary", tools=tools))

    assert response.choices[0].message.content == "served by backup"


@pytest.mark.asyncio
async def test_fallback_on_unsupported_params_false_keeps_the_400() -> None:
    router: Final = Router(
        model_list=[_deployment("primary"), _deployment("backup", additional_drop_params=["tools"])],
        fallbacks=[{"primary": ["backup"]}],
        fallback_on_unsupported_params=False,
    )

    with pytest.raises(litellm.BadRequestError):
        await router.acompletion(**_proxy_request("primary", tools=5))


@pytest.mark.asyncio
async def test_a_rejected_request_cools_down_no_deployment() -> None:
    cools_on_any_400: Final = {"allowed_fails_policy": {"BadRequestErrorAllowedFails": 0}, "cooldown_time": 120}
    router: Final = Router(
        model_list=[
            {**_deployment("primary"), "model_info": {"id": "primary", **cools_on_any_400}},
            {**_deployment("backup"), "model_info": {"id": "backup", **cools_on_any_400}},
        ],
        fallbacks=[{"primary": ["backup"]}],
    )

    with pytest.raises(litellm.BadRequestError):
        await router.acompletion(**_proxy_request("primary", tools=5))

    primary: Final = await router.acompletion(**_proxy_request("primary"))
    backup: Final = await router.acompletion(**_proxy_request("backup"))
    assert (primary.choices[0].message.content, backup.choices[0].message.content) == (
        "served by primary",
        "served by backup",
    )


@pytest.mark.parametrize("tools", _MALFORMED_TOOLS)
@pytest.mark.parametrize(
    "deployment",
    [_deployment("keeps"), _deployment("keeps", tools=[_GOOD_TOOL])],
    ids=["plain", "with_configured_tools"],
)
@pytest.mark.asyncio
async def test_malformed_tools_answer_400_when_the_served_deployment_keeps_tools(
    tools: object, deployment: Mapping[str, object]
) -> None:
    router: Final = Router(model_list=[deployment])

    with pytest.raises(litellm.BadRequestError) as raised:
        await router.acompletion(**_proxy_request("keeps", tools=tools))

    assert (raised.value.status_code, raised.value.param, raised.value.type) == (
        400,
        "tools",
        "invalid_request_error",
    )


@pytest.mark.asyncio
async def test_a_request_drop_params_false_overrides_a_deployment_that_drops() -> None:
    router: Final = Router(model_list=[_deployment("drops", drop_params=True)])

    with pytest.raises(litellm.BadRequestError):
        await router.acompletion(**_proxy_request("drops", tools=5, drop_params=False))


@pytest.mark.parametrize(
    "route", ["acompletion", "aresponses", "anthropic_messages", "agenerate_content", "acreate_interaction"]
)
@pytest.mark.asyncio
async def test_drop_params_drops_every_malformed_tool_field(route: str) -> None:
    request: Final = _proxy_request(
        "drops", route, drop_params=True, tools=5, tool_choice="auto", web_search_options="high"
    )

    sent: Final = await async_pre_call_deployment_hook(dict(request), route)

    assert {"tools", "tool_choice", "web_search_options"}.isdisjoint(sent)
    assert sent["messages"] == request["messages"]


@pytest.mark.asyncio
async def test_additional_drop_params_drops_only_the_malformed_tool_entries() -> None:
    request: Final = _proxy_request("drops", additional_drop_params=["tools"], tools=[_GOOD_TOOL, "get_weather"])

    sent: Final = await async_pre_call_deployment_hook(dict(request), "acompletion")

    assert sent["tools"] == [_GOOD_TOOL]


@pytest.mark.asyncio
async def test_an_sdk_call_keeps_the_tools_it_passed() -> None:
    tools: Final = (_GOOD_TOOL,)
    request: Final = {
        key: value for key, value in _proxy_request("sdk", tools=tools).items() if key != "proxy_server_request"
    }

    sent: Final = await async_pre_call_deployment_hook(dict(request), "acompletion")

    assert sent["tools"] is tools


@pytest.mark.parametrize("schema", ["", [], False, 0, {}, '{"type": "object"}'])
@pytest.mark.parametrize(
    "tool",
    [
        lambda schema: {"type": "function", "function": {"name": "get_weather", "parameters": schema}},
        lambda schema: {"type": "function", "name": "get_weather", "parameters": schema},
        lambda schema: {"name": "get_weather", "input_schema": schema},
    ],
    ids=["chat", "responses", "messages"],
)
@pytest.mark.asyncio
async def test_empty_and_json_string_schemas_reach_the_deployment(
    tool: Callable[[object], Mapping[str, object]], schema: object
) -> None:
    tools: Final = [tool(schema)]

    sent: Final = await async_pre_call_deployment_hook(dict(_proxy_request("keeps", tools=tools)), "acompletion")

    assert sent["tools"] is tools


@pytest.mark.parametrize("tools", [(_GOOD_TOOL,), [{"type": "web_search", "function": None}]], ids=["tuple", "null"])
@pytest.mark.asyncio
async def test_tools_core_accepts_reach_the_deployment(tools: object) -> None:
    sent: Final = await async_pre_call_deployment_hook(dict(_proxy_request("keeps", tools=tools)), "acompletion")

    assert sent["tools"] is tools


def test_a_hosted_mcp_error_names_every_server_and_marks_an_unlabeled_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    with pytest.raises(litellm.UnsupportedParamsError) as raised:
        tools_without_hosted_mcp(
            tools=[
                {"type": "mcp", "server_label": "deepwiki", "server_url": "https://mcp.deepwiki.com/mcp"},
                {"type": "mcp", "server_url": "https://mcp.example.com/mcp"},
            ],
            model="gemini-3.8-flash",
            route="gemini",
            drop_params=False,
        )
    assert "the hosted MCP tool `deepwiki`, `unlabeled` " in str(raised.value)


_HOSTED_MCP_TOOL: Final = {"type": "mcp", "server_label": "deepwiki", "server_url": "https://mcp.deepwiki.com/mcp"}


def test_dropping_a_hosted_mcp_tool_keeps_the_other_tool_params_and_leaves_the_caller_dict_alone() -> None:
    tool_choice: Final = {"type": "function", "function": {"name": "get_weather"}}
    sent: Final = {
        "tools": [_HOSTED_MCP_TOOL, _GOOD_TOOL],
        "tool_choice": tool_choice,
        "parallel_tool_calls": False,
        "temperature": 0.2,
    }
    params: Final = tool_params_without_hosted_mcp(
        sent, model="deepseek-v4-flash", route="deepseek", drop_params=True, handling="drops"
    )

    assert params == {
        "tools": [_GOOD_TOOL],
        "tool_choice": tool_choice,
        "parallel_tool_calls": False,
        "temperature": 0.2,
    }
    assert sent["tools"] == [_HOSTED_MCP_TOOL, _GOOD_TOOL]


def test_dropping_every_hosted_mcp_tool_clears_the_tool_selection_params() -> None:
    params: Final = tool_params_without_hosted_mcp(
        {"tools": [_HOSTED_MCP_TOOL], "tool_choice": "required", "parallel_tool_calls": True, "temperature": 0.2},
        model="deepseek-v4-flash",
        route="deepseek",
        drop_params=True,
        handling="drops",
    )

    assert params == {"temperature": 0.2}


_FORCED_DEEPWIKI: Final = {"type": "function", "function": {"name": "deepwiki"}}


_LEGACY_FUNCTION_CASES: Final = pytest.mark.parametrize(
    "functions",
    [None, [{"name": "get_weather", "parameters": {"type": "object"}}], ["deepwiki"]],
    ids=["tools_kept", "legacy_functions_kept", "malformed_legacy_function_kept"],
)


@pytest.mark.parametrize(
    "tool_choice",
    [_FORCED_DEEPWIKI, {"type": "mcp", "server_label": "deepwiki"}],
    ids=["function_choice", "responses_mcp_choice"],
)
@_LEGACY_FUNCTION_CASES
def test_dropping_a_hosted_mcp_tool_clears_a_tool_choice_that_forces_it(
    functions: object, tool_choice: Mapping[str, object]
) -> None:
    sent_tools: Final = [_HOSTED_MCP_TOOL, _GOOD_TOOL] if functions is None else [_HOSTED_MCP_TOOL]
    legacy: Final = {} if functions is None else {"functions": functions}
    params: Final = tool_params_without_hosted_mcp(
        {"tools": sent_tools, "tool_choice": tool_choice, "parallel_tool_calls": False, **legacy},
        model="deepseek-v4-flash",
        route="deepseek",
        drop_params=True,
        handling="drops",
    )

    assert "tool_choice" not in params
    assert params == {
        **({"tools": [_GOOD_TOOL]} if functions is None else {}),
        "parallel_tool_calls": False,
        **legacy,
    }


@pytest.mark.parametrize(
    ("tool_choice", "cause"),
    [
        ({"type": "mcp", "server_label": "unsent"}, "This request sends no hosted MCP tool with server_label `unsent`."),
        ({"type": "mcp"}, "This tool_choice names no server_label, so no hosted MCP tool drops with it."),
    ],
    ids=["mcp_choice_for_an_unsent_server", "unlabeled_mcp_choice"],
)
@_LEGACY_FUNCTION_CASES
def test_an_mcp_tool_choice_naming_no_dropped_server_is_refused_like_a_real_time_call(
    functions: object, tool_choice: Mapping[str, object], cause: str
) -> None:
    sent_tools: Final = [_HOSTED_MCP_TOOL, _GOOD_TOOL] if functions is None else [_HOSTED_MCP_TOOL]
    legacy: Final = {} if functions is None else {"functions": functions}

    with pytest.raises(litellm.BadRequestError, match=r"tool_choice type 'mcp' forces an OpenAI hosted tool") as exc:
        tool_params_without_hosted_mcp(
            {"tools": sent_tools, "tool_choice": tool_choice, "parallel_tool_calls": False, **legacy},
            model="deepseek-v4-flash",
            route="deepseek",
            drop_params=True,
            handling="drops",
        )

    assert exc.value.status_code == 400
    assert cause in exc.value.message
    assert "drop_params does not drop it" not in exc.value.message


@pytest.mark.parametrize(
    ("tools", "functions"),
    [
        ([_HOSTED_MCP_TOOL, {"type": "function", "function": {"name": "deepwiki"}}], None),
        ([_HOSTED_MCP_TOOL], [{"name": "deepwiki", "parameters": {"type": "object"}}]),
    ],
    ids=["kept_tool", "legacy_function"],
)
def test_a_tool_choice_naming_a_kept_function_survives_a_dropped_mcp_tool_with_the_same_label(
    tools: list[object], functions: object
) -> None:
    legacy: Final = {} if functions is None else {"functions": functions}
    params: Final = tool_params_without_hosted_mcp(
        {"tools": tools, "tool_choice": _FORCED_DEEPWIKI, **legacy},
        model="deepseek-v4-flash",
        route="deepseek",
        drop_params=True,
        handling="drops",
    )

    assert params["tool_choice"] == _FORCED_DEEPWIKI


def test_a_hosted_mcp_error_takes_the_callers_remedy_in_place_of_the_gateway_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    remedy: Final = "Remove the tool from the row."

    with pytest.raises(litellm.UnsupportedParamsError) as default_error:
        tool_params_without_hosted_mcp(
            {"tools": [_HOSTED_MCP_TOOL]}, model="gemini-3.8-flash", route="gemini", drop_params=False, handling="drops"
        )
    with pytest.raises(litellm.UnsupportedParamsError) as remedied_error:
        tool_params_without_hosted_mcp(
            {"tools": [_HOSTED_MCP_TOOL]},
            model="gemini-3.8-flash",
            route="gemini",
            drop_params=False,
            handling="drops",
            remedy=remedy,
        )

    assert "litellm_proxy/mcp/<server name>" in default_error.value.message
    assert "drop_params" in default_error.value.message
    assert remedied_error.value.message.endswith(remedy)
    assert "litellm_proxy/mcp" not in remedied_error.value.message
    assert remedied_error.value.message.removesuffix(remedy) in default_error.value.message


_SWEPT_ROUTES: Final = (
    *((provider, "compress" if provider is LlmProviders.SCALEDOWN else "some-model") for provider in LlmProviders),
    (LlmProviders.VERTEX_AI, "claude-sonnet-5"),
    (LlmProviders.VERTEX_AI, "gemini-3.8-flash"),
    (LlmProviders.BEDROCK, "invoke/anthropic.claude-sonnet-5"),
    (LlmProviders.BEDROCK, "claude_platform/claude-sonnet-5"),
    (LlmProviders.BEDROCK, "converse/anthropic.claude-sonnet-5"),
    (LlmProviders.AZURE_AI, "claude-sonnet-5"),
    (LlmProviders.DATABRICKS, "databricks-claude-sonnet-5"),
)
_CONFIGURED_ROUTES: Final = tuple(
    (provider, model, config)
    for provider, model in _SWEPT_ROUTES
    if (config := ProviderConfigManager.get_provider_chat_config(model=model, provider=provider)) is not None
)
_DROPPING_ROUTES: Final = tuple(
    (provider, model) for provider, model, config in _CONFIGURED_ROUTES if config.hosted_mcp_tool_handling(model) == "drops"
)


def _sent_tool_params(provider: LlmProviders, model: str, **params: object) -> str:
    return json.dumps(
        get_optional_params(
            model=model,
            custom_llm_provider=provider.value,
            tools=[_HOSTED_MCP_TOOL, _GOOD_TOOL],
            drop_params=True,
            messages=[{"role": "user", "content": "weather in Paris?"}],
            **params,
        ),
        default=str,
    )


@pytest.mark.parametrize(
    ("provider", "model"),
    [(provider, model) for provider, model, _ in _CONFIGURED_ROUTES],
    ids=[f"{provider.value}-{model}" for provider, model, _ in _CONFIGURED_ROUTES],
)
def test_a_route_reports_dropping_hosted_mcp_tools_exactly_when_its_transformation_drops_them(
    provider: LlmProviders, model: str
) -> None:
    config: Final = ProviderConfigManager.get_provider_chat_config(model=model, provider=provider)
    assert config is not None
    sent: Final = _sent_tool_params(provider, model)

    assert ("deepwiki" not in sent and "get_weather" in sent) == (config.hosted_mcp_tool_handling(model) == "drops")


@pytest.mark.parametrize(
    ("provider", "model"),
    _DROPPING_ROUTES,
    ids=[f"{provider.value}-{model}" for provider, model in _DROPPING_ROUTES],
)
def test_a_route_that_drops_hosted_mcp_tools_drops_a_forced_mcp_choice_with_its_tool(
    provider: LlmProviders, model: str
) -> None:
    sent: Final = _sent_tool_params(provider, model, tool_choice={"type": "mcp", "server_label": "deepwiki"})

    assert "deepwiki" not in sent
    assert "get_weather" in sent


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: Final[list[str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.mark.parametrize(
    "drop",
    [
        lambda model: tools_without_hosted_mcp(
            [{"type": "mcp", "server_label": "wiki\r\nFORGED", "server_url": "https://mcp.example.com/mcp"}],
            model=model,
            route="deepseek",
            drop_params=True,
        ),
        lambda model: tools_without_mcp_servers(
            [{"type": "url", "name": "wiki\r\nFORGED", "url": "https://mcp.example.com/mcp"}],
            tools=[],
            model=model,
            route="bedrock",
            drop_params=True,
        ),
    ],
    ids=["hosted_mcp_tool", "mcp_servers"],
)
def test_a_drop_warning_stays_on_one_log_line(drop: Callable[[str], object]) -> None:
    records: Final = _Records()
    verbose_logger.addHandler(records)
    try:
        drop("deepseek-v4-flash\nFORGED level=CRITICAL")
    finally:
        verbose_logger.removeHandler(records)

    assert len(records.messages) == 1
    assert "\n" not in records.messages[0] and "\r" not in records.messages[0]
    assert "FORGED level=CRITICAL" in records.messages[0]


_ALLOWED_DEEPWIKI: Final = {"type": "mcp", "server_label": "deepwiki"}
_ALLOWED_WEATHER: Final = {
    "responses": {"type": "function", "name": "get_weather"},
    "chat": {"type": "function", "function": {"name": "get_weather"}},
}


def _allowed_tools_choice(layout: str, entries: list[Mapping[str, object]]) -> Mapping[str, object]:
    if layout == "responses":
        return {"type": "allowed_tools", "mode": "required", "tools": entries}
    return {"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": entries}}


@pytest.mark.parametrize("layout", ["responses", "chat"])
def test_dropping_a_hosted_mcp_tool_removes_it_from_an_allowed_tools_choice(layout: str) -> None:
    weather: Final = _ALLOWED_WEATHER[layout]
    params: Final = tool_params_without_hosted_mcp(
        {
            "tools": [_HOSTED_MCP_TOOL, _GOOD_TOOL],
            "tool_choice": _allowed_tools_choice(layout, [_ALLOWED_DEEPWIKI, weather]),
            "parallel_tool_calls": False,
        },
        model="deepseek-v4-flash",
        route="deepseek",
        drop_params=True,
        handling="drops",
    )

    assert params == {
        "tools": [_GOOD_TOOL],
        "tool_choice": _allowed_tools_choice(layout, [weather]),
        "parallel_tool_calls": False,
    }


@pytest.mark.parametrize("layout", ["responses", "chat"])
def test_an_allowed_tools_choice_naming_only_dropped_mcp_tools_drops_the_choice_and_keeps_the_function_tools(
    layout: str,
) -> None:
    params: Final = tool_params_without_hosted_mcp(
        {
            "tools": [_HOSTED_MCP_TOOL, _GOOD_TOOL],
            "tool_choice": _allowed_tools_choice(layout, [_ALLOWED_DEEPWIKI]),
            "parallel_tool_calls": False,
            "temperature": 0.2,
        },
        model="deepseek-v4-flash",
        route="deepseek",
        drop_params=True,
        handling="drops",
    )

    assert params == {"tools": [_GOOD_TOOL], "parallel_tool_calls": False, "temperature": 0.2}


@pytest.mark.parametrize("layout", ["responses", "chat"])
def test_an_allowed_tools_choice_naming_only_dropped_mcp_tools_leaves_no_tool_params_when_no_tool_is_left(
    layout: str,
) -> None:
    params: Final = tool_params_without_hosted_mcp(
        {
            "tools": [_HOSTED_MCP_TOOL],
            "tool_choice": _allowed_tools_choice(layout, [_ALLOWED_DEEPWIKI]),
            "parallel_tool_calls": False,
            "temperature": 0.2,
        },
        model="deepseek-v4-flash",
        route="deepseek",
        drop_params=True,
        handling="drops",
    )

    assert params == {"temperature": 0.2}


@pytest.mark.parametrize("layout", ["responses", "chat"])
def test_an_allowed_tools_choice_without_mcp_entries_survives_a_dropped_mcp_tool(layout: str) -> None:
    tool_choice: Final = _allowed_tools_choice(layout, [_ALLOWED_WEATHER[layout]])
    params: Final = tool_params_without_hosted_mcp(
        {"tools": [_HOSTED_MCP_TOOL, _GOOD_TOOL], "tool_choice": tool_choice},
        model="deepseek-v4-flash",
        route="deepseek",
        drop_params=True,
        handling="drops",
    )

    assert params == {"tools": [_GOOD_TOOL], "tool_choice": tool_choice}
