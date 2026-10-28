import datetime
from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm import Router
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.utils import async_pre_call_deployment_hook

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
