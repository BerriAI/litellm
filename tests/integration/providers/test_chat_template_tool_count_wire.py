import json
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.openai_wire import chat_reply
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from litellm.litellm_core_utils.default_encoding import encoding as cl100k_base
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_API_KEY: Final = "scripted-provider-key"
_DEEPSEEK_V4: Final = "azure_ai/DeepSeek-V4-Flash-0731"
_QWEN3_5: Final = "databricks/databricks-qwen35-122b-a10b"
_GROK: Final = "azure_ai/grok-4-1"
_DEEPSEEK_V4_TOOL_PREAMBLE: Final = 222
_QWEN3_5_TOOL_PREAMBLE: Final = 196
_QWEN3_5_OPENED_SYSTEM_TURN: Final = 5
_OPENAI_TOOL_OVERHEAD: Final = 9
_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}
_FUNCTION: Final[dict[str, JsonValue]] = {"name": "lookup", "parameters": _SCHEMA}
_OPENAI_NAMESPACE: Final = "namespace functions {\n\ntype lookup = (_: {\nquery: string,\n}) => any;\n\n} // namespace functions"
_ANTHROPIC_TOOL: Final[dict[str, JsonValue]] = {"name": "lookup", "input_schema": _SCHEMA}
_OPENAI_TOOL: Final[dict[str, JsonValue]] = {"type": "function", "function": _FUNCTION}
_RESPONSES_TOOL: Final[dict[str, JsonValue]] = {"type": "function", **_FUNCTION}
_NAMED_TOOL_CHOICE: Final[dict[str, JsonValue]] = {"type": "function", "function": {"name": "lookup"}}
_WEB_SEARCH_TOOL: Final[dict[str, JsonValue]] = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
_PROMPT: Final = "Find record 14."
_MESSAGES: Final[list[JsonValue]] = [{"role": "user", "content": _PROMPT}]
_INPUT_PRICE: Final = 0.001
_OUTPUT_PRICE: Final = 0.002
_TINY_BUDGET: Final = 0.000000001
_ESTIMATED_COST: Final = re.compile(r"Estimated request cost: ([^\s,]+)")


def _tokens(text: str) -> int:
    return len(cl100k_base.encode(text))


def _deepseek_v4_tool_tokens(function: Mapping[str, JsonValue]) -> int:
    return _DEEPSEEK_V4_TOOL_PREAMBLE + _tokens(json.dumps(function, ensure_ascii=False))


def _qwen3_5_tool_tokens(function: Mapping[str, JsonValue], *, with_system: bool) -> int:
    opened_system_turn: Final = 0 if with_system else _QWEN3_5_OPENED_SYSTEM_TURN
    schema: Final = json.dumps({"type": "function", "function": function}, ensure_ascii=False)
    return _QWEN3_5_TOOL_PREAMBLE + opened_system_turn + _tokens("\n" + schema)


def _completion(_request: Request) -> Reply:
    return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", "scripted-model", "OK", stream=False)


def _deployment(scenario: Scenario, group: str, model: str, api_base: str) -> str:
    name: Final = f"{group}-{uuid.uuid4().hex}"
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"model": model, "api_base": api_base, "api_key": _API_KEY},
            "model_info": {},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return name


@dataclass(frozen=True, slots=True)
class _CountRoute:
    path: str
    field: str
    tool: dict[str, JsonValue]

    def body(self, model: str, tools: Sequence[JsonValue]) -> dict[str, JsonValue]:
        prompt: Final[dict[str, JsonValue]] = (
            {"input": _PROMPT} if self.path == "/v1/responses/input_tokens" else {"messages": _MESSAGES}
        )
        return {"model": model, **prompt, **({"tools": list(tools)} if tools else {})}


_COUNT_ROUTES: Final[Mapping[str, _CountRoute]] = {
    "messages-count-tokens": _CountRoute("/v1/messages/count_tokens", "input_tokens", _ANTHROPIC_TOOL),
    "utils-token-counter": _CountRoute("/utils/token_counter", "total_tokens", _OPENAI_TOOL),
    "responses-input-tokens": _CountRoute("/v1/responses/input_tokens", "input_tokens", _RESPONSES_TOOL),
}
_MESSAGES_COUNT: Final = _COUNT_ROUTES["messages-count-tokens"]


def _count(gateway: Gateway, route: _CountRoute, body: Mapping[str, JsonValue]) -> int:
    response: Final = gateway.request("POST", route.path, body)
    assert response.status_code == 200, response.text
    counted: Final = JSON_OBJECT.validate_json(response.content)[route.field]
    assert isinstance(counted, int), response.text
    return counted


@pytest.mark.parametrize("route", sorted(_COUNT_ROUTES))
def test_count_routes_count_deepseek_v4_tools_the_way_its_chat_template_renders_them(
    gateway: Gateway, route: str
) -> None:
    selected: Final = _COUNT_ROUTES[route]
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "count", _DEEPSEEK_V4, wire.url)
        without_tools: Final = _count(gateway, selected, selected.body(model, ()))
        with_tool: Final = _count(gateway, selected, selected.body(model, (selected.tool,)))
        upstream: Final = wire.drain()
    assert upstream == ()
    assert with_tool - without_tools == _deepseek_v4_tool_tokens(_FUNCTION)


@pytest.mark.parametrize("with_system", (False, True), ids=("no-system", "system"))
def test_messages_count_tokens_counts_qwen3_5_tools_the_way_its_chat_template_renders_them(
    gateway: Gateway, with_system: bool
) -> None:
    system: Final[dict[str, JsonValue]] = {"system": "Be brief."} if with_system else {}
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "count", _QWEN3_5, wire.url)
        without_tools: Final = _count(gateway, _MESSAGES_COUNT, {**_MESSAGES_COUNT.body(model, ()), **system})
        with_tool: Final = _count(
            gateway, _MESSAGES_COUNT, {**_MESSAGES_COUNT.body(model, (_ANTHROPIC_TOOL,)), **system}
        )
        upstream: Final = wire.drain()
    assert upstream == ()
    assert with_tool - without_tools == _qwen3_5_tool_tokens(_FUNCTION, with_system=with_system)


def test_messages_count_tokens_skips_web_search_and_counts_unmapped_tool_keys_for_deepseek_v4(
    gateway: Gateway,
) -> None:
    deferred_function: Final[dict[str, JsonValue]] = {"name": "lookup", "parameters": {**_SCHEMA, "defer_loading": True}}
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "count", _DEEPSEEK_V4, wire.url)
        plain: Final = _count(gateway, _MESSAGES_COUNT, _MESSAGES_COUNT.body(model, (_ANTHROPIC_TOOL,)))
        with_web_search: Final = _count(
            gateway, _MESSAGES_COUNT, _MESSAGES_COUNT.body(model, (_ANTHROPIC_TOOL, _WEB_SEARCH_TOOL))
        )
        deferred: Final = _count(
            gateway, _MESSAGES_COUNT, _MESSAGES_COUNT.body(model, ({**_ANTHROPIC_TOOL, "defer_loading": True},))
        )
        upstream: Final = wire.drain()
    assert upstream == ()
    assert with_web_search == plain
    assert deferred - plain == _deepseek_v4_tool_tokens(deferred_function) - _deepseek_v4_tool_tokens(_FUNCTION)


def test_messages_count_tokens_keeps_openai_tool_framing_for_other_models(gateway: Gateway) -> None:
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "count", _GROK, wire.url)
        without_tools: Final = _count(gateway, _MESSAGES_COUNT, _MESSAGES_COUNT.body(model, ()))
        plain: Final = _count(gateway, _MESSAGES_COUNT, _MESSAGES_COUNT.body(model, (_ANTHROPIC_TOOL,)))
        deferred: Final = _count(
            gateway, _MESSAGES_COUNT, _MESSAGES_COUNT.body(model, ({**_ANTHROPIC_TOOL, "defer_loading": True},))
        )
        upstream: Final = wire.drain()
    assert upstream == ()
    assert plain - without_tools == _OPENAI_TOOL_OVERHEAD + _tokens(_OPENAI_NAMESPACE)
    assert deferred == plain


@dataclass(frozen=True, slots=True)
class _Endpoint:
    path: str
    body: dict[str, JsonValue]
    tools: dict[str, JsonValue]


_ENDPOINTS: Final[Mapping[str, _Endpoint]] = {
    "chat-completions": _Endpoint(
        "/v1/chat/completions",
        {"messages": _MESSAGES, "max_tokens": 1},
        {"tools": [_OPENAI_TOOL], "tool_choice": _NAMED_TOOL_CHOICE},
    ),
    "messages": _Endpoint(
        "/v1/messages",
        {"messages": _MESSAGES, "max_tokens": 1},
        {"tools": [_ANTHROPIC_TOOL], "tool_choice": {"type": "auto"}},
    ),
    "responses": _Endpoint(
        "/v1/responses",
        {"input": _PROMPT, "max_output_tokens": 16},
        {"tools": [_RESPONSES_TOOL]},
    ),
}


def _send(gateway: Gateway, endpoint: _Endpoint, group: str, key: str, *, tools: bool) -> httpx.Response:
    return gateway.request(
        "POST",
        endpoint.path,
        {"model": group, **endpoint.body, **(endpoint.tools if tools else {})},
        key=key,
    )


@dataclass(frozen=True, slots=True)
class _BudgetRig:
    gateway: Gateway
    wire: Wire
    group: str
    grok_group: str


def _priced_deployment(group: str, model: str, api_base: str) -> dict[str, JsonValue]:
    return {
        "model_name": group,
        "litellm_params": {
            "model": model,
            "api_base": api_base,
            "api_key": _API_KEY,
            "input_cost_per_token": _INPUT_PRICE,
            "output_cost_per_token": _OUTPUT_PRICE,
        },
    }


def _strict_budget_config(directory: Path, groups: Mapping[str, str], api_base: str) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "model_list": [_priced_deployment(group, model, api_base) for group, model in groups.items()],
        "general_settings": {**object_value(base["general_settings"]), "fail_closed_budget_enforcement": True},
    }
    path: Final = directory / f"chat-template-tool-budget-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def strict_budget(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_BudgetRig]:
    directory: Final = tmp_path_factory.mktemp("chat-template-tool-budget")
    group: Final = f"deepseek-v4-flash-{uuid.uuid4().hex}"
    grok_group: Final = f"grok-4-1-{uuid.uuid4().hex}"
    with ExitStack() as stack:
        rig_gateway: Final = stack.enter_context(gateway_from_environment())
        wire: Final = stack.enter_context(wire_server(_completion))
        config: Final = _strict_budget_config(directory, {group: _DEEPSEEK_V4, grok_group: _GROK}, wire.url)
        owned: Final = stack.enter_context(owned_proxy_process(rig_gateway, directory, {}, config=config))
        yield _BudgetRig(owned.gateway, wire, group, grok_group)


def _estimated_cost(response: httpx.Response) -> float:
    assert response.status_code in (400, 422, 429), f"{response.status_code} {response.text}"
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    estimate: Final = _ESTIMATED_COST.search(string_value(error["message"]))
    assert estimate is not None, response.text
    return float(estimate[1])


@pytest.mark.parametrize("endpoint", ("chat-completions", "messages"))
def test_budget_reservation_counts_deepseek_v4_tools_like_models_without_a_chat_template(
    strict_budget: _BudgetRig, endpoint: str
) -> None:
    selected: Final = _ENDPOINTS[endpoint]
    with strict_budget.gateway.scenario() as scenario:
        key: Final = scenario.key(models=[strict_budget.group, strict_budget.grok_group], max_budget=_TINY_BUDGET)
        deepseek: Final = _estimated_cost(_send(strict_budget.gateway, selected, strict_budget.group, key, tools=True))
        grok: Final = _estimated_cost(_send(strict_budget.gateway, selected, strict_budget.grok_group, key, tools=True))
    assert strict_budget.wire.drain() == ()
    assert deepseek == grok


def _project_key(scenario: Scenario, group: str) -> str:
    team: Final = scenario.team()
    project: Final = scenario.project(team, model_itpm_limit={group: _deepseek_v4_tool_tokens(_FUNCTION)})
    return scenario.key(team_id=team, project_id=project)


@pytest.mark.parametrize(("alias", "model"), (("deepseek-v4-flash", _DEEPSEEK_V4), ("grok-4-1", _GROK)))
@pytest.mark.parametrize("endpoint", sorted(_ENDPOINTS))
def test_project_itpm_admits_tools_under_a_limit_only_chat_template_framing_exceeds(
    gateway: Gateway, endpoint: str, alias: str, model: str
) -> None:
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        group: Final = _deployment(scenario, alias, model, wire.url)
        response: Final = _send(gateway, _ENDPOINTS[endpoint], group, _project_key(scenario, group), tools=True)
        upstream: Final = wire.drain()
    assert response.status_code == 200, response.text
    assert len(upstream) == 1, upstream


_FILES: Final = 3
_READ_FILE_FUNCTION: Final[dict[str, JsonValue]] = {
    "name": "read_file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
}
_READ_FILE_TOOL: Final[dict[str, JsonValue]] = {
    "name": "read_file",
    "input_schema": _READ_FILE_FUNCTION["parameters"],
}


def _read_file_round(marker: str, index: int) -> tuple[JsonValue, ...]:
    return (
        {"role": "user", "content": f"Read file {index}."},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": f"toolu_{index}", "name": "read_file", "input": {"path": f"{index}.txt"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": f"toolu_{index}",
                    "content": f"<file {marker}-{index}>short file body.",
                }
            ],
        },
    )


def _agent_body(group: str, marker: str) -> dict[str, JsonValue]:
    rounds: Final = tuple(chain.from_iterable(_read_file_round(marker, index) for index in range(_FILES)))
    return {
        "model": group,
        "max_tokens": 16,
        "tools": [_READ_FILE_TOOL],
        "messages": [*rounds, {"role": "user", "content": "Summarize what you read."}],
        "context_management": {
            "edits": [
                {
                    "type": "clear_tool_uses_20250919",
                    "trigger": {"type": "input_tokens", "value": _deepseek_v4_tool_tokens(_READ_FILE_FUNCTION)},
                    "keep": {"type": "tool_uses", "value": 1},
                }
            ]
        },
    }


@pytest.mark.parametrize("model", (_DEEPSEEK_V4, _GROK), ids=("deepseek-v4", "grok-4-1"))
def test_messages_context_management_trigger_keeps_openai_tool_framing(gateway: Gateway, model: str) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        group: Final = _deployment(scenario, "context", model, wire.url)
        response: Final = gateway.request("POST", "/v1/messages", _agent_body(group, marker))
        upstream: Final = wire.drain()
    assert response.status_code == 200, response.text
    assert len(upstream) == 1, upstream
    sent: Final = upstream[0].body.decode()
    assert tuple(f"<file {marker}-{index}>" in sent for index in range(_FILES)) == (True,) * _FILES
