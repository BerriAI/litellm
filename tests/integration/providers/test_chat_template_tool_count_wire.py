import json
import re
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
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
from integration._support.deepseek_tokens import (
    CHINESE_TEXT,
    SHORT_TEXT,
    assert_counted_with_deepseek,
    chinese_minus_short,
    cl100k_tokens,
)
from integration._support.openai_wire import chat_reply
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from litellm.litellm_core_utils.default_encoding import encoding as cl100k_base
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_API_KEY: Final = "scripted-provider-key"
_DEEPSEEK_V4: Final = "azure_ai/DeepSeek-V4-Flash-0731"
_DEEPSEEK_R1: Final = "azure_ai/DeepSeek-R1-0528"
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
_PROMPT: Final = SHORT_TEXT
_MESSAGES: Final[list[JsonValue]] = [{"role": "user", "content": _PROMPT}]
_INPUT_PRICE: Final = 0.001
_OUTPUT_PRICE: Final = 0.002
_TINY_BUDGET: Final = 0.000000001
_ESTIMATED_COST: Final = re.compile(r"Estimated request cost: ([^\s,]+)")


def _tokens(text: str) -> int:
    return len(cl100k_base.encode(text))


def _deepseek_v4_tool_tokens(function: Mapping[str, JsonValue], tokens: Callable[[str], int]) -> int:
    return _DEEPSEEK_V4_TOOL_PREAMBLE + tokens(json.dumps(function, ensure_ascii=False))


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

    def body(self, model: str, tools: Sequence[JsonValue], text: str = _PROMPT) -> dict[str, JsonValue]:
        prompt: Final[dict[str, JsonValue]] = (
            {"input": text}
            if self.path == "/v1/responses/input_tokens"
            else {"messages": [{"role": "user", "content": text}]}
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


_TOOL_COUNTS: Final[Mapping[str, tuple[str, str]]] = {
    **{f"{route}-azure_ai-v4": (route, _DEEPSEEK_V4) for route in _COUNT_ROUTES},
    "messages-count-tokens-deepseek-chat": ("messages-count-tokens", "deepseek/deepseek-chat"),
    "messages-count-tokens-deepseek-reasoner": ("messages-count-tokens", "deepseek/deepseek-reasoner"),
    "messages-count-tokens-openrouter-deepseek-chat": ("messages-count-tokens", "openrouter/deepseek/deepseek-chat"),
}


@pytest.mark.parametrize("case", sorted(_TOOL_COUNTS))
def test_count_routes_count_deepseek_v4_tools_the_way_its_chat_template_renders_them(
    gateway: Gateway, case: str
) -> None:
    route, backend = _TOOL_COUNTS[case]
    selected: Final = _COUNT_ROUTES[route]
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "count", backend, wire.url)
        without_tools: Final = _count(gateway, selected, selected.body(model, ()))
        with_tool: Final = _count(gateway, selected, selected.body(model, (selected.tool,)))
        upstream: Final = wire.drain()
    assert upstream == ()
    tool_tokens: Final = with_tool - without_tools
    assert tool_tokens > _DEEPSEEK_V4_TOOL_PREAMBLE, f"counted {tool_tokens} tool tokens, short of V4's tool preamble"
    assert_counted_with_deepseek(tool_tokens, lambda tokens: _deepseek_v4_tool_tokens(_FUNCTION, tokens))


_TEXT_COUNTS: Final[Mapping[str, tuple[str, str]]] = {
    "messages-count-tokens-azure_ai-v4": ("messages-count-tokens", _DEEPSEEK_V4),
    "responses-input-tokens-azure_ai-v4": ("responses-input-tokens", _DEEPSEEK_V4),
    "messages-count-tokens-deepseek-chat": ("messages-count-tokens", "deepseek/deepseek-chat"),
}


def _text_count_delta(gateway: Gateway, route: _CountRoute, model: str) -> int:
    chinese: Final = _count(gateway, route, route.body(model, (), CHINESE_TEXT))
    short: Final = _count(gateway, route, route.body(model, (), SHORT_TEXT))
    return chinese - short


@pytest.mark.parametrize("case", sorted(_TEXT_COUNTS))
def test_count_routes_count_deepseek_v3_and_v4_text_with_deepseeks_tokenizer(gateway: Gateway, case: str) -> None:
    route, backend = _TEXT_COUNTS[case]
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        delta: Final = _text_count_delta(
            gateway, _COUNT_ROUTES[route], _deployment(scenario, "count", backend, wire.url)
        )
        upstream: Final = wire.drain()
    assert upstream == ()
    assert_counted_with_deepseek(delta, chinese_minus_short)


def _counted_with_deepseek(delta: int) -> None:
    assert_counted_with_deepseek(delta, chinese_minus_short)


def _counted_with_cl100k(delta: int) -> None:
    assert delta == chinese_minus_short(cl100k_tokens)


@pytest.mark.parametrize(
    ("backend", "tokenizer_type", "counted_with"),
    (
        (_DEEPSEEK_V4, "huggingface_tokenizer", _counted_with_deepseek),
        (_DEEPSEEK_R1, "openai_tokenizer", _counted_with_cl100k),
    ),
    ids=("deepseek-v4", "deepseek-r1"),
)
def test_utils_token_counter_names_the_tokenizer_that_counted_deepseek_text(
    gateway: Gateway, backend: str, tokenizer_type: str, counted_with: Callable[[int], None]
) -> None:
    route: Final = _COUNT_ROUTES["utils-token-counter"]
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "count", backend, wire.url)
        delta: Final = _text_count_delta(gateway, route, model)
        response: Final = gateway.request("POST", route.path, route.body(model, (), CHINESE_TEXT))
        upstream: Final = wire.drain()
    assert upstream == ()
    assert response.status_code == 200, response.text
    assert JSON_OBJECT.validate_json(response.content)["tokenizer_type"] == tokenizer_type, response.text
    counted_with(delta)


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
    assert_counted_with_deepseek(
        deferred - plain,
        lambda tokens: (
            _deepseek_v4_tool_tokens(deferred_function, tokens) - _deepseek_v4_tool_tokens(_FUNCTION, tokens)
        ),
    )


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
class _OwnedRig:
    gateway: Gateway
    wire: Wire
    group: str
    itpm_wire: Wire
    itpm_group: str


def _owned_config(
    directory: Path,
    model_list: Sequence[JsonValue],
    general_settings: Mapping[str, JsonValue],
    router_settings: Mapping[str, JsonValue],
) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "model_list": list(model_list),
        "general_settings": {**object_value(base["general_settings"]), **general_settings},
        "router_settings": {**object_value(base["router_settings"]), **router_settings},
    }
    path: Final = directory / f"chat-template-tool-count-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _owned_rig_config(directory: Path, group: str, api_base: str, itpm_group: str, itpm_api_base: str) -> Path:
    return _owned_config(
        directory,
        [
            {
                "model_name": group,
                "litellm_params": {
                    "model": _DEEPSEEK_V4,
                    "api_base": api_base,
                    "api_key": _API_KEY,
                    "input_cost_per_token": _INPUT_PRICE,
                    "output_cost_per_token": _OUTPUT_PRICE,
                },
            },
            {
                "model_name": itpm_group,
                "litellm_params": {
                    "model": _DEEPSEEK_V4,
                    "api_base": itpm_api_base,
                    "api_key": _API_KEY,
                    "itpm": cl100k_tokens(CHINESE_TEXT),
                },
                "model_info": {"id": uuid.uuid4().hex},
            },
        ],
        {"fail_closed_budget_enforcement": True},
        {"optional_pre_call_checks": ["enforce_model_rate_limits"]},
    )


@pytest.fixture(scope="module")
def owned_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_OwnedRig]:
    directory: Final = tmp_path_factory.mktemp("chat-template-tool-count")
    group: Final = f"deepseek-v4-flash-{uuid.uuid4().hex}"
    itpm_group: Final = f"itpm-{uuid.uuid4().hex}"
    with ExitStack() as stack:
        rig_gateway: Final = stack.enter_context(gateway_from_environment())
        wire: Final = stack.enter_context(wire_server(_completion))
        itpm_wire: Final = stack.enter_context(wire_server(_completion))
        config: Final = _owned_rig_config(directory, group, wire.url, itpm_group, itpm_wire.url)
        owned: Final = stack.enter_context(owned_proxy_process(rig_gateway, directory, {}, config=config))
        yield _OwnedRig(owned.gateway, wire, group, itpm_wire, itpm_group)


def _estimated_cost(response: httpx.Response) -> float:
    assert response.status_code in (400, 422, 429), f"{response.status_code} {response.text}"
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    estimate: Final = _ESTIMATED_COST.search(string_value(error["message"]))
    assert estimate is not None, response.text
    return float(estimate[1])


@pytest.mark.parametrize("endpoint", ("chat-completions", "messages"))
def test_budget_reservation_estimates_deepseek_v4_tools_the_way_its_chat_template_renders_them(
    owned_rig: _OwnedRig, endpoint: str
) -> None:
    selected: Final = _ENDPOINTS[endpoint]
    with owned_rig.gateway.scenario() as scenario:
        key: Final = scenario.key(models=[owned_rig.group], max_budget=_TINY_BUDGET)
        without_tools: Final = _estimated_cost(_send(owned_rig.gateway, selected, owned_rig.group, key, tools=False))
        with_tools: Final = _estimated_cost(_send(owned_rig.gateway, selected, owned_rig.group, key, tools=True))
    assert owned_rig.wire.drain() == ()
    assert_counted_with_deepseek(
        round((with_tools - without_tools) / _INPUT_PRICE), lambda tokens: _deepseek_v4_tool_tokens(_FUNCTION, tokens)
    )


def _chinese_chat(group: str) -> dict[str, JsonValue]:
    return {"model": group, "messages": [{"role": "user", "content": CHINESE_TEXT}], "max_tokens": 1}


def test_router_itpm_reserves_deepseek_v4_text_with_deepseeks_tokenizer(owned_rig: _OwnedRig) -> None:
    response: Final = owned_rig.gateway.request("POST", "/v1/chat/completions", _chinese_chat(owned_rig.itpm_group))
    upstream: Final = owned_rig.itpm_wire.drain()
    assert response.status_code == 200, response.text
    assert len(upstream) == 1, upstream


def _project_key(scenario: Scenario, group: str, itpm_limit: int) -> str:
    team: Final = scenario.team()
    project: Final = scenario.project(team, model_itpm_limit={group: itpm_limit})
    return scenario.key(team_id=team, project_id=project)


_TOOL_FRAMING_ITPM_LIMIT: Final = _deepseek_v4_tool_tokens(_FUNCTION, _tokens)


@pytest.mark.parametrize("endpoint", sorted(_ENDPOINTS))
def test_project_itpm_reserves_deepseek_v4_tools_the_way_its_chat_template_renders_them(
    gateway: Gateway, endpoint: str
) -> None:
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        group: Final = _deployment(scenario, "deepseek-v4-flash", _DEEPSEEK_V4, wire.url)
        key: Final = _project_key(scenario, group, _TOOL_FRAMING_ITPM_LIMIT)
        response: Final = _send(gateway, _ENDPOINTS[endpoint], group, key, tools=True)
        upstream: Final = wire.drain()
    assert response.status_code == 429, response.text
    assert "model_per_project_itpm" in response.text, response.text
    assert upstream == ()


def test_project_itpm_keeps_openai_tool_framing_for_other_models(gateway: Gateway) -> None:
    chat: Final = _ENDPOINTS["chat-completions"]
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        group: Final = _deployment(scenario, "grok-4-1", _GROK, wire.url)
        response: Final = gateway.request(
            "POST",
            chat.path,
            {"model": group, **chat.body, "tools": [_OPENAI_TOOL]},
            key=_project_key(scenario, group, _TOOL_FRAMING_ITPM_LIMIT),
        )
        upstream: Final = wire.drain()
    assert response.status_code == 200, response.text
    assert len(upstream) == 1, upstream


def test_project_itpm_reserves_deepseek_v4_text_with_deepseeks_tokenizer(gateway: Gateway) -> None:
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        group: Final = _deployment(scenario, "deepseek-v4-flash", _DEEPSEEK_V4, wire.url)
        key: Final = _project_key(scenario, group, cl100k_tokens(CHINESE_TEXT))
        response: Final = gateway.request("POST", "/v1/chat/completions", _chinese_chat(group), key=key)
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


def _read_file_round(marker: str, index: int, file_body: str) -> tuple[JsonValue, ...]:
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
                    "content": f"<file {marker}-{index}>{file_body}",
                }
            ],
        },
    )


def _agent_body(group: str, marker: str, file_body: str, trigger: int) -> dict[str, JsonValue]:
    rounds: Final = tuple(chain.from_iterable(_read_file_round(marker, index, file_body) for index in range(_FILES)))
    return {
        "model": group,
        "max_tokens": 16,
        "tools": [_READ_FILE_TOOL],
        "messages": [*rounds, {"role": "user", "content": "Summarize what you read."}],
        "context_management": {
            "edits": [
                {
                    "type": "clear_tool_uses_20250919",
                    "trigger": {"type": "input_tokens", "value": trigger},
                    "keep": {"type": "tool_uses", "value": 1},
                }
            ]
        },
    }


@pytest.mark.parametrize(
    ("model", "files_kept"),
    ((_DEEPSEEK_V4, (False, False, True)), (_GROK, (True, True, True))),
    ids=("deepseek-v4", "grok-4-1"),
)
def test_messages_context_management_trigger_counts_tools_the_way_the_models_chat_template_renders_them(
    gateway: Gateway, model: str, files_kept: tuple[bool, ...]
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        group: Final = _deployment(scenario, "context", model, wire.url)
        trigger: Final = _deepseek_v4_tool_tokens(_READ_FILE_FUNCTION, _tokens)
        response: Final = gateway.request(
            "POST", "/v1/messages", _agent_body(group, marker, "short file body.", trigger)
        )
        upstream: Final = wire.drain()
    assert response.status_code == 200, response.text
    assert len(upstream) == 1, upstream
    sent: Final = upstream[0].body.decode()
    assert tuple(f"<file {marker}-{index}>" in sent for index in range(_FILES)) == files_kept


def test_messages_context_management_trigger_counts_deepseek_v4_text_with_deepseeks_tokenizer(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        group: Final = _deployment(scenario, "context", _DEEPSEEK_V4, wire.url)
        trigger: Final = _FILES * cl100k_tokens(CHINESE_TEXT)
        response: Final = gateway.request("POST", "/v1/messages", _agent_body(group, marker, CHINESE_TEXT, trigger))
        upstream: Final = wire.drain()
    assert response.status_code == 200, response.text
    assert len(upstream) == 1, upstream
    sent: Final = upstream[0].body.decode()
    kept: Final = tuple(f"<file {marker}-{index}>" in sent for index in range(_FILES))
    assert kept == (True,) * _FILES, f"cleared tool results under a trigger only the cl100k_base count reaches: {kept}"


_BURST: Final = (CHINESE_TEXT, SHORT_TEXT) * 8


def test_concurrent_first_deepseek_v4_counts_in_a_fresh_proxy_agree(tmp_path: Path) -> None:
    group: Final = f"count-{uuid.uuid4().hex}"
    with ExitStack() as stack:
        rig_gateway: Final = stack.enter_context(gateway_from_environment())
        wire: Final = stack.enter_context(wire_server(_completion))
        config: Final = _owned_config(
            tmp_path,
            [
                {
                    "model_name": group,
                    "litellm_params": {"model": _DEEPSEEK_V4, "api_base": wire.url, "api_key": _API_KEY},
                }
            ],
            {},
            {},
        )
        owned: Final = stack.enter_context(owned_proxy_process(rig_gateway, tmp_path, {}, config=config))
        with ThreadPoolExecutor(max_workers=len(_BURST)) as pool:
            counts: Final = tuple(
                pool.map(
                    lambda text: _count(owned.gateway, _MESSAGES_COUNT, _MESSAGES_COUNT.body(group, (), text)), _BURST
                )
            )
        upstream: Final = wire.drain()
    assert upstream == ()
    chinese: Final = frozenset(counts[0::2])
    short: Final = frozenset(counts[1::2])
    assert len(chinese) == 1 and len(short) == 1, counts
    assert_counted_with_deepseek(min(chinese) - min(short), chinese_minus_short)
