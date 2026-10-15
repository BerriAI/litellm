import uuid
from typing import Final

import pytest
from integration._support import prompt_cache_breakpoint as pcb
from integration._support import responses_vendor as rv
from integration._support.client import Gateway
from integration._support.wire import wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(120)

_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
_TOOL_SEARCH: Final[dict[str, JsonValue]] = {"type": "tool_search"}


def _function(name: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"name": name, "description": f"Look up {name}", "parameters": _PARAMETERS, **extra}


def _wire_function(name: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"type": "function", "strict": None, **_function(name), **extra}


@pytest.mark.parametrize(
    "deferred_tool",
    [
        {"type": "function", "function": _function("get_weather"), "defer_loading": True},
        {"type": "function", "function": _function("get_weather", defer_loading=True)},
    ],
    ids=["tool-level", "function-level"],
)
def test_chat_bridge_forwards_defer_loading_so_tool_search_has_a_deferred_tool(
    gateway: Gateway, deferred_tool: dict[str, JsonValue]
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(pcb.respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": pcb.prompt(marker)}],
                "tools": [
                    _TOOL_SEARCH,
                    deferred_tool,
                    {"type": "function", "function": _function("get_time")},
                    {"type": "function", "function": _function("get_news", defer_loading=False)},
                ],
                **pcb.NO_CACHE,
            },
        )
        assert response.status_code == 200, response.text
        message: Final = rv.JSON_OBJECT.validate_python(
            rv.ITEMS.validate_python(rv.JSON_OBJECT.validate_json(response.text)["choices"])[0]["message"]
        )
        assert message["content"] == rv.answer(marker), response.text
        assert pcb.body_of(pcb.posted(wire, marker))["tools"] == [
            _TOOL_SEARCH,
            _wire_function("get_weather", defer_loading=True),
            _wire_function("get_time"),
            _wire_function("get_news", defer_loading=False),
        ]
