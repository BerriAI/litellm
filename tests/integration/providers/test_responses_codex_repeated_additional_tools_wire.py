import json
import uuid
from collections.abc import Mapping
from typing import Final

import httpx
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

BEDROCK_GPT: Final = "us.openai.gpt-5.6-sol"
MANTLE_MODEL: Final = "bedrock_mantle/openai.gpt-5.6-sol"
TOKEN: Final = "synthetic-bedrock-bearer"
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
TOOL_SPECS: Final = TypeAdapter(tuple[dict[str, dict[str, JsonValue]], ...])


def function_tool(name: str, description: str = "") -> dict[str, JsonValue]:
    return {
        "type": "function",
        "name": name,
        "description": description or f"run {name}",
        "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}},
    }


def namespace_tool(name: str, *members: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"type": "namespace", "name": name, "description": "", "tools": list(members)}


def tool_list_item(*tools: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"type": "additional_tools", "role": "developer", "tools": list(tools)}


def user_turn(text: str) -> dict[str, JsonValue]:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def assistant_turn(text: str) -> dict[str, JsonValue]:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def converse_tool_specs(request: Request) -> tuple[dict[str, JsonValue], ...]:
    tool_config: Final = JSON_OBJECT.validate_json(request.body).get("toolConfig")
    assert isinstance(tool_config, dict), request.body
    return tuple(entry["toolSpec"] for entry in TOOL_SPECS.validate_python(tool_config["tools"]))


def bedrock_converse_peer(request: Request) -> Reply:
    names: Final = tuple(str(spec["name"]) for spec in converse_tool_specs(request))
    repeated: Final = tuple(name for index, name in enumerate(names) if name in names[:index])
    if repeated:
        return Reply(
            status=400,
            body=json.dumps(
                {
                    "message": f"The tool {repeated[0]} is already defined at toolConfig.tools.{names.index(repeated[0])}."
                }
            ).encode(),
        )
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "converse wire control"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 30, "outputTokens": 3, "totalTokens": 33},
                "metrics": {"latencyMs": 10},
            }
        ).encode()
    )


def send_to_bedrock_converse(gateway: Gateway, history: list[JsonValue]) -> tuple[httpx.Response, tuple[Request, ...]]:
    with wire_server(bedrock_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/converse/{BEDROCK_GPT}",
            api_key=TOKEN,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
        )
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": history, "store": False, "cache": {"no-cache": True}}
        )
        return response, wire.drain()


def forwarded_specs_by_name(forwarded: tuple[Request, ...]) -> Mapping[str, dict[str, JsonValue]]:
    assert len(forwarded) == 1, forwarded
    return {str(spec["name"]): spec for spec in converse_tool_specs(forwarded[0])}


def test_codex_tool_list_repeated_on_every_turn_reaches_bedrock_converse_once(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    tools: Final = tool_list_item(
        namespace_tool("collaboration", function_tool("followup_task")), function_tool("exec_command")
    )
    history: Final[list[JsonValue]] = [
        tools,
        user_turn(f"first {marker}"),
        assistant_turn("ok"),
        tools,
        user_turn(f"second {marker}"),
        assistant_turn("ok"),
        tools,
        user_turn(f"third {marker}"),
    ]

    response, forwarded = send_to_bedrock_converse(gateway, history)

    assert response.status_code == 200, response.text
    assert response.json()["output"][0]["content"][0]["text"] == "converse wire control", response.text
    names: Final = [str(spec["name"]) for spec in converse_tool_specs(forwarded[0])]
    assert sorted(names) == ["collaboration__followup_task", "exec_command"], names


def test_newest_copy_of_a_repeated_codex_tool_is_what_bedrock_converse_receives(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    history: Final[list[JsonValue]] = [
        tool_list_item(function_tool("exec_command", "old description")),
        user_turn(f"first {marker}"),
        assistant_turn("ok"),
        tool_list_item(function_tool("exec_command", "new description")),
        user_turn(f"second {marker}"),
    ]

    response, forwarded = send_to_bedrock_converse(gateway, history)

    assert response.status_code == 200, response.text
    specs: Final = forwarded_specs_by_name(forwarded)
    assert list(specs) == ["exec_command"], specs
    assert specs["exec_command"]["description"] == "new description", specs


def test_distinct_codex_tools_spread_across_turns_all_reach_bedrock_converse(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    history: Final[list[JsonValue]] = [
        tool_list_item(function_tool("exec_command"), namespace_tool("mcp__docs__", function_tool("search"))),
        user_turn(f"first {marker}"),
        assistant_turn("ok"),
        tool_list_item(function_tool("apply_patch"), namespace_tool("mcp__docs__", function_tool("fetch"))),
        user_turn(f"second {marker}"),
    ]

    response, forwarded = send_to_bedrock_converse(gateway, history)

    assert response.status_code == 200, response.text
    assert sorted(forwarded_specs_by_name(forwarded)) == [
        "apply_patch",
        "exec_command",
        "mcp__docs____fetch",
        "mcp__docs____search",
    ]


def test_codex_tool_list_repeated_on_every_turn_reaches_bedrock_mantle_once(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    shell: Final = function_tool("shell")
    collaboration: Final = namespace_tool("collaboration", function_tool("followup_task"))

    def mantle_peer(request: Request) -> Reply:
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_synthetic_mantle",
                    "object": "response",
                    "created_at": 1789788253,
                    "status": "completed",
                    "model": "openai.gpt-5.6-sol",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_synthetic_mantle",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "mantle wire control", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 21, "output_tokens": 4, "total_tokens": 25},
                }
            ).encode()
        )

    with wire_server(mantle_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=MANTLE_MODEL, api_key=TOKEN, api_base=wire.url, aws_region_name="us-east-2")
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": [
                    tool_list_item(shell, collaboration),
                    user_turn(f"first {marker}"),
                    assistant_turn("ok"),
                    tool_list_item(shell, collaboration),
                    user_turn(f"second {marker}"),
                ],
                "store": False,
            },
        )
        forwarded: Final = wire.drain()

    assert response.status_code == 200, response.text
    assert len(forwarded) == 1, forwarded
    assert JSON_OBJECT.validate_json(forwarded[0].body)["tools"] == [shell, collaboration], forwarded[0].body
