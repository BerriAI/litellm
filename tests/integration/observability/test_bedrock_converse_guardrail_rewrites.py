from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    gateway_from_environment,
    held,
    list_value,
    object_value,
)
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_MODEL_ID: Final = "anthropic.claude-sonnet-5-v1:0"
_MODEL: Final = "converse-guarded"
_PIPELINED_ACTIONS: Final = {
    "converse-pipelined-error-allow": {"on_fail": "block", "on_error": "allow"},
    "converse-pipelined-fail-next": {"on_fail": "next", "on_error": "block"},
    "converse-pipelined-fail-allow": {"on_fail": "allow", "on_error": "block"},
    "converse-pipelined-allow-all": {"on_fail": "allow", "on_error": "allow"},
}
_CARD: Final = "4111 1111 1111 1111"
_CARD_PATTERN: Final = re.compile(r"\b(?:\d{4} ){3}\d{4}\b")
_MASK: Final = "****"
_GUARD_PATH: Final = "/beta/litellm_basic_guardrail_api"
_PNG: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
_JSON: Final = TypeAdapter(JsonValue)
_UNAPPLIABLE: Final = "rewrote the request in a way this endpoint cannot apply"
_ADDED_TOOL: Final = {"type": "function", "function": {"name": "retrieve", "parameters": {"type": "object"}}}
_TOOL_CONFIG: Final = {
    "tools": [
        {
            "toolSpec": {
                "name": "lookup_card",
                "description": f"Looks up cards like {_CARD}",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"card": {"type": "string", "description": f"for example {_CARD}"}},
                    }
                },
            }
        },
        {"cachePoint": {"type": "default"}},
        {
            "toolSpec": {
                "name": "refund",
                "description": "Refunds an order",
                "inputSchema": {"json": {"type": "object"}},
            }
        },
    ]
}


class _Vendor(str, Enum):
    MASKS_ROWS_ONLY = "masks-rows-only"
    MASKS_ROWS_AND_BLANKS_EMPTY_CONTENT = "masks-rows-and-blanks-empty-content"
    JOINS_MASKED_ROWS = "joins-masked-rows"
    ECHOES_ROWS = "echoes-rows"
    RESPACES_UNMASKED_ROWS = "respaces-unmasked-rows"
    COMPACTS_TOOL_RESULTS = "compacts-tool-results"
    DROPS_A_ROW = "drops-a-row"
    MASKS_TEXTS_AND_ROW_CONTENT = "masks-texts-and-row-content"
    CLEARS_CARD_TEXTS = "clears-card-texts"
    MASKS_TEXTS_ONLY = "masks-texts-only"
    BLOCKS = "blocks"
    MASKS_TOOLS = "masks-tools"
    DROPS_A_TOOL = "drops-a-tool"
    ADDS_A_TOOL = "adds-a-tool"
    RENAMES_A_TOOL = "renames-a-tool"
    ECHOES_TOOLS_AS_NOT_STRICT = "echoes-tools-as-not-strict"
    JUDGES_LATEST_USER_MESSAGE = "judges-latest-user-message"


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    vendor: Wire
    bedrock: Wire
    keys: Mapping[str, str]


def _masked(value: JsonValue) -> JsonValue:
    return _JSON.validate_json(_CARD_PATTERN.sub(_MASK, json.dumps(value)))


def _flattened(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    content: Final = row.get("content")
    if not isinstance(content, list):
        return dict(row)
    parts: Final = tuple(object_value(part).get("text") for part in content)
    return {**row, "content": "\n".join(part for part in parts if isinstance(part, str))}


def _normalised_first_part(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    content: Final = row.get("content")
    if not isinstance(content, list) or not content:
        return dict(row)
    first: Final = object_value(content[0])
    text: Final = first.get("text")
    if not isinstance(text, str):
        return dict(row)
    return {**row, "content": [{**first, "text": text.strip() + " "}, *content[1:]]}


def _blank_content(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {**row, "content": ""} if row.get("content") is None else dict(row)


def _card_cleared(value: JsonValue) -> JsonValue:
    match value:
        case str() if _CARD_PATTERN.search(value):
            return ""
        case list():
            return [_card_cleared(item) for item in value]
        case dict():
            return {key: _card_cleared(item) for key, item in value.items()}
        case _:
            return value


def _compacted(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {**row, "content": f"compacted: {row.get('content')}"} if row.get("role") == "tool" else dict(row)


def _content_masked(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {**row, "content": _masked(row.get("content"))}


def _latest_user_message(body: Mapping[str, JsonValue]) -> JsonValue:
    rows: Final = tuple(object_value(row) for row in list_value(body.get("structured_messages") or []))
    users: Final = tuple(row for row in rows if row.get("role") == "user")
    return users[-1] if users else body.get("texts")


def _verdict(vendor: _Vendor, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    if _CARD_PATTERN.search(json.dumps(body)) is None:
        return {"action": "NONE"}
    texts: Final = _masked(list_value(body.get("texts") or []))
    sent_rows: Final = tuple(object_value(row) for row in list_value(body.get("structured_messages") or []))
    rows: Final = tuple(object_value(row) for row in list_value(_masked([dict(row) for row in sent_rows])))
    tools: Final = list(list_value(_masked(list_value(body.get("tools") or []))))
    match vendor:
        case _Vendor.MASKS_ROWS_ONLY:
            return {"action": "GUARDRAIL_INTERVENED", "structured_messages": [dict(row) for row in rows]}
        case _Vendor.MASKS_ROWS_AND_BLANKS_EMPTY_CONTENT:
            return {
                "action": "GUARDRAIL_INTERVENED",
                "texts": texts,
                "structured_messages": list(map(_blank_content, rows)),
            }
        case _Vendor.JOINS_MASKED_ROWS:
            return {
                "action": "GUARDRAIL_INTERVENED",
                "texts": texts,
                "structured_messages": list(map(_flattened, rows)),
            }
        case _Vendor.ECHOES_ROWS:
            return {"action": "GUARDRAIL_INTERVENED", "texts": texts, "structured_messages": list(map(dict, sent_rows))}
        case _Vendor.RESPACES_UNMASKED_ROWS:
            return {
                "action": "GUARDRAIL_INTERVENED",
                "texts": texts,
                "structured_messages": list(map(_normalised_first_part, sent_rows)),
            }
        case _Vendor.COMPACTS_TOOL_RESULTS:
            return {"action": "GUARDRAIL_INTERVENED", "structured_messages": list(map(_compacted, rows))}
        case _Vendor.DROPS_A_ROW:
            return {"action": "GUARDRAIL_INTERVENED", "texts": texts, "structured_messages": list(map(dict, rows[:-1]))}
        case _Vendor.MASKS_TEXTS_AND_ROW_CONTENT:
            return {
                "action": "GUARDRAIL_INTERVENED",
                "texts": texts,
                "structured_messages": list(map(_content_masked, sent_rows)),
            }
        case _Vendor.CLEARS_CARD_TEXTS:
            return {
                "action": "GUARDRAIL_INTERVENED",
                "texts": _card_cleared(list_value(body.get("texts") or [])),
                "structured_messages": _card_cleared([dict(row) for row in sent_rows]),
            }
        case _Vendor.MASKS_TEXTS_ONLY:
            return {"action": "GUARDRAIL_INTERVENED", "texts": texts}
        case _Vendor.BLOCKS:
            return {"action": "BLOCKED", "blocked_reason": "card number"}
        case _Vendor.MASKS_TOOLS:
            return {"action": "GUARDRAIL_INTERVENED", "texts": texts, "tools": tools}
        case _Vendor.DROPS_A_TOOL:
            return {"action": "GUARDRAIL_INTERVENED", "texts": texts, "tools": tools[1:]}
        case _Vendor.ADDS_A_TOOL:
            return {"action": "GUARDRAIL_INTERVENED", "texts": texts, "tools": [*tools, _ADDED_TOOL]}
        case _Vendor.RENAMES_A_TOOL:
            renamed: Final = object_value(tools[0])
            function: Final = object_value(renamed.get("function"))
            return {
                "action": "GUARDRAIL_INTERVENED",
                "texts": texts,
                "tools": [{**renamed, "function": {**function, "name": "lookup_card_v2"}}, *tools[1:]],
            }
        case _Vendor.ECHOES_TOOLS_AS_NOT_STRICT:
            echoed: Final = tuple(object_value(tool) for tool in tools)
            return {
                "action": "GUARDRAIL_INTERVENED",
                "texts": texts,
                "tools": [
                    {**tool, "function": {**object_value(tool.get("function")), "strict": False}} for tool in echoed
                ],
            }
        case _Vendor.JUDGES_LATEST_USER_MESSAGE:
            if _CARD_PATTERN.search(json.dumps(_latest_user_message(body))) is None:
                return {"action": "NONE"}
            return {"action": "BLOCKED", "blocked_reason": "card number"}


def _vendor(request: Request) -> Reply:
    vendor: Final = _Vendor(request.target.removeprefix("/gga/").removesuffix(_GUARD_PATH))
    return Reply(body=json.dumps(_verdict(vendor, JSON_OBJECT.validate_json(request.body))).encode())


def _bedrock(request: Request) -> Reply:
    assert request.target == f"/model/{_MODEL_ID}/converse", request.target
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "guarded reply"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 7, "outputTokens": 2, "totalTokens": 9},
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


def _guardrail(name: str, params: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {"guardrail_name": name, "litellm_params": {"mode": "pre_call", "default_on": False, **params}}


def _generic(vendor: _Vendor) -> str:
    return f"gga-{vendor.value}"


def _guardrails(vendor_url: str) -> tuple[dict[str, JsonValue], ...]:
    generic: Final = {"guardrail": "generic_guardrail_api", "api_key": "synthetic-guardrail-key"}
    return (
        *(
            _guardrail(_generic(vendor), {**generic, "api_base": f"{vendor_url}/gga/{vendor.value}"})
            for vendor in _Vendor
        ),
        _guardrail("mcp-end-user", {"guardrail": "mcp_end_user_permission"}),
        _guardrail(
            "gga-block-skip-tool",
            {**generic, "api_base": f"{vendor_url}/gga/{_Vendor.BLOCKS.value}", "skip_tool_message_in_guardrail": True},
        ),
        _guardrail(
            "gga-mask-tool-only",
            {
                **generic,
                "api_base": f"{vendor_url}/gga/{_Vendor.MASKS_TEXTS_ONLY.value}",
                "scan_only_tool_results": True,
            },
        ),
        _guardrail("tool-policy", {"guardrail": "tool_policy"}),
    )


def _config(directory: Path, guardrails: tuple[dict[str, JsonValue], ...], bedrock: Wire) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["model_list"] = [
        *configuration["model_list"],
        *(
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": f"bedrock/{_MODEL_ID}",
                    "api_base": bedrock.url,
                    "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
                    "aws_secret_access_key": "scripted-secret",
                    "aws_region_name": "us-east-1",
                },
            }
            for model_name in (_MODEL, *_PIPELINED_ACTIONS)
        ),
    ]
    configuration["guardrails"] = list(guardrails)
    configuration["policies"] = {
        model_name: {
            "guardrails": {"add": [_generic(_Vendor.DROPS_A_ROW)]},
            "pipeline": {
                "mode": "pre_call",
                "steps": [{"guardrail": _generic(_Vendor.DROPS_A_ROW), "on_pass": "next", **actions}],
            },
        }
        for model_name, actions in _PIPELINED_ACTIONS.items()
    }
    configuration["policy_attachments"] = [
        {"policy": model_name, "models": [model_name]} for model_name in _PIPELINED_ACTIONS
    ]
    path: Final = directory / "converse-guardrails.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("converse-guardrail-rewrites")
    with (
        gateway_from_environment() as gateway,
        wire_server(_vendor) as vendor,
        wire_server(_bedrock) as bedrock,
    ):
        guardrails: Final = _guardrails(vendor.url)
        names: Final = tuple(str(guardrail["guardrail_name"]) for guardrail in guardrails)
        with (
            owned_proxy(gateway, directory, {}, config=_config(directory, guardrails, bedrock), workers=2) as owned,
            owned.scenario() as scenario,
        ):
            keys: Final = {name: scenario.key(guardrails=[name]) for name in names}
            tool_remover_then_masker: Final = ["mcp-end-user", "gga-mask-tool-only"]
            yield _Rig(
                owned,
                vendor,
                bedrock,
                {**keys, "+".join(tool_remover_then_masker): scenario.key(guardrails=tool_remover_then_masker)},
            )


@pytest.fixture(autouse=True)
def _drained(rig: _Rig) -> Iterator[None]:
    rig.vendor.drain()
    rig.bedrock.drain()
    yield
    rig.vendor.drain()
    rig.bedrock.drain()


def _converse(rig: _Rig, guardrail: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    return rig.gateway.request("POST", f"/bedrock/model/{_MODEL}/converse", body, key=rig.keys[guardrail])


def _forwarded(rig: _Rig, response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    sent: Final = rig.bedrock.drain()
    assert len(sent) == 1, response.text
    return JSON_OBJECT.validate_json(sent[0].body)


def _refused(rig: _Rig, response: httpx.Response) -> str:
    assert response.status_code >= 400, response.text
    assert rig.bedrock.drain() == (), response.text
    return response.text


def _vendor_bodies(rig: _Rig) -> tuple[dict[str, JsonValue], ...]:
    return tuple(JSON_OBJECT.validate_json(request.body) for request in rig.vendor.drain())


def _user(*blocks: JsonValue) -> dict[str, JsonValue]:
    return {"role": "user", "content": list(blocks)}


def _assistant(*blocks: JsonValue) -> dict[str, JsonValue]:
    return {"role": "assistant", "content": list(blocks)}


def _tool_use(card: str) -> dict[str, JsonValue]:
    return {"toolUse": {"toolUseId": "tooluse_lookup", "name": "lookup_card", "input": {"card": card, "limit": 2}}}


def _tool_result(*content: JsonValue) -> dict[str, JsonValue]:
    return {"toolResult": {"toolUseId": "tooluse_lookup", "content": list(content), "status": "success"}}


def _tool_loop(card_line: str, tool_call_card: str) -> dict[str, JsonValue]:
    return {
        "messages": [
            _user({"text": card_line}),
            _assistant(_tool_use(tool_call_card)),
            _user(_tool_result({"text": "card is active"})),
        ]
    }


def test_card_the_vendor_masks_only_in_messages_reaches_bedrock_masked(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": f"Card {_CARD} please."})]}
    sent: Final = _forwarded(rig, _converse(rig, _generic(_Vendor.MASKS_ROWS_ONLY), body))
    assert sent["messages"] == [_user({"text": f"Card {_MASK} please."})]


def test_message_masks_reach_every_block_of_an_agent_loop_with_the_image_intact(rig: _Rig) -> None:
    image: Final = {"image": {"format": "png", "source": {"bytes": _PNG}}}
    body: Final = {
        "system": [{"text": f"Card on file {_CARD}."}],
        "messages": [
            _user({"text": "Check this."}, {"guardContent": {"text": {"text": f"Card {_CARD}."}}}),
            _assistant(_tool_use(_CARD)),
            _user(_tool_result({"text": f"found {_CARD}"}, image, {"json": {"card": _CARD, "status": "active"}})),
        ],
    }
    sent: Final = _forwarded(rig, _converse(rig, _generic(_Vendor.MASKS_ROWS_AND_BLANKS_EMPTY_CONTENT), body))
    assert sent == _masked(body)


def test_flattened_messages_reach_bedrock_masked_in_their_original_blocks(rig: _Rig) -> None:
    body: Final = {
        "system": [{"text": "You are a billing assistant."}, {"text": f"Card on file {_CARD}."}],
        "messages": [_user({"text": "Order details below."}, {"text": f"Card {_CARD} please."})],
    }
    sent: Final = _forwarded(rig, _converse(rig, _generic(_Vendor.JOINS_MASKED_ROWS), body))
    assert sent == _masked(body)


def test_echoed_messages_keep_the_text_mask(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": f"Card {_CARD} please."})]}
    sent: Final = _forwarded(rig, _converse(rig, _generic(_Vendor.ECHOES_ROWS), body))
    assert sent["messages"] == [_user({"text": f"Card {_MASK} please."})]


def test_text_mask_wins_over_a_message_change_that_left_the_card(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": f"Card {_CARD} on file."}, {"text": "Thanks."})]}
    sent: Final = _forwarded(rig, _converse(rig, _generic(_Vendor.RESPACES_UNMASKED_ROWS), body))
    assert sent["messages"] == [_user({"text": f"Card {_MASK} on file."}, {"text": "Thanks."})]


def test_compacted_tool_result_reaches_bedrock_as_one_text_block_with_the_image_in_place(rig: _Rig) -> None:
    image: Final = {"image": {"format": "png", "source": {"bytes": _PNG}}}
    body: Final = {
        "messages": [
            _user({"text": "Look up my card."}),
            _assistant(_tool_use("last four 1111")),
            _user(_tool_result({"text": f"found {_CARD}"}, image, {"json": {"card": _CARD, "status": "active"}})),
        ]
    }
    sent: Final = _forwarded(rig, _converse(rig, _generic(_Vendor.COMPACTS_TOOL_RESULTS), body))
    compacted: Final = f"compacted: found {_MASK}\n{json.dumps({'card': _MASK, 'status': 'active'})}"
    assert sent["messages"] == [
        _user({"text": "Look up my card."}),
        _assistant(_tool_use("last four 1111")),
        _user(_tool_result({"text": compacted}, image)),
    ]


@pytest.mark.parametrize(
    ("user_text", "request_fields"),
    [
        (f"Card {_CARD} please.", {"thinking": {"type": "enabled", "budget_tokens": 1024}}),
        (f"Card {_CARD} please.", {"anthropic_beta": ["interleaved-thinking-2025-05-14"]}),
        ("Hello.", {"system": f"Card {_CARD} please."}),
    ],
    ids=["card-next-to-thinking", "card-next-to-anthropic-beta", "card-in-a-request-field"],
)
def test_guardrail_judging_the_latest_user_message_reads_it_with_the_request_fields(
    rig: _Rig, user_text: str, request_fields: Mapping[str, JsonValue]
) -> None:
    body: Final = {"messages": [_user({"text": user_text})], "additionalModelRequestFields": dict(request_fields)}
    assert "card number" in _refused(rig, _converse(rig, _generic(_Vendor.JUDGES_LATEST_USER_MESSAGE), body))


def test_dropped_message_is_refused_before_bedrock(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": "Hello."}), _assistant({"text": "Hi."}), _user({"text": f"{_CARD}"})]}
    assert _UNAPPLIABLE in _refused(rig, _converse(rig, _generic(_Vendor.DROPS_A_ROW), body))


def test_text_mask_the_messages_leave_out_of_a_tool_call_is_refused_before_bedrock(rig: _Rig) -> None:
    body: Final = _tool_loop(f"Charge card {_CARD}.", _CARD)
    assert _UNAPPLIABLE in _refused(rig, _converse(rig, _generic(_Vendor.MASKS_TEXTS_AND_ROW_CONTENT), body))


@pytest.mark.parametrize("model_name", tuple(_PIPELINED_ACTIONS))
def test_pipeline_step_refuses_a_rewrite_it_cannot_apply_whatever_its_actions(rig: _Rig, model_name: str) -> None:
    body: Final = {"messages": [_user({"text": "Hello."}), _assistant({"text": "Hi."}), _user({"text": f"{_CARD}"})]}
    response: Final = rig.gateway.request("POST", f"/bedrock/model/{model_name}/converse", body)
    assert _UNAPPLIABLE in _refused(rig, response)


def test_skip_tool_messages_keeps_tool_call_arguments_away_from_the_guardrail(rig: _Rig) -> None:
    body: Final = _tool_loop("Check my card.", _CARD)
    sent: Final = _forwarded(rig, _converse(rig, "gga-block-skip-tool", body))
    assert sent["messages"] == body["messages"]
    assert all(_CARD not in json.dumps(scanned) for scanned in _vendor_bodies(rig))


def test_tool_only_guardrail_scans_and_masks_only_tool_results(rig: _Rig) -> None:
    body: Final = {
        "messages": [
            _user({"text": f"My card is {_CARD}."}),
            _assistant(_tool_use("lookup")),
            _user(_tool_result({"text": f"found {_CARD}"})),
        ],
        "toolConfig": _TOOL_CONFIG,
    }
    sent: Final = _forwarded(rig, _converse(rig, "gga-mask-tool-only", body))
    scanned: Final = _vendor_bodies(rig)
    assert [(payload["texts"], payload.get("tools")) for payload in scanned] == [([f"found {_CARD}"], None)]
    assert scanned[0]["structured_messages"] == [
        {"role": "tool", "tool_call_id": "tooluse_lookup", "content": f"found {_CARD}"}
    ]
    assert sent == {**body, "messages": [*body["messages"][:2], _user(_tool_result({"text": f"found {_MASK}"}))]}


def test_masked_tool_definitions_reach_bedrock_masked(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": "Which cards are on file?"})], "toolConfig": _TOOL_CONFIG}
    sent: Final = _forwarded(rig, _converse(rig, _generic(_Vendor.MASKS_TOOLS), body))
    assert sent == _masked(body)


def test_tool_the_guardrail_removed_never_reaches_bedrock_nor_does_its_cache_point(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": "Which cards are on file?"})], "toolConfig": _TOOL_CONFIG}
    sent: Final = _forwarded(rig, _converse(rig, _generic(_Vendor.DROPS_A_TOOL), body))
    assert sent["toolConfig"] == {"tools": _TOOL_CONFIG["tools"][2:]}


def test_tool_choice_the_guardrail_removed_is_refused_before_bedrock(rig: _Rig) -> None:
    body: Final = {
        "messages": [_user({"text": "Which cards are on file?"})],
        "toolConfig": {**_TOOL_CONFIG, "toolChoice": {"tool": {"name": "lookup_card"}}},
    }
    assert _UNAPPLIABLE in _refused(rig, _converse(rig, _generic(_Vendor.DROPS_A_TOOL), body))


def _converted_tool(name: str, description: str, properties: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    schema: Final = {"type": "object", "properties": dict(properties), "required": []}
    return {"toolSpec": {"name": name, "description": description, "inputSchema": {"json": schema}}}


def _masked_lookup(name: str) -> dict[str, JsonValue]:
    properties: Final = {"card": {"type": "string", "description": f"for example {_MASK}"}}
    return _converted_tool(name, f"Looks up cards like {_MASK}", properties)


def test_tool_the_guardrail_added_reaches_bedrock_after_the_others(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": "Which cards are on file?"})], "toolConfig": _TOOL_CONFIG}
    sent: Final = _forwarded(rig, _converse(rig, _generic(_Vendor.ADDS_A_TOOL), body))
    assert sent["toolConfig"] == {
        "tools": [
            *list_value(object_value(_masked(_TOOL_CONFIG))["tools"]),
            _converted_tool("retrieve", "retrieve", {}),
        ]
    }


def test_tool_the_guardrail_added_to_a_request_without_tools_reaches_bedrock(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": f"Card {_CARD} please."})]}
    sent: Final = _forwarded(rig, _converse(rig, _generic(_Vendor.ADDS_A_TOOL), body))
    assert sent == {
        "messages": [_user({"text": f"Card {_MASK} please."})],
        "toolConfig": {"tools": [_converted_tool("retrieve", "retrieve", {})]},
    }


def test_tool_the_guardrail_renamed_reaches_bedrock_without_the_old_cache_point(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": "Which cards are on file?"})], "toolConfig": _TOOL_CONFIG}
    sent: Final = _forwarded(rig, _converse(rig, _generic(_Vendor.RENAMES_A_TOOL), body))
    assert sent["toolConfig"] == {"tools": [_masked_lookup("lookup_card_v2"), _TOOL_CONFIG["tools"][2]]}


def test_tools_echoed_with_an_added_key_are_rebuilt_and_keep_their_cache_point(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": "Which cards are on file?"})], "toolConfig": _TOOL_CONFIG}
    sent: Final = _forwarded(rig, _converse(rig, _generic(_Vendor.ECHOES_TOOLS_AS_NOT_STRICT), body))
    assert sent["toolConfig"] == {
        "tools": [
            _masked_lookup("lookup_card"),
            {"cachePoint": {"type": "default"}},
            _converted_tool("refund", "Refunds an order", {}),
        ]
    }


def test_text_block_the_guardrail_cleared_is_dropped_before_bedrock(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": "Order details below."}, {"text": f"Card {_CARD} please."})]}
    sent: Final = _forwarded(rig, _converse(rig, _generic(_Vendor.CLEARS_CARD_TEXTS), body))
    assert sent["messages"] == [_user({"text": "Order details below."})]


def test_turn_the_guardrail_cleared_entirely_is_sent_as_written(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": f"Card {_CARD} please."})]}
    sent: Final = _forwarded(rig, _converse(rig, _generic(_Vendor.CLEARS_CARD_TEXTS), body))
    assert sent["messages"] == [_user({"text": ""})]


def test_deeply_nested_tool_result_json_reaches_bedrock_with_the_message_mask(rig: _Rig) -> None:
    nested: JsonValue = "status active"
    for _ in range(500):
        nested = {"k": nested}
    body: Final = {
        "messages": [
            _user({"text": "Look up my card."}),
            _assistant(_tool_use("last four 1111")),
            _user(_tool_result({"json": nested}), {"text": f"Card {_CARD} please."}),
        ]
    }
    response: Final = _converse(rig, _generic(_Vendor.MASKS_ROWS_AND_BLANKS_EMPTY_CONTENT), body)
    assert response.status_code == 200, response.text
    sent: Final = rig.bedrock.drain()
    assert len(sent) == 1, response.text
    forwarded: object = json.loads(sent[0].body)
    assert forwarded == {
        "messages": [
            _user({"text": "Look up my card."}),
            _assistant(_tool_use("last four 1111")),
            _user(_tool_result({"json": nested}), {"text": f"Card {_MASK} please."}),
        ]
    }


def _named_tools(*names: str) -> dict[str, JsonValue]:
    return {"tools": [{"toolSpec": {"name": name, "inputSchema": {"json": {"type": "object"}}}} for name in names]}


def test_tool_policy_refuses_a_converse_tool_whose_input_policy_is_blocked(rig: _Rig) -> None:
    blocked: Final = f"converse_blocked_{uuid.uuid4().hex[:8]}"
    rig.gateway.post("/v1/tool/policy", {"tool_name": blocked, "input_policy": "blocked"})
    body: Final = {"messages": [_user({"text": "Refund my order."})], "toolConfig": _named_tools(blocked, "refund")}
    held(lambda: _converse(rig, "tool-policy", body).status_code, lambda status: status == 400, holding=2, seconds=60)
    rig.bedrock.drain()
    assert "Violated tool policy" in _refused(rig, _converse(rig, "tool-policy", body))


def _end_user_on_github_only(rig: _Rig) -> dict[str, str]:
    end_user: Final = f"converse-mcp-{uuid.uuid4().hex[:8]}"
    rig.gateway.post("/customer/new", {"user_id": end_user, "object_permission": {"mcp_servers": ["github"]}})
    return {"x-litellm-end-user-id": end_user}


def test_mcp_end_user_permission_drops_tools_of_servers_the_end_user_cannot_use(rig: _Rig) -> None:
    body: Final = {
        "messages": [_user({"text": "File the bug."})],
        "toolConfig": _named_tools("github-create_issue", "slack-post_message", "get_weather"),
    }
    response: Final = rig.gateway.request(
        "POST",
        f"/bedrock/model/{_MODEL}/converse",
        body,
        key=rig.keys["mcp-end-user"],
        headers=_end_user_on_github_only(rig),
    )
    assert _forwarded(rig, response)["toolConfig"] == _named_tools("github-create_issue", "get_weather")


@pytest.mark.parametrize(
    "guardrail_config",
    [{}, {"guardrailConfig": {"guardrailIdentifier": "gr-scripted", "guardrailVersion": "1"}}],
    ids=["plain", "with-guardrail-config"],
)
def test_mcp_end_user_permission_removing_every_tool_drops_the_tool_config_and_turns_tool_history_into_text(
    rig: _Rig, guardrail_config: Mapping[str, JsonValue]
) -> None:
    tool_use: Final = {"toolUseId": "tooluse_post", "name": "slack-post_message", "input": {"channel": "#ops"}}
    body: Final = {
        **guardrail_config,
        "messages": [
            _user({"text": "Post the update."}),
            _assistant({"toolUse": tool_use}),
            _user(
                {"toolResult": {"toolUseId": "tooluse_post", "content": [{"text": "posted to #ops"}]}},
                {"text": "Again."},
            ),
        ],
        "toolConfig": {**_named_tools("slack-post_message"), "toolChoice": {"any": {}}},
    }
    response: Final = rig.gateway.request(
        "POST",
        f"/bedrock/model/{_MODEL}/converse",
        body,
        key=rig.keys["mcp-end-user"],
        headers=_end_user_on_github_only(rig),
    )
    assert _forwarded(rig, response) == {
        **guardrail_config,
        "messages": [
            _user({"text": "Post the update."}),
            _assistant({"text": '[tool call tooluse_post: slack-post_message({"channel": "#ops"})]'}),
            _user({"text": "[tool result for tooluse_post: posted to #ops]"}, {"text": "Again."}),
        ],
    }


def test_tool_result_of_a_tool_mcp_end_user_permission_removed_is_masked_by_a_later_tool_results_only_guardrail(
    rig: _Rig,
) -> None:
    tool_use: Final = {"toolUseId": "tooluse_post", "name": "slack-post_message", "input": {"channel": "#ops"}}
    body: Final = {
        "messages": [
            _user({"text": "Post the update."}),
            _assistant({"toolUse": tool_use}),
            _user(
                {"toolResult": {"toolUseId": "tooluse_post", "content": [{"text": f"posted card {_CARD}"}]}},
                {"text": "Again."},
            ),
        ],
        "toolConfig": _named_tools("slack-post_message"),
    }
    response: Final = rig.gateway.request(
        "POST",
        f"/bedrock/model/{_MODEL}/converse",
        body,
        key=rig.keys["mcp-end-user+gga-mask-tool-only"],
        headers=_end_user_on_github_only(rig),
    )
    sent: Final = _forwarded(rig, response)
    assert "toolConfig" not in sent
    assert sent["messages"][-1] == _user(
        {"text": f"[tool result for tooluse_post: posted card {_MASK}]"}, {"text": "Again."}
    )
