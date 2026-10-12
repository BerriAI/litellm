from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, list_value, object_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_MODEL_ID: Final = "anthropic.claude-sonnet-5-v1:0"
_MODEL: Final = "converse-guarded"
_CARD: Final = "4111 1111 1111 1111"
_CARD_PATTERN: Final = re.compile(r"\b(?:\d{4} ){3}\d{4}\b")
_MASK: Final = "****"
_FLAGGED: Final = "flagged-by-moderation"
_GUARD_PATH: Final = "/beta/litellm_basic_guardrail_api"
_PNG: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
_JSON: Final = TypeAdapter(JsonValue)
_UNAPPLIABLE: Final = "rewrote the request in a way this endpoint cannot apply"
_GENERIC_MODES: Final = (
    "masksm",
    "masksm_flat",
    "masksm_blank",
    "smonly",
    "sm_partial",
    "echo_sm",
    "mask",
    "block",
    "rowcount",
)


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


def _generic_verdict(mode: str, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    texts: Final = list_value(body.get("texts") or [])
    rows: Final = tuple(object_value(row) for row in list_value(body.get("structured_messages") or []))
    masked_texts: Final = _masked(list(texts))
    masked_rows: Final = list_value(_masked([dict(row) for row in rows]))
    if _CARD_PATTERN.search(json.dumps(body)) is None:
        return {"action": "NONE"}
    verdicts: Final[Mapping[str, Callable[[], dict[str, JsonValue]]]] = {
        "masksm": lambda: {"action": "GUARDRAIL_INTERVENED", "texts": masked_texts, "structured_messages": masked_rows},
        "masksm_flat": lambda: {
            "action": "GUARDRAIL_INTERVENED",
            "texts": masked_texts,
            "structured_messages": [_flattened(object_value(row)) for row in masked_rows],
        },
        "masksm_blank": lambda: {
            "action": "GUARDRAIL_INTERVENED",
            "texts": masked_texts,
            "structured_messages": [_blank_content(object_value(row)) for row in masked_rows],
        },
        "smonly": lambda: {"action": "GUARDRAIL_INTERVENED", "structured_messages": masked_rows},
        "sm_partial": lambda: {
            "action": "GUARDRAIL_INTERVENED",
            "texts": masked_texts,
            "structured_messages": [_normalised_first_part(row) for row in rows],
        },
        "echo_sm": lambda: {
            "action": "GUARDRAIL_INTERVENED",
            "texts": masked_texts,
            "structured_messages": [dict(row) for row in rows],
        },
        "mask": lambda: {"action": "GUARDRAIL_INTERVENED", "texts": masked_texts},
        "block": lambda: {"action": "BLOCKED", "blocked_reason": "card number"},
        "rowcount": lambda: {
            "action": "GUARDRAIL_INTERVENED",
            "texts": masked_texts,
            "structured_messages": masked_rows[:-1],
        },
    }
    return verdicts[mode]()


def _vendor(request: Request) -> Reply:
    body: Final = JSON_OBJECT.validate_json(request.body)
    if request.target == "/moderation/moderations":
        flagged: Final = _FLAGGED in str(body.get("input"))
        result: Final = {
            "flagged": flagged,
            "categories": {"violence": flagged},
            "category_scores": {"violence": 0.99 if flagged else 0.0},
            "category_applied_input_types": {"violence": ["text"]},
        }
        return Reply(
            body=json.dumps({"id": "modr-scripted", "model": "omni-moderation-latest", "results": [result]}).encode()
        )
    if request.target == "/promptguard/api/v1/guard":
        messages: Final = body.get("messages")
        if _CARD_PATTERN.search(json.dumps(messages)) is None:
            return Reply(body=json.dumps({"decision": "allow"}).encode())
        return Reply(body=json.dumps({"decision": "redact", "redacted_messages": _masked(messages)}).encode())
    mode: Final = request.target.removeprefix("/gga/").removesuffix(_GUARD_PATH)
    return Reply(body=json.dumps(_generic_verdict(mode, body)).encode())


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


def _config(directory: Path, vendor: Wire, bedrock: Wire) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["model_list"] = [
        *configuration["model_list"],
        {
            "model_name": _MODEL,
            "litellm_params": {
                "model": f"bedrock/{_MODEL_ID}",
                "api_base": bedrock.url,
                "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
                "aws_secret_access_key": "scripted-secret",
                "aws_region_name": "us-east-1",
            },
        },
    ]
    generic: Final = {"guardrail": "generic_guardrail_api", "api_key": "synthetic-guardrail-key"}
    configuration["guardrails"] = [
        *(_guardrail(f"gga-{mode}", {**generic, "api_base": f"{vendor.url}/gga/{mode}"}) for mode in _GENERIC_MODES),
        _guardrail(
            "gga-block-skip-tool",
            {**generic, "api_base": f"{vendor.url}/gga/block", "skip_tool_message_in_guardrail": True},
        ),
        _guardrail(
            "gga-mask-tool-only",
            {**generic, "api_base": f"{vendor.url}/gga/mask", "scan_only_tool_results": True},
        ),
        _guardrail(
            "promptguard",
            {"guardrail": "promptguard", "api_base": f"{vendor.url}/promptguard", "api_key": "synthetic-pg-key"},
        ),
        _guardrail(
            "moderation",
            {"guardrail": "openai_moderation", "api_base": f"{vendor.url}/moderation", "api_key": "synthetic-key"},
        ),
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
        owned_proxy(gateway, directory, {}, config=_config(directory, vendor, bedrock), workers=2) as owned,
        owned.scenario() as scenario,
    ):
        names: Final = (
            *(f"gga-{mode}" for mode in _GENERIC_MODES),
            "gga-block-skip-tool",
            "gga-mask-tool-only",
            "promptguard",
            "moderation",
        )
        yield _Rig(owned, vendor, bedrock, {name: scenario.key(guardrails=[name]) for name in names})


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


def test_multi_block_user_message_with_flattened_vendor_reply_reaches_bedrock_masked(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": "Order details below."}, {"text": f"Card {_CARD} please."})]}
    sent: Final = _forwarded(rig, _converse(rig, "gga-masksm_flat", body))
    assert sent["messages"] == [_user({"text": "Order details below."}, {"text": f"Card {_MASK} please."})]


def test_multi_block_system_prompt_with_flattened_vendor_reply_reaches_bedrock_masked(rig: _Rig) -> None:
    body: Final = {
        "system": [{"text": "You are a billing assistant."}, {"text": f"Card on file {_CARD}."}],
        "messages": [_user({"text": "What card is on file?"})],
    }
    sent: Final = _forwarded(rig, _converse(rig, "gga-masksm_flat", body))
    assert sent["system"] == [{"text": "You are a billing assistant."}, {"text": f"Card on file {_MASK}."}]
    assert sent["messages"] == body["messages"]


def test_tool_result_with_text_and_json_parts_reaches_bedrock_masked(rig: _Rig) -> None:
    body: Final = {
        "messages": [
            _user({"text": "Look up my card."}),
            _assistant(_tool_use("last four 1111")),
            _user(_tool_result({"text": f"found {_CARD}"}, {"json": {"card": _CARD, "status": "active"}})),
        ]
    }
    sent: Final = _forwarded(rig, _converse(rig, "gga-masksm", body))
    assert sent["messages"] == [
        _user({"text": "Look up my card."}),
        _assistant(_tool_use("last four 1111")),
        _user(_tool_result({"text": f"found {_MASK}"}, {"json": {"card": _MASK, "status": "active"}})),
    ]


def test_tool_result_with_text_and_image_parts_reaches_bedrock_masked_with_image_intact(rig: _Rig) -> None:
    image: Final = {"image": {"format": "png", "source": {"bytes": _PNG}}}
    body: Final = {
        "messages": [
            _user({"text": "Read the screenshot."}),
            _assistant(_tool_use("screenshot")),
            _user(_tool_result({"text": f"screenshot shows {_CARD}"}, image)),
        ]
    }
    sent: Final = _forwarded(rig, _converse(rig, "gga-masksm", body))
    assert sent["messages"] == [
        _user({"text": "Read the screenshot."}),
        _assistant(_tool_use("screenshot")),
        _user(_tool_result({"text": f"screenshot shows {_MASK}"}, image)),
    ]


def test_flat_text_mask_wins_over_an_unmasked_structured_leaf(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": f"Card {_CARD} on file."}, {"text": "Thanks."})]}
    sent: Final = _forwarded(rig, _converse(rig, "gga-sm_partial", body))
    assert sent["messages"] == [_user({"text": f"Card {_MASK} on file."}, {"text": "Thanks."})]


def test_promptguard_redaction_keeps_every_turn_in_place(rig: _Rig) -> None:
    body: Final = {
        "system": [{"text": "You are a support bot."}],
        "messages": [
            _user({"text": "My name is Ann."}),
            _assistant({"text": "Hi Ann, how can I help?"}),
            _user({"text": f"Repeat exactly: card {_CARD} ok"}),
        ],
    }
    sent: Final = _forwarded(rig, _converse(rig, "promptguard", body))
    assert sent["system"] == [{"text": "You are a support bot."}]
    assert sent["messages"] == [
        _user({"text": "My name is Ann."}),
        _assistant({"text": "Hi Ann, how can I help?"}),
        _user({"text": f"Repeat exactly: card {_MASK} ok"}),
    ]


def test_moderation_blocks_a_flagged_latest_user_turn_before_bedrock(rig: _Rig) -> None:
    body: Final = {
        "system": [{"text": "You are a support bot."}],
        "messages": [
            _user({"text": "Say hi."}),
            _assistant({"text": "Hi."}),
            _user({"text": f"please {_FLAGGED} now"}),
        ],
    }
    refused: Final = _refused(rig, _converse(rig, "moderation", body))
    assert "Violated OpenAI moderation policy" in refused


def test_skip_tool_messages_keeps_tool_call_arguments_away_from_the_guardrail(rig: _Rig) -> None:
    body: Final = {
        "messages": [
            _user({"text": "Check my card."}),
            _assistant(_tool_use(_CARD)),
            _user(_tool_result({"text": "card is active"})),
        ]
    }
    sent: Final = _forwarded(rig, _converse(rig, "gga-block-skip-tool", body))
    assert sent["messages"] == body["messages"]
    assert all(_CARD not in json.dumps(scanned) for scanned in _vendor_bodies(rig))


def test_tool_only_guardrail_receives_no_tool_definitions(rig: _Rig) -> None:
    body: Final = {
        "messages": [
            _user({"text": f"My card is {_CARD}."}),
            _assistant(_tool_use("lookup")),
            _user(_tool_result({"text": f"found {_CARD}"})),
        ],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": "lookup_card",
                        "description": "Look up a card",
                        "inputSchema": {"json": {"type": "object", "properties": {"card": {"type": "string"}}}},
                    }
                }
            ]
        },
    }
    sent: Final = _forwarded(rig, _converse(rig, "gga-mask-tool-only", body))
    scanned: Final = _vendor_bodies(rig)
    assert len(scanned) == 1
    assert scanned[0].get("tools") is None, scanned[0]
    assert scanned[0]["texts"] == [f"found {_CARD}"]
    assert sent["messages"] == [
        _user({"text": f"My card is {_CARD}."}),
        _assistant(_tool_use("lookup")),
        _user(_tool_result({"text": f"found {_MASK}"})),
    ]


def test_structured_only_vendor_mask_reaches_bedrock(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": f"Card {_CARD} please."})]}
    sent: Final = _forwarded(rig, _converse(rig, "gga-smonly", body))
    assert sent["messages"] == [_user({"text": f"Card {_MASK} please."})]


def test_guard_content_turn_reaches_bedrock_masked(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": "Check this."}, {"guardContent": {"text": {"text": f"Card {_CARD}."}}})]}
    sent: Final = _forwarded(rig, _converse(rig, "gga-masksm", body))
    assert sent["messages"] == [_user({"text": "Check this."}, {"guardContent": {"text": {"text": f"Card {_MASK}."}}})]


def test_tool_call_arguments_in_history_reach_bedrock_masked(rig: _Rig) -> None:
    body: Final = {
        "messages": [
            _user({"text": "Check my card."}),
            _assistant(_tool_use(_CARD)),
            _user(_tool_result({"text": "card is active"})),
        ]
    }
    sent: Final = _forwarded(rig, _converse(rig, "gga-masksm", body))
    assert sent["messages"] == [
        _user({"text": "Check my card."}),
        _assistant(_tool_use(_MASK)),
        _user(_tool_result({"text": "card is active"})),
    ]


def test_vendor_that_blanks_an_empty_assistant_turn_still_reaches_bedrock_masked(rig: _Rig) -> None:
    body: Final = {
        "messages": [
            _user({"text": "Check my card."}),
            _assistant(_tool_use(_CARD)),
            _user(_tool_result({"text": "card is active"})),
        ]
    }
    sent: Final = _forwarded(rig, _converse(rig, "gga-masksm_blank", body))
    assert sent["messages"] == [
        _user({"text": "Check my card."}),
        _assistant(_tool_use(_MASK)),
        _user(_tool_result({"text": "card is active"})),
    ]


def test_echoed_structured_rows_keep_the_text_mask(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": f"Card {_CARD} please."})]}
    sent: Final = _forwarded(rig, _converse(rig, "gga-echo_sm", body))
    assert sent["messages"] == [_user({"text": f"Card {_MASK} please."})]


def test_single_block_string_reply_reaches_bedrock_masked(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": f"Card {_CARD} please."})]}
    sent: Final = _forwarded(rig, _converse(rig, "gga-masksm_flat", body))
    assert sent["messages"] == [_user({"text": f"Card {_MASK} please."})]


def test_missing_structured_row_is_refused_before_bedrock(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": "Hello."}), _assistant({"text": "Hi."}), _user({"text": f"{_CARD}"})]}
    assert _UNAPPLIABLE in _refused(rig, _converse(rig, "gga-rowcount", body))


def test_blocked_verdict_never_reaches_bedrock(rig: _Rig) -> None:
    body: Final = {"messages": [_user({"text": f"Card {_CARD} please."})]}
    _refused(rig, _converse(rig, "gga-block", body))


def test_concurrent_requests_keep_their_own_rewrites(rig: _Rig) -> None:
    cards: Final = tuple(f"4111 1111 1111 {index:04d}" for index in range(24))
    bodies: Final = tuple(
        {"messages": [_user({"text": f"request {index}"}, {"text": f"card {card}"})]}
        for index, card in enumerate(cards)
    )
    with ThreadPoolExecutor(max_workers=12) as pool:
        responses: Final = tuple(pool.map(lambda body: _converse(rig, "gga-masksm_flat", body), bodies))
    assert [response.status_code for response in responses] == [200] * len(bodies)
    sent: Final = sorted(
        (JSON_OBJECT.validate_json(request.body)["messages"] for request in rig.bedrock.drain()), key=json.dumps
    )
    expected: Final = sorted(
        ([_user({"text": f"request {index}"}, {"text": f"card {_MASK}"})] for index in range(len(cards))),
        key=json.dumps,
    )
    assert sent == expected
