from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Final

import httpx
import openai
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.responses_vendor import same_response
from integration._support.wire import Reply, Request, Wire
from integration._support.worker_readiness import wait_for_worker_readiness
from openai.types.responses import CompactedResponse, ResponseInputParam
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

MODEL: Final = "gpt-5"
API_KEY: Final = "synthetic-responses-key"
ALIASES: Final = ("/v1", "", "/openai/v1")
COMPACT_STATE: Final = "synthetic-compact-state"
COMPACT_USAGE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "input_tokens": 7,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 10,
    }
)
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
PROXY_WORKERS: Final = max(1, int(os.environ.get("INTEGRATION_PROXY_WORKERS", "1")))
SPEND_MARKER_HEADER: Final = "x-litellm-spend-logs-metadata"
_DISABLE_HINT: Final = (
    "To disable this security feature, set general_settings::disable_responses_id_security to True "
    "in the config.yaml file."
)
OTHER_USER_DETAIL: Final = (
    f"Forbidden. The response id is not associated with the user, who this key belongs to. {_DISABLE_HINT}"
)
OTHER_TEAM_DETAIL: Final = (
    f"Forbidden. The response id is not associated with the team, who this key belongs to. {_DISABLE_HINT}"
)
UNMANAGED_DETAIL: Final = (
    "Forbidden. This response id was not issued by this proxy, so the proxy cannot tell who owns it. "
    "To let keys address responses this proxy did not issue, set "
    "general_settings::allow_unmanaged_response_ids to True in the config.yaml file."
)
SCRIPTED_FAILURES: Final[Mapping[str, int]] = MappingProxyType(
    {"fail400": 400, "fail401": 401, "fail404": 404, "fail500": 500}
)


class _InputText(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str


class _InputMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: tuple[_InputText, ...]


_COMPACT_INPUT: Final = TypeAdapter(tuple[_InputMessage, ...])


def marker(label: str) -> str:
    return f"{label}_{uuid.uuid4().hex}"


def minted_id(created_marker: str) -> str:
    return f"resp_minted_{created_marker}"


def compacted_id(compact_marker: str) -> str:
    return f"resp_compacted_{compact_marker}"


def vendor_id() -> str:
    return f"resp_vendor_{uuid.uuid4().hex}"


def compact_input(text: str) -> ResponseInputParam:
    return [{"role": "user", "content": [{"type": "input_text", "text": text}]}]


def compact_input_json(text: str) -> list[JsonValue]:
    return [{"role": "user", "content": [{"type": "input_text", "text": text}]}]


def compact_body(model: str, text: str, extra: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {"model": model, "input": compact_input_json(text), **extra}


def upstream_compact_body(text: str, extra: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {"model": MODEL, "input": compact_input_json(text), **extra}


def compact_text(body: Mapping[str, JsonValue]) -> str:
    return _COMPACT_INPUT.validate_python(body["input"])[0].content[0].text


def _created_body(identity: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": MODEL,
            "output": [
                {
                    "id": f"msg_{identity}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "created", "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    ).encode()


def _compacted_body(identity: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "created_at": 1,
            "object": "response.compaction",
            "output": [
                {
                    "id": f"rs_{identity}",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Compacted state"}],
                    "encrypted_content": COMPACT_STATE,
                }
            ],
            "usage": dict(COMPACT_USAGE),
        }
    ).encode()


def _failure_body(status: int, text: str) -> bytes:
    kind: Final = "server_error" if status >= 500 else "invalid_request_error"
    return json.dumps(
        {"error": {"message": f"scripted {status} for {text}", "type": kind, "param": None, "code": None}}
    ).encode()


@dataclass(frozen=True, slots=True)
class CompactUpstream:
    gates: Mapping[str, threading.Event] = field(default_factory=lambda: MappingProxyType({}))

    def respond(self, request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(
                body=json.dumps(
                    {"object": "list", "data": [{"id": MODEL, "object": "model", "created": 1, "owned_by": "openai"}]}
                ).encode()
            )
        body: Final = JSON_OBJECT.validate_json(request.body)
        if request.method == "POST" and request.target == "/v1/responses":
            return Reply(body=_created_body(minted_id(str(body["input"]))))
        assert request.method == "POST" and request.target == "/v1/responses/compact", request.target
        text: Final = compact_text(body)
        label: Final = text.split("_", 1)[0]
        gate: Final = self.gates.get(label)
        if gate is not None:
            assert gate.wait(timeout=180), f"The held {label} compact was never released"
        status: Final = SCRIPTED_FAILURES.get(label)
        if status is not None:
            return Reply(status=status, body=_failure_body(status, text))
        return Reply(body=_compacted_body(compacted_id(text)))


@dataclass(frozen=True, slots=True)
class Upstreamed:
    target: str
    body: Mapping[str, JsonValue]


def upstreamed(wire: Wire) -> tuple[Upstreamed, ...]:
    return tuple(
        Upstreamed(request.target, JSON_OBJECT.validate_json(request.body))
        for request in wire.drain()
        if request.target != "/v1/models"
    )


def compacts(requests: tuple[Upstreamed, ...]) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(request.body for request in requests if request.target == "/v1/responses/compact")


def compact_hits(requests: tuple[Upstreamed, ...], text: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(body for body in compacts(requests) if compact_text(body) == text)


def leaks(requests: tuple[Upstreamed, ...], raw_id: str) -> bool:
    return any(raw_id in json.dumps(dict(request.body)) for request in requests)


@dataclass(frozen=True, slots=True)
class Tenants:
    model: str
    owner: str
    other_user: str
    other_team: str


def deployment(scenario: Scenario, wire: Wire) -> str:
    model: Final = scenario.model(model=f"openai/{MODEL}", api_key=API_KEY, api_base=f"{wire.url}/v1")
    wait_for_worker_readiness(scenario.gateway, model, 1, PROXY_WORKERS)
    return model


def tenants(scenario: Scenario, wire: Wire) -> Tenants:
    model: Final = deployment(scenario, wire)
    team_a: Final = scenario.team(models=[model])
    team_b: Final = scenario.team(models=[model])
    user_a: Final = scenario.member(team_a)
    user_b: Final = scenario.member(team_b)
    scenario.gateway.post("/team/member_add", {"team_id": team_b, "member": {"role": "user", "user_id": user_a}})
    return Tenants(
        model=model,
        owner=scenario.key(team_id=team_a, user_id=user_a, models=[model]),
        other_user=scenario.key(team_id=team_b, user_id=user_b, models=[model]),
        other_team=scenario.key(team_id=team_b, user_id=user_a, models=[model]),
    )


def base_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def spend_marker(text: str) -> dict[str, str]:
    return {SPEND_MARKER_HEADER: json.dumps({"marker": text})}


def sdk(gateway: Gateway, key: str, prefix: str) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{base_url(gateway)}{prefix}", api_key=key, max_retries=0)


def mint(gateway: Gateway, key: str, model: str, text: str) -> str:
    created: Final = sdk(gateway, key, "/v1").responses.create(
        model=model, input=text, extra_headers=spend_marker(text)
    )
    assert created.id != minted_id(text) and same_response(created.id, minted_id(text)), created.model_dump_json()
    return created.id


def sdk_compact(gateway: Gateway, key: str, prefix: str, model: str, text: str, previous: str) -> CompactedResponse:
    return sdk(gateway, key, prefix).responses.compact(
        model=model,
        input=compact_input(text),
        previous_response_id=previous,
        extra_headers=spend_marker(text),
    )


async def _async_compact(
    gateway: Gateway, key: str, prefix: str, model: str, text: str, previous: str
) -> CompactedResponse:
    async with openai.AsyncOpenAI(base_url=f"{base_url(gateway)}{prefix}", api_key=key, max_retries=0) as client:
        return await client.responses.compact(
            model=model,
            input=compact_input(text),
            previous_response_id=previous,
            extra_headers=spend_marker(text),
        )


def async_compact(gateway: Gateway, key: str, prefix: str, model: str, text: str, previous: str) -> CompactedResponse:
    return asyncio.run(_async_compact(gateway, key, prefix, model, text, previous))


def compact_post(gateway: Gateway, key: str, prefix: str, body: Mapping[str, JsonValue], text: str) -> httpx.Response:
    return httpx.post(
        f"{base_url(gateway)}{prefix}/responses/compact",
        json=dict(body),
        headers={"Authorization": f"Bearer {key}", **spend_marker(text)},
        timeout=120,
        trust_env=False,
    )


def compact_raw(gateway: Gateway, key: str, content: bytes, text: str) -> httpx.Response:
    return httpx.post(
        f"{base_url(gateway)}/v1/responses/compact",
        content=content,
        headers={"Authorization": f"Bearer {key}", "content-type": "application/json", **spend_marker(text)},
        timeout=120,
        trust_env=False,
    )


def assert_compacted(compacted: CompactedResponse, model: str, text: str) -> None:
    assert compacted.id != compacted_id(text) and same_response(compacted.id, compacted_id(text)), (
        compacted.model_dump_json()
    )
    assert compacted.model_dump(mode="json", exclude_none=True) == {
        "id": compacted.id,
        "created_at": 1,
        "model": model,
        "object": "response.compaction",
        "output": [
            {
                "id": f"rs_{compacted_id(text)}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Compacted state"}],
                "encrypted_content": COMPACT_STATE,
            }
        ],
        "usage": dict(COMPACT_USAGE),
    }, compacted.model_dump_json()


def assert_compacted_reply(response: httpx.Response, model: str, text: str) -> str:
    assert response.status_code == 200, response.text
    compacted: Final = CompactedResponse.model_validate_json(response.content)
    assert_compacted(compacted, model, text)
    return compacted.id


def assert_refused(response: httpx.Response, detail: str) -> None:
    assert response.status_code == 403, response.text
    assert response.json() == {
        "error": {"message": detail, "type": "permission_error", "param": None, "code": "403"}
    }, response.text


def spend_rows(key: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        read_rows(
            "SELECT request_id, status, call_type, metadata->'spend_logs_metadata'->>'marker' AS marker "
            'FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
            (sha256(key.encode()).hexdigest(),),
        )
    )


def _marker_counts(key: str, texts: frozenset[str]) -> Mapping[str, int]:
    return Counter(str(row["marker"]) for row in spend_rows(key) if row["marker"] in texts)


def landed_once(key: str, texts: frozenset[str]) -> tuple[Mapping[str, JsonValue], ...]:
    counts: Final = eventually(lambda: _marker_counts(key, texts), lambda found: frozenset(found) == texts, seconds=90)
    assert dict(counts) == dict.fromkeys(texts, 1), counts
    return tuple(row for row in spend_rows(key) if row["marker"] in texts)


def assert_compact_rows(key: str, texts: frozenset[str]) -> None:
    rows: Final = landed_once(key, texts)
    assert {(row["status"], row["call_type"]) for row in rows} == {("success", "acompact_responses")}, rows


def successful_markers(key: str, texts: frozenset[str]) -> frozenset[str]:
    return frozenset(
        str(row["marker"]) for row in spend_rows(key) if row["marker"] in texts and row["status"] == "success"
    )
