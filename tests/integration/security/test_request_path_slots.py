"""Request-path slots D1 to D4: a credential the client sends with the request reaches only the provider.

Each slot is a credential the proxy receives on the request itself and must hand to the provider
without keeping a copy:

- D1: ``api_key`` in the request body.
- D2: ``x-api-key`` forwarded with ``general_settings.forward_llm_provider_auth_headers``.
- D3: an ``x-goog-api-key`` client header forwarded with
  ``litellm_settings.model_group_settings.forward_client_headers_to_llm_api`` (with
  ``forward_llm_provider_auth_headers`` on, which lets a provider auth header through).
- D4: an Anthropic OAuth token (``Authorization: Bearer sk-ant-oat...``) sent next to
  ``x-litellm-api-key``, forwarded to an Anthropic deployment.

A test is one slot on one route. It sends three requests carrying the same canary: one the
provider answers, one it rejects with a 4xx and one it fails with a 5xx, because failure logging
takes a different path. Positive control: every provider request of every outcome must carry the
canary where the slot delivers it. Sensitivity control: the marker sent in the same requests must
be in the spend-log row of every outcome and in a sink event of every outcome, and each sweep must
report it where stored prompts belong. The route sweep fills its request-id routes with the
successful row, so the Logs drawer and the spend-log filter are also read for each failed row,
where the marker must show too. Then no sweep may find the slot's canary anywhere.

The requests go one at a time, and each waits for its sink event before the next is sent. The
``generic_api`` logger clears its whole queue after a batch POST, so an event queued while a POST
is in flight would be dropped, and the sweep would then miss that outcome's callback payload.

One owned proxy per slot serves every route of that slot. The canary travels on the request and
never in the config, so a fresh core per test needs no fresh proxy; the config only turns the
slot's setting on. Rows and sink events left by earlier tests carry other cores, which the sweeps
of a later test do not search for.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Final
from urllib.parse import quote, urlencode

import httpx
import pytest
from integration._support.client import Scenario, eventually, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request
from integration.security._canary import MARKER, Canary, canary, find_canary
from integration.security._sinks import GENERIC_SINK, PROVIDER_4XX, Caller, Rig, canary_rig
from integration.security._sweeps import Hit, assert_marker_seen, assert_no_hits, record_route_sweep, sweep_all

PROVIDER_5XX: Final = "canary-provider-5xx"
OPENAI_MODEL: Final = "canary-request-openai"
ANTHROPIC_MODEL: Final = "canary-request-anthropic"
FORWARDED_HEADER: Final = "x-goog-api-key"
DEPLOYMENT_KEY: Final = "canary-deployment-placeholder-key"
OUTCOMES: Final = MappingProxyType({"success": 200, "provider_4xx": 400, "provider_5xx": 500})


@dataclass(frozen=True, slots=True)
class Route:
    """A client route: its path, the field that carries the prompt text, and fixed extra fields."""

    path: str
    text_field: str
    extra: Mapping[str, object] = MappingProxyType({})

    def body(self, model: str, text: str) -> dict[str, object]:
        prompt: Final[object] = [{"role": "user", "content": text}] if self.text_field == "messages" else text
        return {"model": model, self.text_field: prompt, **self.extra}


ROUTES: Final = MappingProxyType(
    {
        "chat": Route("/v1/chat/completions", "messages"),
        "chat_stream": Route("/v1/chat/completions", "messages", MappingProxyType({"stream": True})),
        "messages": Route("/v1/messages", "messages", MappingProxyType({"max_tokens": 16})),
        "messages_stream": Route("/v1/messages", "messages", MappingProxyType({"max_tokens": 16, "stream": True})),
        "responses": Route("/v1/responses", "input"),
        "embeddings": Route("/v1/embeddings", "input"),
    }
)


@dataclass(frozen=True, slots=True)
class RequestSlot:
    """How a slot's canary rides the request, where the provider must receive it, and its setting."""

    model: str
    routes: tuple[str, ...]
    body: Callable[[Canary], Mapping[str, object]]
    headers: Callable[[Canary, str], Mapping[str, str]]
    delivered: Callable[[Request], str | None]
    expected: Callable[[Canary], str]
    configure: Callable[[dict[str, object]], None]


def _no_body(_canary: Canary) -> Mapping[str, object]:
    return {}


def _bearer_key(_canary: Canary, key: str) -> Mapping[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _authorization(request: Request) -> str | None:
    return request.headers.get("authorization")


def _bearer(value: Canary) -> str:
    return f"Bearer {value.value}"


def _no_setting(_config: dict[str, object]) -> None:
    return None


def _forward_provider_auth(config: dict[str, object]) -> None:
    general: Final = config["general_settings"]
    assert isinstance(general, dict)
    general["forward_llm_provider_auth_headers"] = True


def _forward_client_headers(config: dict[str, object]) -> None:
    _forward_provider_auth(config)
    settings: Final = config["litellm_settings"]
    assert isinstance(settings, dict)
    settings["model_group_settings"] = {"forward_client_headers_to_llm_api": [OPENAI_MODEL]}


OPENAI_ROUTES: Final = ("chat", "chat_stream", "messages", "responses", "embeddings")
FORWARDED_HEADER_ROUTES: Final = ("chat", "chat_stream", "messages", "responses")
ANTHROPIC_ROUTES: Final = ("messages", "messages_stream", "chat", "responses")

REQUEST_SLOTS: Final = MappingProxyType(
    {
        "D1": RequestSlot(
            OPENAI_MODEL,
            OPENAI_ROUTES,
            lambda value: {"api_key": value.value},
            _bearer_key,
            _authorization,
            _bearer,
            _no_setting,
        ),
        "D2": RequestSlot(
            OPENAI_MODEL,
            OPENAI_ROUTES,
            _no_body,
            lambda value, key: {"Authorization": f"Bearer {key}", "x-api-key": value.value},
            _authorization,
            _bearer,
            _forward_provider_auth,
        ),
        "D3": RequestSlot(
            OPENAI_MODEL,
            FORWARDED_HEADER_ROUTES,
            _no_body,
            lambda value, key: {"Authorization": f"Bearer {key}", FORWARDED_HEADER: value.value},
            lambda request: request.headers.get(FORWARDED_HEADER),
            lambda value: value.value,
            _forward_client_headers,
        ),
        "D4": RequestSlot(
            ANTHROPIC_MODEL,
            ANTHROPIC_ROUTES,
            _no_body,
            lambda value, key: {"Authorization": f"Bearer {value.value}", "x-litellm-api-key": key},
            _authorization,
            _bearer,
            _no_setting,
        ),
    }
)


def _sse(events: tuple[tuple[str | None, dict[str, object]], ...], done: bool) -> tuple[bytes, ...]:
    frames: Final = tuple(
        (f"event: {name}\n" if name else "").encode() + b"data: " + json.dumps(data).encode() + b"\n\n"
        for name, data in events
    )
    return (*frames, b"data: [DONE]\n\n") if done else frames


def _anthropic_reply(stream: bool) -> Reply:
    message: Final = {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    if not stream:
        return Reply(body=json.dumps(message).encode())
    events: Final = (
        ("message_start", {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    return Reply(content_type="text/event-stream", chunks=_sse(events, done=False))


def _chat_reply(stream: bool) -> Reply:
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                }
            ).encode()
        )
    base: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    events: Final = (
        (
            None,
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
        ),
        (None, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        (None, {**base, "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}),
    )
    return Reply(content_type="text/event-stream", chunks=_sse(events, done=True))


def _responses_reply() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"resp_{uuid.uuid4().hex}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex}",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            }
        ).encode()
    )


def _embeddings_reply() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            }
        ).encode()
    )


def _error(status: int, anthropic: bool) -> Reply:
    kind: Final = "invalid_request_error" if status < 500 else "api_error"
    body: Final = (
        {"type": "error", "error": {"type": kind, "message": "rejected"}}
        if anthropic
        else {"error": {"type": kind, "code": "canary_rejected", "message": "rejected"}}
    )
    return Reply(status=status, body=json.dumps(body).encode())


def provider_upstream(request: Request) -> Reply:
    """OpenAI chat, responses and embeddings plus Anthropic messages; fails on the outcome triggers."""
    anthropic: Final = request.target.startswith("/v1/messages")
    if PROVIDER_5XX.encode() in request.body:
        return _error(500, anthropic)
    if PROVIDER_4XX.encode() in request.body:
        return _error(400, anthropic)
    stream: Final = json.loads(request.body or b"{}").get("stream") is True
    if anthropic:
        return _anthropic_reply(stream)
    if request.target.startswith("/v1/responses"):
        return _responses_reply()
    if request.target.startswith("/v1/embeddings"):
        return _embeddings_reply()
    return _chat_reply(stream)


def _configure(slot: RequestSlot) -> Callable[[dict[str, object], str], None]:
    def configure(config: dict[str, object], provider_url: str) -> None:
        models: Final = config["model_list"]
        assert isinstance(models, list)
        models.extend(
            (
                {
                    "model_name": OPENAI_MODEL,
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_base": provider_url + "/v1",
                        "api_key": DEPLOYMENT_KEY,
                    },
                },
                {
                    "model_name": ANTHROPIC_MODEL,
                    "litellm_params": {
                        "model": "anthropic/claude-sonnet-4-5",
                        "api_base": provider_url,
                        "api_key": DEPLOYMENT_KEY,
                    },
                },
            )
        )
        slot.configure(config)

    return configure


@pytest.fixture(scope="module")
def rig(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    """One owned proxy per slot, shared by every route of that slot (see the module docstring)."""
    slot_id: Final = str(request.param)
    with canary_rig(
        tmp_path_factory.mktemp(f"canary-{slot_id}"),
        configure=_configure(REQUEST_SLOTS[slot_id]),
        upstream=provider_upstream,
    ) as value:
        yield value


def _caller(scenario: Scenario, model: str) -> Caller:
    team: Final = scenario.team()
    user: Final = scenario.user(user_role="internal_user")
    scenario.gateway.post("/team/member_add", {"team_id": team, "member": {"user_id": user, "role": "user"}})
    return Caller(team, user, scenario.key(team_id=team, user_id=user, models=[model]))


def _trigger(outcome: str) -> str:
    return {"success": "", "provider_4xx": f" {PROVIDER_4XX}", "provider_5xx": f" {PROVIDER_5XX}"}[outcome]


def _tag(outcome: str, marker: Canary) -> str:
    return f"{outcome} {marker.value}"


def _spend_rows(marker: Canary) -> list[dict[str, object]]:
    return [
        dict(row)
        for row in read_rows(
            'SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE proxy_server_request::text LIKE %s',
            (f"%{marker.core}%",),
        )
    ]


def _request_row_hits(
    rig: Rig, callers: Mapping[str, str], request_ids: tuple[str, ...], canaries: tuple[Canary, ...]
) -> tuple[Hit, ...]:
    """S2 for the rows the route sweep does not fill in: the Logs drawer and the spend-log filter per row."""
    found: Final[list[Hit]] = []  # mutable-ok: accumulated across rows and callers
    for request_id in request_ids:
        for path in (
            f"/spend/logs/ui/{quote(request_id, safe='')}",
            f"/spend/logs?{urlencode({'request_id': request_id})}",
        ):
            for label, key in callers.items():
                response = rig.proxy.client.get(path, headers={"Authorization": f"Bearer {key}"})
                where = f"GET {path} as {label} -> {response.status_code}"
                found.extend(
                    Hit("S2", where, match.slot, match.encoding) for match in find_canary(response.content, canaries)
                )
    return tuple(found)


CASES: Final = tuple(
    pytest.param(slot_id, slot_id, route, id=f"{slot_id}-{route}")
    for slot_id, slot in REQUEST_SLOTS.items()
    for route in slot.routes
)


@pytest.mark.timeout(240)  # three requests, then the full S1/S2 walk as two callers
@pytest.mark.parametrize(("rig", "slot_id", "route"), CASES, indirect=["rig"], scope="module")
def test_request_credential_reaches_only_the_provider(
    rig: Rig, slot_id: str, route: str, request: pytest.FixtureRequest
) -> None:
    slot: Final = REQUEST_SLOTS[slot_id]
    endpoint: Final = ROUTES[route]
    credential: Final = canary(slot_id)
    marker: Final = canary(MARKER)
    started: Final = datetime.now(UTC)
    with rig.proxy.scenario() as scenario:
        caller: Final = _caller(scenario, slot.model)
        responses: Final[list[httpx.Response]] = []
        for outcome, status in OUTCOMES.items():
            response = rig.proxy.client.post(
                endpoint.path,
                json={
                    **endpoint.body(slot.model, f"slot {slot_id} {_tag(outcome, marker)}{_trigger(outcome)}"),
                    **slot.body(credential),
                },
                headers=dict(slot.headers(credential, caller.key)),
            )
            responses.append(response)
            assert response.status_code == status, f"{outcome}: {response.status_code} {response.text}"
            for name, sink in rig.sinks.items():
                assert eventually(
                    lambda sink=sink, outcome=outcome: sink.carrying(_tag(outcome, marker)),
                    bool,
                    seconds=30,
                    return_last_on_timeout=True,
                ), f"Sensitivity control: {name} never received the {outcome} event"

        for outcome in OUTCOMES:
            delivered = rig.provider.carrying(_tag(outcome, marker))
            assert delivered and all(slot.delivered(each) == slot.expected(credential) for each in delivered), (
                f"Positive control: the provider double never received the {slot_id} canary for {outcome}: "
                f"{[dict(each.headers) for each in delivered]}"
            )

        rows: Final = eventually(lambda: _spend_rows(marker), lambda found: len(found) == len(OUTCOMES), seconds=70)
        assert sorted(string_value(row["status"]) for row in rows) == ["failure", "failure", "success"], rows
        request_id: Final = next(string_value(row["request_id"]) for row in rows if row["status"] == "success")
        failed_ids: Final = tuple(string_value(row["request_id"]) for row in rows if row["status"] == "failure")

        report: Final = sweep_all(
            rig.proxy,
            (marker, credential),
            responses=tuple(responses),
            sinks={name: sink.requests() for name, sink in rig.sinks.items()},
            ids={
                "request_id": request_id,
                "team_id": caller.team_id,
                "user_id": caller.user_id,
                "model_id": slot.model,
                "model": slot.model,
            },
            callers=caller.callers(rig),
            own_headers=rig.own_headers,
            since=started,
        )
        record_route_sweep(report.routes, request.node.nodeid)
        assert_marker_seen(
            report,
            {
                "S1": "LiteLLM_SpendLogs.proxy_server_request",
                "S2": f"GET /spend/logs/ui/{quote(request_id, safe='')} as admin -> 200",
                "S4": f"{GENERIC_SINK}[",
            },
        )
        assert_marker_seen(report, {"S2": f"GET /spend/logs?{urlencode({'request_id': request_id})} as admin -> 200"})
        failure_rows: Final = _request_row_hits(rig, caller.callers(rig), failed_ids, (marker, credential))
        for failed_id in failed_ids:
            drawer = f"GET /spend/logs/ui/{quote(failed_id, safe='')} as admin -> 200"
            assert any(hit.slot == MARKER and hit.location == drawer for hit in failure_rows), (
                f"Sensitivity control: the marker is missing from {drawer}"
            )
        assert_no_hits(
            (*report.credential_hits(), *(hit for hit in failure_rows if hit.slot != MARKER)),
            f"slot {slot_id}, {endpoint.path} ({route})",
        )
