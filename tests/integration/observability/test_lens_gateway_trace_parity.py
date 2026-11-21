"""The trace the gateway sends Lens for a Claude Code run, against the trace Claude Code uploads itself.

`lens_claude_code_direct_trace.json` is the trace a real Claude Code 2.1.329 run uploaded straight to
Lens (recorded 2026-11-20, account and host attributes removed). The test replays that run's two model
calls through a real proxy and compares what the proxy exports to Lens with the recording
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator, Mapping
from itertools import chain
from pathlib import Path
from typing import Final

from _openinference_support import _attribute_text, _rig
from integration._support import claude_code as cc
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans
from pydantic import BaseModel, JsonValue

_MODEL: Final = "claude-haiku-4-5"
LENS_AGENT: Final = "lens-signals"
_TRACE_PATH: Final = "/v1/traces"


class _Span(BaseModel, frozen=True):
    name: str
    attributes: Mapping[str, str]


class _DirectTrace(BaseModel, frozen=True):
    resource: Mapping[str, str]
    spans: tuple[_Span, ...]


class _ForwardedBlock(BaseModel, frozen=True):
    type: str = ""
    tool_use_id: str = ""


class _ForwardedMessage(BaseModel, frozen=True):
    content: tuple[_ForwardedBlock, ...] | str = ""


class _ForwardedBody(BaseModel, frozen=True):
    messages: tuple[_ForwardedMessage, ...]


_DIRECT: Final = _DirectTrace.model_validate_json(
    (Path(__file__).parent / "lens_claude_code_direct_trace.json").read_text()
)
_DIRECT_MODEL_CALLS: Final = tuple(span for span in _DIRECT.spans if span.attributes.get("span.type") == "llm_request")
_DIRECT_TOOL_STEPS: Final = tuple(
    span for span in _DIRECT.spans if span.attributes.get("gen_ai.operation.name") == "execute_tool"
)
_SESSION: Final = _DIRECT.spans[0].attributes["session.id"]


def _usage(call: _Span) -> dict[str, JsonValue]:
    return {
        "input_tokens": int(call.attributes["input_tokens"]),
        "cache_read_input_tokens": int(call.attributes["cache_read_tokens"]),
        "cache_creation_input_tokens": int(call.attributes["cache_creation_tokens"]),
        "output_tokens": int(call.attributes["output_tokens"]),
    }


def _tool_uses() -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        {
            "type": "tool_use",
            "id": step.attributes["gen_ai.tool.call.id"],
            "name": step.attributes["gen_ai.tool.name"],
            "input": {"command": step.attributes["full_command"]},
        }
        for step in _DIRECT_TOOL_STEPS
    )


def _recorded_replies() -> Iterator[Reply]:
    first, last = _DIRECT_MODEL_CALLS
    yield Reply(body=cc.message_reply("msg_recorded_1", _MODEL, _tool_uses(), _usage(first)))
    yield Reply(
        body=cc.message_reply("msg_recorded_2", _MODEL, ({"type": "text", "text": "Listed both."},), _usage(last))
    )


def _lens(request: Request) -> Reply:
    return Reply(body=b"", content_type="application/x-protobuf")


def _gateway_spans(lens: Wire) -> tuple[tuple[dict[str, str], dict[str, str]], ...]:
    """Each exported span's attributes with its resource's attributes."""
    exports: Final = tuple(
        ExportTraceServiceRequest.FromString(
            gzip.decompress(request.body) if request.headers.get("content-encoding") == "gzip" else request.body
        )
        for request in lens.drain()
        if request.target == _TRACE_PATH and request.body
    )
    resources: Final = tuple(chain.from_iterable(export.resource_spans for export in exports))
    return tuple(chain.from_iterable(_resource_spans(resource) for resource in resources))


def _resource_spans(resource: ResourceSpans) -> tuple[tuple[dict[str, str], dict[str, str]], ...]:
    resource_attributes: Final = {item.key: _attribute_text(item.value) for item in resource.resource.attributes}
    spans: Final = chain.from_iterable(scope.spans for scope in resource.scope_spans)
    return tuple(
        ({item.key: _attribute_text(item.value) for item in span.attributes}, resource_attributes) for span in spans
    )


def _forwarded_tool_results(request: Request) -> tuple[str, ...]:
    messages: Final = _ForwardedBody.model_validate_json(request.body).messages
    blocks: Final = chain.from_iterable(message.content for message in messages if isinstance(message.content, tuple))
    return tuple(block.tool_use_id for block in blocks if block.type == "tool_result" and block.tool_use_id)


def _direct_input_tokens(call: _Span) -> int:
    return sum(int(call.attributes[key]) for key in ("input_tokens", "cache_read_tokens", "cache_creation_tokens"))


def test_gateway_trace_of_a_claude_code_run_matches_the_trace_claude_code_uploads_itself(
    gateway: Gateway, tmp_path: Path
) -> None:
    replies: Final = _recorded_replies()

    def upstream(_request: Request) -> Reply:
        return next(replies)

    with (
        wire_server(_lens) as lens,
        _rig(
            gateway,
            tmp_path,
            upstream,
            callbacks=(),
            model_name=f"anthropic/{_MODEL}",
            api_base_suffix="",
            destination_wire=lens,
            environment={
                "LITELLM_LENS_URL": lens.url,
                "LITELLM_LENS_SERVICE_TOKEN": "integration-lens-service-token-32-chars",
            },
            workers=1,
        ) as rig,
    ):
        headers: Final = {**cc.cli_headers(rig.proxy.key), "x-claude-code-session-id": _SESSION}
        first_turn: Final[dict[str, JsonValue]] = {
            **cc.claude_code_request("lens-parity"),
            "model": rig.model,
            "stream": False,
        }
        first: Final = rig.proxy.client.post("/v1/messages", json=first_turn, headers=headers)
        assert first.status_code == 200, first.text
        second: Final = rig.proxy.client.post(
            "/v1/messages",
            json=cc.tool_loop_turn2(
                first_turn,
                _tool_uses(),
                tuple((step.attributes["gen_ai.tool.call.id"], "recorded output") for step in _DIRECT_TOOL_STEPS),
            ),
            headers=headers,
        )
        assert second.status_code == 200, second.text
        assert tuple(_forwarded_tool_results(request) for request in rig.provider.drain()) == (
            (),
            tuple(step.attributes["gen_ai.tool.call.id"] for step in _DIRECT_TOOL_STEPS),
        ), "the provider must get the tool results the agent sent"

        expected_count: Final = len(_DIRECT_MODEL_CALLS) + len(_DIRECT_TOOL_STEPS)
        collected: Final[list[tuple[dict[str, str], dict[str, str]]]] = []  # mutable-ok: the sink is drained per poll
        spans: Final = eventually(
            lambda: collected.extend(_gateway_spans(lens)) or tuple(collected),
            lambda found: len(found) >= expected_count,
            seconds=60,
        )

    model_calls: Final = tuple(span for span, _ in spans if span.get("gen_ai.operation.name") == "chat")
    tool_steps: Final = tuple(span for span, _ in spans if span.get("gen_ai.operation.name") == "execute_tool")

    assert {
        "one run per session": {
            (resource["lens.session.capture"], span[resource["lens.session.attribute"]]) for span, resource in spans
        },
        "agent": {span.get("gen_ai.agent.name") for span, _ in spans},
        "model calls": sorted(
            (
                call["gen_ai.response.model"],
                int(call["gen_ai.usage.input_tokens"]),
                int(call["gen_ai.usage.output_tokens"]),
            )
            for call in model_calls
        ),
        "tool steps": sorted(
            (
                step["gen_ai.tool.name"],
                step["gen_ai.tool.call.id"],
                json.loads(step["gen_ai.tool.call.arguments"])["command"],
            )
            for step in tool_steps
        ),
    } == {
        "one run per session": {("true", _SESSION)},
        "agent": {_DIRECT.resource["service.name"]},
        "model calls": sorted(
            (call.attributes["gen_ai.request.model"], _direct_input_tokens(call), int(call.attributes["output_tokens"]))
            for call in _DIRECT_MODEL_CALLS
        ),
        "tool steps": sorted(
            (
                step.attributes["gen_ai.tool.name"],
                step.attributes["gen_ai.tool.call.id"],
                step.attributes["full_command"],
            )
            for step in _DIRECT_TOOL_STEPS
        ),
    }


def test_a_model_call_lens_makes_for_its_own_analysis_is_a_trace_under_the_lens_agent(
    gateway: Gateway, tmp_path: Path
) -> None:
    def upstream(_request: Request) -> Reply:
        return Reply(
            body=cc.message_reply(
                "msg_reply", _MODEL, ({"type": "text", "text": "ok"},), {"input_tokens": 3, "output_tokens": 1}
            )
        )

    with (
        wire_server(_lens) as lens,
        _rig(
            gateway,
            tmp_path,
            upstream,
            callbacks=(),
            model_name=f"anthropic/{_MODEL}",
            api_base_suffix="",
            destination_wire=lens,
            environment={
                "LITELLM_LENS_URL": lens.url,
                "LITELLM_LENS_SERVICE_TOKEN": "integration-lens-service-token-32-chars",
            },
            workers=1,
        ) as rig,
    ):
        body: Final[dict[str, JsonValue]] = {
            **cc.claude_code_request("lens-internal"),
            "model": rig.model,
            "stream": False,
        }
        analysis: Final = rig.proxy.client.post(
            "/v1/messages",
            json=body,
            headers={
                **cc.cli_headers(rig.proxy.key),
                "x-claude-code-session-id": "lens-analysis-session",
                "user-agent": LENS_AGENT,
            },
        )
        assert analysis.status_code == 200, analysis.text
        customer: Final = rig.proxy.client.post(
            "/v1/messages",
            json=body,
            headers={**cc.cli_headers(rig.proxy.key), "x-claude-code-session-id": "customer-session"},
        )
        assert customer.status_code == 200, customer.text

        collected: Final[list[tuple[dict[str, str], dict[str, str]]]] = []  # mutable-ok: the sink is drained per poll
        spans: Final = eventually(
            lambda: collected.extend(_gateway_spans(lens)) or tuple(collected),
            lambda found: (
                {"customer-session", "lens-analysis-session"}
                <= {span.get("gen_ai.conversation.id") for span, _ in found}
            ),
            seconds=60,
        )

    agents: Final = {span.get("gen_ai.conversation.id", ""): span.get("gen_ai.agent.name") for span, _ in spans}
    assert agents == {"lens-analysis-session": LENS_AGENT, "customer-session": "claude-code"}, (
        "Lens leaves its own calls out of analysis by this agent name, so the gateway has to export them under it"
    )
