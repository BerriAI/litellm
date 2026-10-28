import base64
import binascii
import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway, Scenario, eventually
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span

IDENTITIES: Final = (
    "user_api_key_alias",
    "user_api_key_user_id",
    "user_api_key_end_user_id",
    "user_api_key_team_id",
    "user_api_key_team_alias",
)
EXPORTED_FIELDS: Final = frozenset({*IDENTITIES, "user_api_key_org_id"})


@dataclass(frozen=True, slots=True)
class LangfuseRig:
    candidate: Gateway
    provider: Wire
    collector: Wire
    proxy: OwnedProxy


def exported_spans(body: bytes) -> Iterator[Span]:
    for resource in ExportTraceServiceRequest.FromString(body).resource_spans:
        for scope in resource.scope_spans:
            yield from scope.spans


def _attribute_values(span: Span) -> dict[str, object]:
    return {
        attribute.key: getattr(attribute.value, attribute.value.WhichOneof("value")) for attribute in span.attributes
    }


def span_attributes(body: bytes) -> tuple[dict[str, object], ...]:
    return tuple(map(_attribute_values, exported_spans(body)))


def _sse_chat(response_id: str) -> tuple[bytes, ...]:
    chunk: Final = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
    }
    return (
        f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'langfuse-echo'}, 'finish_reason': None}]})}\n\n".encode(),
        f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode(),
        f"data: {json.dumps({**chunk, 'choices': [], 'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}})}\n\n".encode(),
        b"data: [DONE]\n\n",
    )


def _sse_messages(response_id: str) -> tuple[bytes, ...]:
    def event(name: str, payload: Mapping[str, object]) -> bytes:
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()

    message: Final = {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
    return (
        event("message_start", {"type": "message_start", "message": message}),
        event(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "langfuse-echo"}},
        ),
        event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        ),
        event("message_stop", {"type": "message_stop"}),
    )


def _sse_responses(response_id: str) -> tuple[bytes, ...]:
    def event(name: str, payload: Mapping[str, object]) -> bytes:
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()

    response: Final = {
        "id": response_id,
        "object": "response",
        "model": "gpt-4o-mini",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": f"item-{response_id}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "langfuse-echo"}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }
    return (
        event(
            "response.created",
            {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
        ),
        event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "response_id": response_id,
                "item_id": f"item-{response_id}",
                "output_index": 0,
                "content_index": 0,
                "delta": "langfuse-echo",
            },
        ),
        event("response.completed", {"type": "response.completed", "response": response}),
    )


def _chat_body(response_id: str) -> dict[str, object]:
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "langfuse-echo"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _message_body(response_id: str) -> dict[str, object]:
    return {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "langfuse-echo"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


_RESPONSE_FIELDS: Final = {
    "created_at": 1,
    "error": None,
    "incomplete_details": None,
    "instructions": None,
    "metadata": {},
    "parallel_tool_calls": True,
    "temperature": 1.0,
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1.0,
    "max_output_tokens": None,
    "previous_response_id": None,
    "store": True,
    "truncation": "disabled",
    "user": None,
}


def _response_body(response_id: str) -> dict[str, object]:
    return {
        **_RESPONSE_FIELDS,
        "id": response_id,
        "object": "response",
        "model": "gpt-4o-mini",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": f"item-{response_id}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "langfuse-echo"}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }


@contextmanager
def langfuse_rig(
    gateway: Gateway,
    tmp_path: Path,
    settings: Mapping[str, object],
    marker: str,
    *,
    sink_reply: Callable[[Request], Reply] | None = None,
    overrides: Mapping[str, str] | None = None,
    model_list: Callable[[str], Sequence[Mapping[str, object]]] | None = None,
) -> Iterator[LangfuseRig]:
    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b'{"data":[]}')
        body: Final = json.loads(request.body or b"{}")
        suffix: Final = uuid.uuid4().hex[:8]
        if "-fail-" in json.dumps(body):
            return Reply(
                status=500, body=b'{"error": {"message": "scripted provider failure", "type": "server_error"}}'
            )
        if request.target.endswith("/chat/completions"):
            chat_id: Final = f"chatcmpl-{marker}-{suffix}"
            if body.get("stream"):
                return Reply(content_type="text/event-stream", chunks=_sse_chat(chat_id))
            return Reply(body=json.dumps(_chat_body(chat_id)).encode())
        if request.target.endswith("/messages"):
            message_id: Final = f"msg-{marker}-{suffix}"
            if body.get("stream"):
                return Reply(content_type="text/event-stream", chunks=_sse_messages(message_id))
            return Reply(body=json.dumps(_message_body(message_id)).encode())
        if request.target.endswith("/responses"):
            resp_id: Final = f"resp-{marker}-{suffix}"
            if body.get("stream"):
                return Reply(content_type="text/event-stream", chunks=_sse_responses(resp_id))
            return Reply(body=json.dumps(_response_body(resp_id)).encode())
        raise AssertionError(request.target)

    def sink(request: Request) -> Reply:
        return sink_reply(request) if sink_reply is not None else Reply()

    with wire_server(upstream) as provider, wire_server(sink) as collector:
        raw_config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config: Final = {
            **raw_config,
            "model_list": [*raw_config.get("model_list", []), *(model_list(provider.url) if model_list else ())],
            "litellm_settings": {**raw_config["litellm_settings"], "callbacks": ["langfuse_otel"], **settings},
        }
        path: Final = tmp_path / "langfuse_otel.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(
            gateway,
            tmp_path,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-lf-integration",
                "LANGFUSE_SECRET_KEY": "sk-lf-integration",
                "LANGFUSE_HOST": collector.url,
                "LITELLM_OTEL_V2": "1",
                **{
                    name: (collector.url if value == "__collector__" else value)
                    for name, value in dict(overrides or {}).items()
                },
            },
            config=path,
            workers=2,
        ) as owned:
            yield LangfuseRig(owned.gateway, provider, collector, owned)


def decoded_response_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if value.startswith("resp_"):
        try:
            decoded: Final = base64.b64decode(value[5:] + "===").decode()
        except (binascii.Error, UnicodeDecodeError):
            return value
        _, _, suffix = decoded.partition("response_id:")
        if suffix:
            return suffix.split(";", 1)[0]
    return value


def span_response_id(span: Mapping[str, object]) -> str | None:
    return decoded_response_id(span.get("gen_ai.response.id"))


def watched_spans(collector: Wire) -> Callable[[], tuple[dict[str, object], ...]]:
    drained: list[
        dict[str, object]
    ] = []  # mutable-ok: drain() consumes the wire queue, so observed spans must accumulate across eventually polls

    def snapshot() -> tuple[dict[str, object], ...]:
        drained.extend(attributes for batch in collector.drain() for attributes in span_attributes(batch.body))
        return tuple(drained)

    return snapshot


def assert_key_alias(spans: Sequence[Mapping[str, object]], key_alias: str | None) -> None:
    assert spans, "no span landed"
    for span in spans:
        assert json.loads(str(span["langfuse.observation.metadata"]))["user_api_key_alias"] == key_alias, span


def exported_llm_span(collector: Wire, response_id: str) -> dict[str, object]:
    def matches(span: Mapping[str, object]) -> bool:
        return decoded_response_id(span.get("gen_ai.response.id")) == decoded_response_id(response_id)

    observed: Final = eventually(
        watched_spans(collector), lambda spans: sum(matches(span) for span in spans) == 1, seconds=25
    )
    return next(span for span in observed if matches(span))


def scenario_identity(
    scenario: Scenario,
    provider: Wire,
    *,
    spend_logs_metadata: Mapping[str, object] | None = None,
    team: bool = True,
) -> tuple[str, dict[str, str | None]]:
    team_alias: Final = ("lf-team-" + uuid.uuid4().hex) if team else None
    team_id: Final = scenario.team(team_alias=team_alias) if team else None
    key_alias: Final = "lf-key-" + uuid.uuid4().hex
    key: Final = scenario.key(
        **({"team_id": team_id} if team_id else {}),
        key_alias=key_alias,
        metadata={"spend_logs_metadata": spend_logs_metadata or {"ticket": "LIT-8283"}},
    )
    model: Final = scenario.model(api_base=provider.url + "/v1")
    end_user: Final = "lf-end-user-" + uuid.uuid4().hex
    return key, {
        "key_alias": key_alias,
        "team_id": team_id,
        "team_alias": team_alias,
        "end_user": end_user,
        "model": model,
    }


def assert_identity(attrs: Mapping[str, object], expected: Mapping[str, object]) -> None:
    assert "langfuse.observation.metadata" in attrs, sorted(attrs)
    observation: Final = json.loads(str(attrs["langfuse.observation.metadata"]))
    assert {key: observation.get(key) for key in expected} == dict(expected), observation
    assert set(observation) <= EXPORTED_FIELDS, observation
    assert all(isinstance(value, str) and value for value in observation.values()), observation
    flattened: Final = {
        attribute: attrs.get(attribute)
        for attribute in (f"langfuse.trace.metadata.{field}" for field in IDENTITIES)
        if attribute in attrs
    }
    assert flattened == {f"langfuse.trace.metadata.{field}": value for field, value in expected.items()}, attrs


def chat_identity_fields(identity: Mapping[str, str | None]) -> dict[str, object]:
    return {
        "user_api_key_alias": identity["key_alias"],
        **({"user_api_key_team_id": identity["team_id"]} if identity["team_id"] else {}),
        **({"user_api_key_team_alias": identity["team_alias"]} if identity["team_alias"] else {}),
        **({"user_api_key_end_user_id": identity["end_user"]} if identity["end_user"] else {}),
    }


def span_records(body: bytes) -> tuple[tuple[dict[str, object], int], ...]:
    return tuple((_attribute_values(span), span.dropped_attributes_count) for span in exported_spans(body))


def identity_attributes(attrs: Mapping[str, object]) -> frozenset[str]:
    names: Final = {"langfuse.observation.metadata", *(f"langfuse.trace.metadata.{field}" for field in IDENTITIES)}
    return frozenset(key for key in attrs if key in names)
