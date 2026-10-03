import base64
import binascii
import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import psutil
import yaml
from anthropic import Anthropic, AsyncAnthropic
from integration._support.client import Gateway, Scenario, eventually
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import AsyncOpenAI, OpenAI
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

_IDENTITIES: Final = (
    "user_api_key_alias",
    "user_api_key_user_id",
    "user_api_key_end_user_id",
    "user_api_key_team_id",
    "user_api_key_team_alias",
)


@dataclass(frozen=True, slots=True)
class _Rig:
    candidate: Gateway
    provider: Wire
    collector: Wire
    proxy: OwnedProxy


def _span_attributes(body: bytes) -> tuple[dict[str, object], ...]:
    request: Final = ExportTraceServiceRequest.FromString(body)
    return tuple(
        {attribute.key: getattr(attribute.value, attribute.value.WhichOneof("value")) for attribute in span.attributes}
        for resource in request.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    )


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
def _langfuse_rig(
    gateway: Gateway,
    tmp_path: Path,
    settings: Mapping[str, object],
    marker: str,
    *,
    sink_reply: Callable[[Request], Reply] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> Iterator[_Rig]:
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
                **{
                    name: (collector.url if value == "__collector__" else value)
                    for name, value in dict(overrides or {}).items()
                },
            },
            config=path,
            workers=2,
        ) as owned:
            yield _Rig(owned.gateway, provider, collector, owned)


def _response_id(value: object) -> str | None:
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


def _span_response_id(span: Mapping[str, object]) -> str | None:
    return _response_id(span.get("llm.response.id"))


def _watched_spans(collector: Wire) -> Callable[[], tuple[dict[str, object], ...]]:
    drained: list[
        dict[str, object]
    ] = []  # mutable-ok: drain() consumes the wire queue, so observed spans must accumulate across eventually polls

    def snapshot() -> tuple[dict[str, object], ...]:
        drained.extend(attributes for batch in collector.drain() for attributes in _span_attributes(batch.body))
        return tuple(drained)

    return snapshot


def _observation_span(collector: Wire, response_id: str) -> dict[str, object]:
    def spans() -> tuple[dict[str, object], ...]:
        return tuple(
            attributes
            for batch in collector.drain()
            for attributes in _span_attributes(batch.body)
            if _span_response_id(attributes) == _response_id(response_id)
        )

    observed: Final = eventually(spans, lambda values: len(values) == 1, seconds=25)
    return observed[0]


def _marked_span(collector: Wire, marker: str, needle: str) -> dict[str, object]:
    def spans() -> tuple[dict[str, object], ...]:
        return tuple(
            attributes
            for batch in collector.drain()
            for attributes in _span_attributes(batch.body)
            if needle in json.dumps(attributes, default=str) and marker in json.dumps(attributes, default=str)
        )

    observed: Final = eventually(spans, lambda values: len(values) == 1, seconds=25)
    return observed[0]


def _scenario_identity(
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


def _assert_identity(
    attrs: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    spend_logs_metadata: Mapping[str, object] | None = None,
) -> None:
    assert "metadata" in attrs, sorted(attrs)
    assert "langfuse.observation.metadata" in attrs, sorted(attrs)
    observation: Final = json.loads(str(attrs["langfuse.observation.metadata"]))
    assert {key: observation.get(key) for key in expected} == dict(expected), observation
    assert not [key for key, value in observation.items() if value is None], observation
    if spend_logs_metadata is not None:
        assert observation["spend_logs_metadata"] == dict(spend_logs_metadata), observation
    flattened: Final = {
        attribute: attrs.get(attribute)
        for attribute in (f"langfuse.trace.metadata.{field}" for field in _IDENTITIES)
        if attribute in attrs
    }
    assert flattened == {f"langfuse.trace.metadata.{field}": value for field, value in expected.items()}, attrs


def _chat_identity_fields(identity: Mapping[str, str | None]) -> dict[str, object]:
    return {
        "user_api_key_alias": identity["key_alias"],
        **({"user_api_key_team_id": identity["team_id"]} if identity["team_id"] else {}),
        **({"user_api_key_team_alias": identity["team_alias"]} if identity["team_alias"] else {}),
        **({"user_api_key_end_user_id": identity["end_user"]} if identity["end_user"] else {}),
    }


def test_langfuse_otel_emits_request_metadata_under_langfuse_observation_and_trace_keys(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "lf-meta-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "user": identity["end_user"],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "langfuse-echo"

        attrs: Final = _observation_span(rig.collector, response.json()["id"])
        _assert_identity(attrs, _chat_identity_fields(identity), spend_logs_metadata={"ticket": "LIT-8283"})


def test_langfuse_otel_metadata_redacts_user_api_key_fields_like_vanilla_langfuse(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "lf-redact-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {"redact_user_api_key_info": True}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(
            key_alias="lf-redact-" + uuid.uuid4().hex,
            metadata={"spend_logs_metadata": {"ticket": "LIT-8283"}},
        )
        model: Final = scenario.model(api_base=rig.provider.url + "/v1")
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text

        attrs: Final = _observation_span(rig.collector, response.json()["id"])
        assert "langfuse.observation.metadata" in attrs, sorted(attrs)
        observation: Final = json.loads(str(attrs["langfuse.observation.metadata"]))
        assert not [name for name in observation if name.startswith("user_api_key")], observation
        assert not [key_ for key_, value in observation.items() if value is None], observation
        assert observation["spend_logs_metadata"] == {"ticket": "LIT-8283"}, observation
        assert not [name for name in attrs if name.startswith("langfuse.trace.metadata.")], sorted(attrs)


def test_langfuse_otel_metadata_reaches_langfuse_for_openai_sdk_stream(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-sdkstream-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider)
        with OpenAI(
            base_url=str(rig.candidate.client.base_url).rstrip("/") + "/v1",
            api_key=key,
            http_client=httpx.Client(trust_env=False, timeout=30),
        ) as client:
            chunks: Final = tuple(
                client.chat.completions.create(
                    model=identity["model"],
                    messages=[{"role": "user", "content": marker}],
                    stream=True,
                    user=identity["end_user"],
                    extra_body={"cache": {"no-cache": True}},
                )
            )
        assert chunks, "stream produced no chunks"
        response_id: Final = chunks[-1].id
        attrs: Final = _observation_span(rig.collector, response_id)
        _assert_identity(attrs, _chat_identity_fields(identity), spend_logs_metadata={"ticket": "LIT-8283"})


async def test_langfuse_otel_metadata_reaches_langfuse_for_openai_sdk_async(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-sdkasync-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider)
        async with AsyncOpenAI(
            base_url=str(rig.candidate.client.base_url).rstrip("/") + "/v1",
            api_key=key,
            http_client=httpx.AsyncClient(trust_env=False, timeout=30),
        ) as client:
            response: Final = await client.chat.completions.create(
                model=identity["model"],
                messages=[{"role": "user", "content": marker}],
                user=identity["end_user"],
                extra_body={"cache": {"no-cache": True}},
            )
        attrs: Final = _observation_span(rig.collector, response.id)
        _assert_identity(attrs, _chat_identity_fields(identity), spend_logs_metadata={"ticket": "LIT-8283"})


def test_langfuse_otel_metadata_reaches_langfuse_for_anthropic_sdk_messages(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-anthropic-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider)
        model: Final = scenario.model(model="anthropic/claude-sonnet-4-6", api_base=rig.provider.url)
        with Anthropic(
            base_url=str(rig.candidate.client.base_url).rstrip("/"),
            api_key=key,
            http_client=httpx.Client(trust_env=False, timeout=30),
        ) as client:
            message: Final = client.messages.create(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": marker}],
                extra_body={"litellm_metadata": {"tags": ["lf8283"]}, "metadata": {"user_id": identity["end_user"]}},
            )
        attrs: Final = _observation_span(rig.collector, message.id)
        _assert_identity(attrs, _chat_identity_fields(identity), spend_logs_metadata={"ticket": "LIT-8283"})


async def test_langfuse_otel_metadata_reaches_langfuse_for_anthropic_sdk_stream(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "lf-anthstream-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider)
        model: Final = scenario.model(model="anthropic/claude-sonnet-4-6", api_base=rig.provider.url)
        async with (
            AsyncAnthropic(
                base_url=str(rig.candidate.client.base_url).rstrip("/"),
                api_key=key,
                http_client=httpx.AsyncClient(trust_env=False, timeout=30),
            ) as client,
            client.messages.stream(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": marker}],
                extra_body={"metadata": {"user_id": identity["end_user"]}},
            ) as stream,
        ):
            final: Final = await stream.get_final_message()
        attrs: Final = _observation_span(rig.collector, final.id)
        _assert_identity(attrs, _chat_identity_fields(identity), spend_logs_metadata={"ticket": "LIT-8283"})


def test_langfuse_otel_metadata_reaches_langfuse_for_responses_api(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-responses-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider)
        client: Final = OpenAI(
            base_url=str(rig.candidate.client.base_url).rstrip("/") + "/v1",
            api_key=key,
            http_client=httpx.Client(trust_env=False, timeout=30),
        )
        response: Final = client.responses.create(
            model=identity["model"],
            input=marker,
            user=identity["end_user"],
            extra_body={"cache": {"no-cache": True}},
        )
        attrs: Final = _observation_span(rig.collector, response.id)
        _assert_identity(attrs, _chat_identity_fields(identity), spend_logs_metadata={"ticket": "LIT-8283"})


def test_langfuse_otel_metadata_reaches_langfuse_for_responses_sse_stream(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-respstream-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider)
        with rig.candidate.client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": identity["model"],
                "input": marker,
                "user": identity["end_user"],
                "stream": True,
                "cache": {"no-cache": True},
            },
            headers={"Authorization": f"Bearer {key}"},
        ) as stream:
            events: Final = tuple(
                json.loads(line.removeprefix("data: "))
                for line in stream.iter_lines()
                if line.startswith("data: ") and line != "data: [DONE]"
            )
        response_id: Final = next(
            event["response_id"] for event in events if event.get("type") == "response.output_text.delta"
        )
        attrs: Final = _observation_span(rig.collector, response_id)
        _assert_identity(attrs, _chat_identity_fields(identity), spend_logs_metadata={"ticket": "LIT-8283"})


def test_langfuse_otel_caller_trace_metadata_survives_flattened_identity(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-deploy-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "user": identity["end_user"],
                "metadata": {"trace_metadata": {"deploy": "blue"}, "tags": ["lf8283"]},
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        attrs: Final = _observation_span(rig.collector, response.json()["id"])
        assert json.loads(str(attrs["langfuse.trace.metadata"])) == {"deploy": "blue"}, attrs
        _assert_identity(attrs, _chat_identity_fields(identity), spend_logs_metadata={"ticket": "LIT-8283"})


def test_langfuse_otel_failure_span_carries_request_metadata(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-fail-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": f"{marker} scripted -fail- upstream"}],
                "user": identity["end_user"],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code >= 500, response.text
        attrs: Final = _marked_span(rig.collector, marker, "langfuse.observation.")
        _assert_identity(attrs, _chat_identity_fields(identity), spend_logs_metadata={"ticket": "LIT-8283"})


def test_langfuse_otel_teamless_key_emits_no_team_identity(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-noteam-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider, team=False)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        attrs: Final = _observation_span(rig.collector, response.json()["id"])
        assert "langfuse.observation.metadata" in attrs, sorted(attrs)
        observation: Final = json.loads(str(attrs["langfuse.observation.metadata"]))
        for field in ("user_api_key_team_id", "user_api_key_team_alias", "user_api_key_end_user_id"):
            assert field not in observation, observation
        assert not [key_ for key_, value in observation.items() if value is None], observation
        assert {
            attribute: attrs.get(attribute)
            for attribute in (f"langfuse.trace.metadata.{field}" for field in _IDENTITIES)
            if attribute in attrs
        } == {"langfuse.trace.metadata.user_api_key_alias": identity["key_alias"]}, attrs


def test_langfuse_otel_teamless_key_keeps_caller_trace_metadata(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-noteam-meta-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider, team=False)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "metadata": {"trace_metadata": {"deploy": "blue"}},
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        attrs: Final = _observation_span(rig.collector, response.json()["id"])
        assert json.loads(str(attrs["langfuse.trace.metadata"])) == {"deploy": "blue"}, attrs
        assert "langfuse.trace.metadata.user_api_key_team_id" not in attrs, attrs
        assert "langfuse.trace.metadata.user_api_key_team_alias" not in attrs, attrs
        assert attrs.get("langfuse.trace.metadata.user_api_key_alias") == identity["key_alias"], attrs


def test_langfuse_otel_spend_logs_metadata_roundtrips_oversized_and_nonstring_values(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "lf-bigmeta-" + uuid.uuid4().hex
    spend_logs_metadata: Final = {
        "blob": "x" * 5000,
        "count": 7,
        "steps": ["a", "b"],
        "nested": {"inner": {"leaf": True}},
    }
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider, spend_logs_metadata=spend_logs_metadata)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "user": identity["end_user"],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        attrs: Final = _observation_span(rig.collector, response.json()["id"])
        observation: Final = json.loads(str(attrs["langfuse.observation.metadata"]))
        assert observation["spend_logs_metadata"] == dict(spend_logs_metadata), observation
        _assert_identity(attrs, _chat_identity_fields(identity))


def test_langfuse_otel_unauthenticated_request_leaks_no_identity(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-unauth-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider)
        denied: Final = rig.candidate.client.post(
            "/v1/chat/completions",
            json={
                "model": identity["model"],
                "messages": [{"role": "user", "content": f"{marker}-denied"}],
            },
            headers={},
        )
        assert denied.status_code == 401, denied.text
        watch: Final = _watched_spans(rig.collector)
        first: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert first.status_code == 200, first.text
        first_id: Final = first.json()["id"]
        eventually(watch, lambda spans: sum(_span_response_id(s) == first_id for s in spans) == 1, seconds=25)
        second: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker + "-settle"}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert second.status_code == 200, second.text
        second_id: Final = second.json()["id"]
        settled: Final = eventually(
            watch, lambda spans: sum(_span_response_id(s) == second_id for s in spans) == 1, seconds=25
        )
        denied_spans: Final = tuple(span for span in settled if f"{marker}-denied" in json.dumps(span, default=str))
        assert len(denied_spans) == 1, [s.get("llm.response.id") for s in settled]
        denied_span: Final = denied_spans[0]
        assert not [name for name in denied_span if name.startswith("langfuse.trace.metadata.user_api_key")], (
            denied_span
        )
        leaked: Final = json.loads(str(denied_span.get("langfuse.observation.metadata", "{}")))
        for field in (
            "user_api_key_alias",
            "user_api_key_team_id",
            "user_api_key_team_alias",
            "user_api_key_end_user_id",
        ):
            assert field not in leaked, leaked


def test_langfuse_otel_repeated_identical_requests_emit_one_span_each(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-repeat-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider)
        watch: Final = _watched_spans(rig.collector)
        responses: Final = tuple(
            rig.candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": identity["model"],
                    "messages": [{"role": "user", "content": marker}],
                    "cache": {"no-cache": True},
                },
                key=key,
            )
            for _ in range(3)
        )
        assert all(response.status_code == 200 for response in responses), [r.text for r in responses]
        response_ids: Final = tuple(response.json()["id"] for response in responses)
        assert len(set(response_ids)) == 3, response_ids
        settled: Final = eventually(
            watch,
            lambda spans: all(sum(_span_response_id(s) == rid for s in spans) == 1 for rid in response_ids),
            seconds=40,
        )
        for rid in response_ids:
            assert sum(_span_response_id(s) == rid for s in settled) == 1


def test_langfuse_otel_sink_outage_mid_burst_lands_every_response_id_once(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-chaos-outage-" + uuid.uuid4().hex
    outage: Final = threading.Event()
    rejected: Final = []  # mutable-ok: the wire server thread records each 503 it serves

    def sink_reply(request: Request) -> Reply:
        if outage.is_set():
            rejected.append(1)
            return Reply(status=503, body=b'{"error": "sink outage"}')
        return Reply()

    with (
        _langfuse_rig(gateway, tmp_path, {}, marker, sink_reply=sink_reply) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider)
        anthropic_model: Final = scenario.model(model="anthropic/claude-sonnet-4-6", api_base=rig.provider.url)
        watch: Final = _watched_spans(rig.collector)

        def call(index: int) -> str:
            path: Final = ("/v1/chat/completions", "/v1/messages", "/v1/responses")[index % 3]
            streaming: Final = index % 2 == 0
            body: Final = {
                "/v1/chat/completions": {
                    "model": identity["model"],
                    "messages": [{"role": "user", "content": f"{marker}-{index}"}],
                    "stream": streaming,
                    "cache": {"no-cache": True},
                },
                "/v1/messages": {
                    "model": anthropic_model,
                    "messages": [{"role": "user", "content": f"{marker}-{index}"}],
                    "max_tokens": 64,
                    "stream": streaming,
                },
                "/v1/responses": {"model": identity["model"], "input": f"{marker}-{index}", "stream": streaming},
            }[path]
            if streaming:
                with rig.candidate.client.stream(
                    "POST", path, json=body, headers={"Authorization": f"Bearer {key}"}
                ) as stream:
                    assert stream.status_code == 200, f"burst {index}: {stream.status_code}"
                    events: Final = tuple(
                        line.removeprefix("data: ")
                        for line in stream.iter_lines()
                        if line.startswith("data: ") and not line.endswith("[DONE]")
                    )
                payloads: Final = tuple(json.loads(line) for line in events)
                found: Final = next(
                    payload.get("response_id") or (payload.get("message") or {}).get("id") or payload.get("id")
                    for payload in payloads
                    if payload.get("response_id") or payload.get("message") or payload.get("id")
                )
                return str(found)
            response: Final = rig.candidate.client.post(path, json=body, headers={"Authorization": f"Bearer {key}"})
            assert response.status_code == 200, f"burst {index}: {response.text}"
            return response.json()["id"]

        with ThreadPoolExecutor(max_workers=30) as pool:
            futures: Final = tuple(pool.submit(call, index) for index in range(30))
            while sum(f.done() for f in futures) < 10:
                wait(futures, timeout=0.05)
            outage.set()
            eventually(
                lambda: (sum(f.done() for f in futures), len(rejected)),
                lambda progress: progress[0] >= 25 and progress[1] >= 1,
                seconds=20,
            )
            outage.clear()
            response_ids: Final = tuple(f.result() for f in futures)
        assert len(set(response_ids)) == 30, response_ids
        settled: Final = eventually(
            watch,
            lambda spans: all(sum(_span_response_id(s) == rid for s in spans) == 1 for rid in response_ids),
            seconds=80,
        )
        assert sorted(_span_response_id(span) for span in settled if _span_response_id(span) in response_ids) == sorted(
            response_ids
        )
        assert len(rejected) >= 1, "sink outage never served a 503"


def test_langfuse_otel_worker_kill_mid_burst_keeps_serving_and_exports(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-chaos-kill-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider)
        watch: Final = _watched_spans(rig.collector)

        def call(index: int) -> tuple[int, str]:
            try:
                response: Final = rig.candidate.client.post(
                    "/v1/chat/completions",
                    json={
                        "model": identity["model"],
                        "messages": [{"role": "user", "content": f"{marker}-{index}"}],
                        "cache": {"no-cache": True},
                    },
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=30,
                )
            except httpx.HTTPError as error:
                return 0, repr(error)
            return response.status_code, response.json()["id"] if response.status_code == 200 else response.text

        workers: Final = tuple(
            child
            for child in psutil.Process(rig.proxy.process.pid).children(recursive=True)
            if "spawn_main" in " ".join(child.cmdline())
        )
        assert len(workers) >= 2, f"expected multiple uvicorn workers, got {workers!r}"
        victim: Final = workers[0]
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures: Final = tuple(pool.submit(call, index) for index in range(20))
            while sum(f.done() for f in futures) < 5:
                wait(futures, timeout=0.05)
            victim.kill()
            outcomes: Final = tuple(f.result() for f in futures)
        assert not victim.is_running() or victim.status() == psutil.STATUS_ZOMBIE
        succeeded: Final = tuple(identifier for status, identifier in outcomes if status == 200)
        assert succeeded, outcomes
        post_kill: Final = tuple(call(20 + index) for index in range(5))
        post_kill_ids: Final = tuple(identifier for _, identifier in post_kill)
        assert all(status == 200 for status, _ in post_kill), post_kill
        settled: Final = eventually(
            watch,
            lambda spans: all(sum(_span_response_id(s) == rid for s in spans) == 1 for rid in post_kill_ids),
            seconds=60,
        )
        for rid in succeeded:
            assert sum(_span_response_id(s) == rid for s in settled) <= 1, rid


def test_langfuse_otel_slow_sink_does_not_deadlock_exports(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-chaos-slow-" + uuid.uuid4().hex

    def sink_reply(request: Request) -> Reply:
        gate: Final = threading.Event()
        threading.Timer(2, gate.set).start()
        return Reply(chunks=(b"{}",), gate_after_first=gate)

    with (
        _langfuse_rig(gateway, tmp_path, {}, marker, sink_reply=sink_reply) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider)
        watch: Final = _watched_spans(rig.collector)
        responses: Final = tuple(
            rig.candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": identity["model"],
                    "messages": [{"role": "user", "content": f"{marker}-{index}"}],
                    "cache": {"no-cache": True},
                },
                key=key,
            )
            for index in range(10)
        )
        assert all(response.status_code == 200 for response in responses), [r.text for r in responses]
        response_ids: Final = tuple(response.json()["id"] for response in responses)
        settled: Final = eventually(
            watch,
            lambda spans: all(sum(_span_response_id(s) == rid for s in spans) == 1 for rid in response_ids),
            seconds=60,
        )
        assert sorted(_span_response_id(span) for span in settled if _span_response_id(span) in response_ids) == sorted(
            response_ids
        )


def test_langfuse_otel_v2_mapper_emits_request_metadata_and_identity(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-v2-" + uuid.uuid4().hex
    with (
        _langfuse_rig(
            gateway,
            tmp_path,
            {},
            marker,
            overrides={"LITELLM_OTEL_V2": "1"},
        ) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "user": identity["end_user"],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        watch: Final = _watched_spans(rig.collector)
        observed: Final = eventually(
            watch,
            lambda spans: sum(s.get("gen_ai.response.id") == response.json()["id"] for s in spans) == 1,
            seconds=25,
        )
        attrs: Final = next(s for s in observed if s.get("gen_ai.response.id") == response.json()["id"])
        assert attrs.get("langfuse.trace.metadata.team_id") == identity["team_id"], attrs
        assert attrs.get("langfuse.trace.metadata.team_alias") == identity["team_alias"], attrs
        assert "langfuse.observation.metadata" in attrs, sorted(attrs)
        observation: Final = json.loads(str(attrs["langfuse.observation.metadata"]))
        expected: Final = _chat_identity_fields(identity)
        assert {key: observation.get(key) for key in expected} == dict(expected), observation
        assert observation["spend_logs_metadata"] == {"ticket": "LIT-8283"}, observation
        assert not [key_ for key_, value in observation.items() if value is None], observation
        flattened: Final = {
            attribute: attrs.get(attribute)
            for attribute in (f"langfuse.trace.metadata.{field}" for field in _IDENTITIES)
            if attribute in attrs
        }
        assert flattened == {f"langfuse.trace.metadata.{field}": value for field, value in expected.items()}, attrs


def test_langfuse_otel_v2_mapper_redacts_user_api_key_fields(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-v2-redact-" + uuid.uuid4().hex
    with (
        _langfuse_rig(
            gateway,
            tmp_path,
            {"redact_user_api_key_info": True},
            marker,
            overrides={"LITELLM_OTEL_V2": "1"},
        ) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(
            key_alias="lf-v2-redact-" + uuid.uuid4().hex,
            metadata={"spend_logs_metadata": {"ticket": "LIT-8283"}},
        )
        model: Final = scenario.model(api_base=rig.provider.url + "/v1")
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        watch: Final = _watched_spans(rig.collector)
        observed: Final = eventually(
            watch,
            lambda spans: sum(s.get("gen_ai.response.id") == response.json()["id"] for s in spans) == 1,
            seconds=25,
        )
        attrs: Final = next(s for s in observed if s.get("gen_ai.response.id") == response.json()["id"])
        assert "langfuse.observation.metadata" in attrs, sorted(attrs)
        observation: Final = json.loads(str(attrs["langfuse.observation.metadata"]))
        assert not [name for name in observation if name.startswith("user_api_key")], observation
        assert not [key_ for key_, value in observation.items() if value is None], observation
        assert observation["spend_logs_metadata"] == {"ticket": "LIT-8283"}, observation
        assert not [name for name in attrs if name.startswith("langfuse.trace.metadata.user_api_key_")], sorted(attrs)


def test_langfuse_otel_v2_mapper_teamless_key_emits_no_team_identity(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-v2-noteam-" + uuid.uuid4().hex
    with (
        _langfuse_rig(
            gateway,
            tmp_path,
            {},
            marker,
            overrides={"LITELLM_OTEL_V2": "1"},
        ) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = _scenario_identity(scenario, rig.provider, team=False)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        watch: Final = _watched_spans(rig.collector)
        observed: Final = eventually(
            watch,
            lambda spans: sum(s.get("gen_ai.response.id") == response.json()["id"] for s in spans) == 1,
            seconds=25,
        )
        attrs: Final = next(s for s in observed if s.get("gen_ai.response.id") == response.json()["id"])
        assert "langfuse.observation.metadata" in attrs, sorted(attrs)
        observation: Final = json.loads(str(attrs["langfuse.observation.metadata"]))
        for field in ("user_api_key_team_id", "user_api_key_team_alias", "user_api_key_end_user_id"):
            assert field not in observation, observation
        assert not [key_ for key_, value in observation.items() if value is None], observation
        assert {
            attribute: attrs.get(attribute)
            for attribute in (f"langfuse.trace.metadata.{field}" for field in _IDENTITIES)
            if attribute in attrs
        } == {"langfuse.trace.metadata.user_api_key_alias": identity["key_alias"]}, attrs
