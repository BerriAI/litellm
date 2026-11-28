import base64
import json
import re
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, gateway_from_environment, object_value
from integration._support.process import OwnedProxy, WorkerHealthcheck, graceful_stop_seconds, owned_proxy_process
from integration._support.responses_stream import completed, created, data_frames, frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 180), pytest.mark.shares_worker_state]

_API_KEY: Final = "synthetic-tool-argument-key"
_EVENT_TOKEN: Final = re.compile(rb"event-([A-Za-z0-9_-]+)")
_EVENT: Final = TypeAdapter(dict[str, JsonValue])
_OPENAI_MODEL: Final = "tool-argument-strings-openai"
_FIREWORKS_MODEL: Final = "tool-argument-strings-fireworks"
_DEPLOYMENTS: Final = (
    (_OPENAI_MODEL, "openai/gpt-6.2-sol"),
    (_FIREWORKS_MODEL, "fireworks_ai/accounts/fireworks/models/kimi-k3p1-pro"),
)
_MODELS: Final = tuple(pytest.param(name, id=name.removeprefix("tool-argument-strings-")) for name, _ in _DEPLOYMENTS)
_STRING_FIELDS: Final = (
    pytest.param("response.function_call_arguments.delta", "delta", "fc_1", id="function_call_arguments.delta"),
    pytest.param("response.function_call_arguments.done", "arguments", "fc_1", id="function_call_arguments.done"),
    pytest.param("response.mcp_call_arguments.delta", "delta", "mcp_1", id="mcp_call_arguments.delta"),
    pytest.param("response.mcp_call_arguments.done", "arguments", "mcp_1", id="mcp_call_arguments.done"),
    pytest.param("response.mcp_call.arguments.delta", "delta", "mcp_1", id="mcp_call.arguments.delta"),
    pytest.param("response.mcp_call.arguments.done", "arguments", "mcp_1", id="mcp_call.arguments.done"),
    pytest.param("response.custom_tool_call_input.delta", "delta", "ctc_1", id="custom_tool_call_input.delta"),
    pytest.param("response.custom_tool_call_input.done", "input", "ctc_1", id="custom_tool_call_input.done"),
)


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: OwnedProxy
    upstream: Wire


def _encoded(event: Mapping[str, JsonValue]) -> str:
    return base64.urlsafe_b64encode(json.dumps(event).encode()).decode().rstrip("=")


def _decoded(token: bytes) -> dict[str, JsonValue]:
    return _EVENT.validate_json(base64.urlsafe_b64decode(token + b"=" * (-len(token) % 4)))


def _respond(request: Request) -> Reply:
    found: Final = _EVENT_TOKEN.search(request.body)
    assert request.method == "POST" and request.target.endswith("/responses") and found, (request.target, request.body)
    identity: Final = "resp_" + uuid.uuid4().hex
    stream: Final = (frame(created(identity)), frame(_decoded(found.group(1))), frame(completed(identity, "done")))
    return Reply(content_type="text/event-stream", chunks=stream)


def _config(upstream_url: str, directory: Path) -> Path:
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    router_settings: Final = object_value(stock.get("router_settings") or {})
    path: Final = directory / "responses-tool-argument-strings.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **stock,
                "model_list": [
                    {
                        "model_name": name,
                        "litellm_params": {"model": model, "api_base": upstream_url, "api_key": _API_KEY},
                    }
                    for name, model in _DEPLOYMENTS
                ],
                "router_settings": {**router_settings, "num_retries": 0},
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("responses-tool-argument-strings")
    with (
        wire_server(_respond) as upstream,
        gateway_from_environment() as shared,
        owned_proxy_process(
            shared,
            directory,
            {},
            config=_config(upstream.url, directory),
            workers=2,
            healthcheck=WorkerHealthcheck.LENIENT,
        ) as owned,
    ):
        yield _Rig(owned, upstream)


def _tool_event(kind: str, field: str, item_id: str, value: JsonValue) -> dict[str, JsonValue]:
    return {"type": kind, "sequence_number": 1, "item_id": item_id, "output_index": 0, field: value}


def _streamed_event(rig: _Rig, model: str, event: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.proxy.gateway.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": f"marker-{marker} event-{_encoded(event)}",
            "stream": True,
            "num_retries": 0,
            "cache": {"no-cache": True},
        },
    )
    assert response.status_code == 200, response.text
    upstream_calls: Final = tuple(request for request in rig.upstream.drain() if marker.encode() in request.body)
    assert len(upstream_calls) == 1, [request.target for request in upstream_calls]
    frames: Final = data_frames(response.text)
    assert frames[-1]["type"] == "response.completed", response.text
    matching: Final = tuple(streamed for streamed in frames if streamed["type"] == event["type"])
    assert len(matching) == 1, response.text
    return matching[0]


def _assert_arrived_unchanged(sent: Mapping[str, JsonValue], received: Mapping[str, JsonValue]) -> None:
    names: Final = tuple(name for name in sent if name != "sequence_number")
    assert {name: received.get(name) for name in names} == {name: sent[name] for name in names}, received


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize(("kind", "field", "item_id"), _STRING_FIELDS)
@pytest.mark.parametrize(
    "value", [pytest.param(42, id="int"), pytest.param({"q": 1}, id="object"), pytest.param(None, id="null")]
)
def test_non_string_tool_argument_field_reaches_the_client_as_a_json_string(
    rig: _Rig, model: str, kind: str, field: str, item_id: str, value: JsonValue
) -> None:
    received: Final = _streamed_event(rig, model, _tool_event(kind, field, item_id, value))
    assert field in received, received
    text: Final = received[field]
    assert isinstance(text, str), received
    assert (text == "") if value is None else (json.loads(text) == value), received


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize(("kind", "field", "item_id"), _STRING_FIELDS)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param('{"city": "Paris"}', id="complete-json"),
        pytest.param('{"ci', id="partial-json"),
        pytest.param("", id="empty"),
        pytest.param('"42"', id="json-string-literal"),
    ],
)
def test_string_tool_argument_field_arrives_unchanged(
    rig: _Rig, model: str, kind: str, field: str, item_id: str, value: str
) -> None:
    sent: Final = _tool_event(kind, field, item_id, value)
    received: Final = _streamed_event(rig, model, sent)
    _assert_arrived_unchanged(sent, received)


@pytest.mark.parametrize("model", _MODELS)
def test_unrelated_text_delta_arrives_unchanged(rig: _Rig, model: str) -> None:
    sent: Final[dict[str, JsonValue]] = {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": '{"q": 1} 雪',
    }
    received: Final = _streamed_event(rig, model, sent)
    _assert_arrived_unchanged(sent, received)
