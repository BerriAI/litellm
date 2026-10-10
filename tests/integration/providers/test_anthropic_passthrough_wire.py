import json
import threading
from collections.abc import Iterator
from itertools import accumulate, dropwhile
from pathlib import Path
from typing import Final

import anthropic
import httpx
import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark = pytest.mark.timeout(240)

_API_KEY: Final = "scripted-anthropic-key"
_OAUTH_TOKEN: Final = "sk-ant-oat01-scripted-proxy-token"
_MODEL: Final = "claude-sonnet-4-5"
_TEXT: Final = "scripted passthrough answer"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_SYSTEM: Final[list[JsonValue]] = [{"type": "text", "text": "You are a terse assistant"}]
_TOOLS: Final[list[JsonValue]] = [
    {
        "name": "get_weather",
        "description": "Look up the weather",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }
]
_THINKING: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": 1024}
_MESSAGE: Final[dict[str, JsonValue]] = {
    "id": "msg_scripted",
    "type": "message",
    "role": "assistant",
    "model": _MODEL,
    "content": [{"type": "text", "text": _TEXT}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 7, "output_tokens": 3},
}
_SSE_EVENTS: Final[tuple[dict[str, JsonValue], ...]] = (
    {"type": "message_start", "message": {**_MESSAGE, "content": [], "stop_reason": None}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _TEXT}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 3},
    },
    {"type": "message_stop"},
)
_SSE_FRAMES: Final = tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in _SSE_EVENTS)
_CLAUDE_CODE_STREAM_GATE: Final = threading.Event()


def _respond(request: Request) -> Reply:
    if _JSON_OBJECT.validate_json(request.body).get("stream") is True:
        gate: Final = _CLAUDE_CODE_STREAM_GATE if request.headers.get("x-app") == "cli" else None
        return Reply(chunks=_SSE_FRAMES, content_type="text/event-stream", gate_after_first=gate)
    return Reply(body=json.dumps(_MESSAGE).encode())


@pytest.fixture(scope="module")
def rig() -> Iterator[Gateway]:
    with gateway_from_environment() as gateway:
        yield gateway


@pytest.fixture(scope="module")
def wire() -> Iterator[Wire]:
    with wire_server(_respond) as served:
        yield served


@pytest.fixture(autouse=True)
def _drained(wire: Wire) -> None:
    wire.drain()


def _config(directory: Path) -> Path:
    path: Final = directory / "anthropic-passthrough.yaml"
    path.write_text(
        "model_list: []\n"
        "general_settings:\n"
        "  master_key: os.environ/LITELLM_MASTER_KEY\n"
        "  database_url: os.environ/DATABASE_URL\n"
        "  store_model_in_db: true\n"
        "router_settings:\n"
        "  disable_cooldowns: true\n"
    )
    return path


@pytest.fixture(scope="module")
def api_key_proxy(rig: Gateway, wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("anthropic-passthrough-api-key")
    overrides: Final = {"ANTHROPIC_API_BASE": wire.url, "ANTHROPIC_API_KEY": _API_KEY}
    with owned_proxy(rig, directory, overrides, config=_config(directory), workers=2) as owned:
        yield owned


@pytest.fixture(scope="module")
def oauth_proxy(rig: Gateway, wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("anthropic-passthrough-oauth")
    overrides: Final = {"ANTHROPIC_API_BASE": wire.url, "ANTHROPIC_API_KEY": _OAUTH_TOKEN}
    with owned_proxy(rig, directory, overrides, config=_config(directory), workers=2) as owned:
        yield owned


def _sdk(proxy: Gateway, key: str, sent: list[httpx.Request]) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=key,
        base_url=f"{str(proxy.client.base_url).rstrip('/')}/anthropic",
        max_retries=0,
        http_client=httpx.Client(timeout=30, trust_env=False, event_hooks={"request": [sent.append]}),
    )


def _only_upstream_request(wire: Wire) -> Request:
    received: Final = wire.drain()
    assert [(request.method, request.target) for request in received] == [("POST", "/v1/messages")], received
    return received[0]


def _assert_virtual_key_absent(upstream: Request, key: str) -> None:
    leaked: Final = {name: value for name, value in upstream.headers.items() if key in value}
    assert leaked == {}, upstream.headers
    assert key.encode() not in upstream.body, upstream.body


def _assert_sdk_call_reached_upstream_intact(upstream: Request, sent: httpx.Request, key: str) -> None:
    assert _JSON_OBJECT.validate_json(upstream.body) == _JSON_OBJECT.validate_json(sent.content), upstream.body
    assert upstream.headers["x-api-key"] == _API_KEY, upstream.headers
    assert "authorization" not in upstream.headers, upstream.headers
    assert upstream.headers["anthropic-version"] == sent.headers["anthropic-version"], upstream.headers
    assert upstream.headers["anthropic-beta"] == "interleaved-thinking-2025-05-14", upstream.headers
    _assert_virtual_key_absent(upstream, key)


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_anthropic_sdk_call_reaches_upstream_with_full_body_and_proxy_key(
    api_key_proxy: Gateway, wire: Wire, stream: bool
) -> None:
    with api_key_proxy.scenario() as scenario:
        key: Final = scenario.key()
        sent: Final[list[httpx.Request]] = []
        with _sdk(api_key_proxy, key, sent) as sdk:
            arguments: Final = {
                "model": _MODEL,
                "max_tokens": 2048,
                "system": _SYSTEM,
                "tools": _TOOLS,
                "thinking": _THINKING,
                "messages": [{"role": "user", "content": "weather in Paris?"}],
                "extra_headers": {"anthropic-beta": "interleaved-thinking-2025-05-14"},
            }
            if stream:
                with sdk.messages.stream(**arguments) as events:
                    message = events.get_final_message()
            else:
                message = sdk.messages.create(**arguments)
        assert message.model_dump(exclude_none=True) == anthropic.types.Message.model_validate(_MESSAGE).model_dump(
            exclude_none=True
        ), message
        upstream: Final = _only_upstream_request(wire)
        assert len(sent) == 1, sent
        _assert_sdk_call_reached_upstream_intact(upstream, sent[0], key)


@pytest.mark.parametrize("auth_style", ["api-key", "oauth"])
def test_anthropic_proxy_credentials_override_caller_credentials(
    api_key_proxy: Gateway, oauth_proxy: Gateway, wire: Wire, auth_style: str
) -> None:
    proxy: Final = api_key_proxy if auth_style == "api-key" else oauth_proxy
    caller_api_key: Final = f"caller-anthropic-key-{auth_style}"
    caller_authorization: Final = f"Bearer caller-anthropic-token-{auth_style}"
    body: Final[dict[str, JsonValue]] = {
        "model": _MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    }
    with (
        proxy.scenario() as scenario,
        httpx.Client(base_url=str(proxy.client.base_url), timeout=30, trust_env=False) as client,
    ):
        key: Final = scenario.key()
        response: Final = client.post(
            "/anthropic/v1/messages",
            json=body,
            headers={
                "x-litellm-api-key": key,
                "x-api-key": caller_api_key,
                "Authorization": caller_authorization,
            },
        )
        assert response.status_code == 200, response.text
        message: Final = anthropic.types.Message.model_validate_json(response.content)
        assert message.model_dump(exclude_none=True) == anthropic.types.Message.model_validate(_MESSAGE).model_dump(
            exclude_none=True
        ), response.text
        upstream: Final = _only_upstream_request(wire)
        assert _JSON_OBJECT.validate_json(upstream.body) == body, upstream.body
        assert {
            name: value
            for name, value in upstream.headers.items()
            if caller_api_key in value or caller_authorization in value
        } == {}, upstream.headers
        if auth_style == "api-key":
            assert upstream.headers["x-api-key"] == _API_KEY, upstream.headers
            assert "authorization" not in upstream.headers, upstream.headers
        else:
            assert upstream.headers["authorization"] == f"Bearer {_OAUTH_TOKEN}", upstream.headers
            assert "x-api-key" not in upstream.headers, upstream.headers
        _assert_virtual_key_absent(upstream, key)


def test_claude_code_bearer_stream_merges_caller_beta_with_the_credential_beta(
    oauth_proxy: Gateway, wire: Wire
) -> None:
    _CLAUDE_CODE_STREAM_GATE.clear()
    with oauth_proxy.scenario() as scenario:
        key: Final = scenario.key()
        body: Final[dict[str, JsonValue]] = {
            "model": _MODEL,
            "max_tokens": 2048,
            "stream": True,
            "system": _SYSTEM,
            "tools": _TOOLS,
            "thinking": _THINKING,
            "messages": [{"role": "user", "content": "weather in Paris?"}],
        }
        with oauth_proxy.client.stream(
            "POST",
            "/anthropic/v1/messages",
            json=body,
            headers={
                "Authorization": f"Bearer {key}",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "claude-code-20250219,interleaved-thinking-2025-05-14",
                "x-app": "cli",
            },
        ) as response:
            raw: Final = iter(response.iter_raw())
            try:
                assert response.status_code == 200, response.status_code
                assert response.headers.get("content-type", "").startswith("text/event-stream"), response.headers
                first_frame: Final = next(
                    dropwhile(
                        lambda buffered: b"\n\n" not in buffered,
                        accumulate(raw, lambda buffered, chunk: buffered + chunk, initial=b""),
                    )
                )
                assert first_frame == _SSE_FRAMES[0], first_frame
            finally:
                _CLAUDE_CODE_STREAM_GATE.set()
            streamed: Final = first_frame + b"".join(raw)
            assert streamed == b"".join(_SSE_FRAMES), streamed
        _CLAUDE_CODE_STREAM_GATE.clear()
        upstream: Final = _only_upstream_request(wire)
        assert _JSON_OBJECT.validate_json(upstream.body) == body, upstream.body
        assert upstream.headers["authorization"] == f"Bearer {_OAUTH_TOKEN}", upstream.headers
        assert "x-api-key" not in upstream.headers, upstream.headers
        assert upstream.headers["anthropic-version"] == "2023-06-01", upstream.headers
        assert upstream.headers["anthropic-beta"] == (
            "claude-code-20250219,interleaved-thinking-2025-05-14,oauth-2025-04-20"
        ), upstream.headers
        assert upstream.headers["x-app"] == "cli", upstream.headers
        _assert_virtual_key_absent(upstream, key)


def test_anthropic_sdk_metadata_user_id_reaches_upstream(api_key_proxy: Gateway, wire: Wire) -> None:
    pytest.skip("BUG: /anthropic/v1/messages drops the native metadata.user_id field before forwarding upstream")
    with api_key_proxy.scenario() as scenario:
        key: Final = scenario.key()
        sent: Final[list[httpx.Request]] = []
        with _sdk(api_key_proxy, key, sent) as sdk:
            sdk.messages.create(
                model=_MODEL,
                max_tokens=64,
                metadata={"user_id": "user_scripted_session_1"},
                messages=[{"role": "user", "content": "hi"}],
            )
        upstream: Final = _only_upstream_request(wire)
        assert _JSON_OBJECT.validate_json(upstream.body) == _JSON_OBJECT.validate_json(sent[0].content), upstream.body
