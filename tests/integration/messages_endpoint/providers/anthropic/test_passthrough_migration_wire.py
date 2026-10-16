import json
import uuid
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Final

from integration._support.client import Gateway, JsonValue, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server

_MODEL: Final = "claude-sonnet-4-5-20250929"
_KEY: Final = "synthetic-anthropic-key"

_PROXY_CONFIG: Final = (
    "model_list: []\n"
    "general_settings:\n"
    "  master_key: os.environ/LITELLM_MASTER_KEY\n"
    "  database_url: os.environ/DATABASE_URL\n"
    "  store_model_in_db: true\n"
    "  disable_spend_logs: false\n"
    "  proxy_batch_write_at: 1\n"
)


def _message(message_id: str, model: str = _MODEL) -> dict[str, JsonValue]:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "hello test"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }


def _thinking_message(message_id: str) -> dict[str, JsonValue]:
    return {
        **_message(message_id, "claude-haiku-4-5-20251001"),
        "content": [
            {"type": "thinking", "thinking": "pondering the joke", "signature": "sig1"},
            {"type": "text", "text": "hello thinking"},
        ],
        "usage": {"input_tokens": 11, "output_tokens": 30},
    }


def _sse(event: str, payload: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def _stream_chunks(message_id: str) -> tuple[bytes, ...]:
    return (
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": _MODEL,
                    "content": [],
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            },
        ),
        _sse(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        _sse(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello stream"}},
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}},
        ),
        _sse("message_stop", {"type": "message_stop"}),
    )


def _bad_request_reply() -> Reply:
    return Reply(
        status=400,
        body=json.dumps(
            {"type": "error", "error": {"type": "invalid_request_error", "message": "messages must be objects"}}
        ).encode(),
    )


def _messages_are_objects(body: Mapping[str, JsonValue]) -> bool:
    messages: Final = body.get("messages")
    return isinstance(messages, list) and all(isinstance(message, dict) for message in messages)


_SPEND_COLUMNS: Final = (
    "SELECT request_id, status, call_type, prompt_tokens, completion_tokens, total_tokens, spend, request_tags, "
    'end_user, api_base, custom_llm_provider, model, cache_hit, ("startTime" <= "endTime") AS times_ordered '
    'FROM "LiteLLM_SpendLogs" '
)


def _spend_row(request_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(_SPEND_COLUMNS + "WHERE request_id=%s", (request_id,)),
        lambda values: len(values) == 1,
        seconds=90,
    )
    return rows[0]


def _key_spend_row(key: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            _SPEND_COLUMNS + "WHERE api_key=%s AND call_type=%s",
            (sha256(key.encode()).hexdigest(), "pass_through_endpoint"),
        ),
        lambda values: len(values) == 1,
        seconds=90,
    )
    return rows[0]


_MODEL_LIST_PROBE: Final = ("GET", "/v1/models")


def _is_model_list_probe(request: Request) -> bool:
    return (request.method, request.target) == _MODEL_LIST_PROBE


@contextmanager
def _upstream(respond: Callable[[Request], Reply]) -> Generator[Wire]:
    with wire_server(
        lambda request: (
            Reply(body=b'{"object":"list","data":[]}') if _is_model_list_probe(request) else respond(request)
        )
    ) as wire:
        yield wire


def _provider_calls(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if not _is_model_list_probe(request))


def _tags(row: Mapping[str, JsonValue]) -> list[JsonValue]:
    raw: Final = row["request_tags"]
    tags: Final = json.loads(raw) if isinstance(raw, str) else raw
    assert isinstance(tags, list), row
    return [tag for tag in tags if not (isinstance(tag, str) and tag.startswith("User-Agent: "))]


def _assert_usage_row(row: Mapping[str, JsonValue], call_type: str, tags: list[str]) -> None:
    assert row["status"] == "success", row
    assert row["call_type"] == call_type, row
    assert row["prompt_tokens"] == 11, row
    assert row["completion_tokens"] == 7, row
    assert row["total_tokens"] == 18, row
    spend: Final = row["spend"]
    assert isinstance(spend, (int, float)) and spend > 0, row
    assert _tags(row) == tags, row
    assert row["custom_llm_provider"] == "anthropic", row
    assert str(row["cache_hit"]).lower() != "true", row
    assert row["times_ordered"] is True, row


def _stream_text(gateway: Gateway, path: str, body: Mapping[str, JsonValue], key: str | None = None) -> str:
    with gateway.client.stream(
        "POST",
        f"{gateway.client.base_url}{path}",
        json=body,
        headers={"Authorization": f"Bearer {key or gateway.key}"},
    ) as stream:
        assert stream.status_code == 200, stream.read()
        return "".join(stream.iter_text())


def _owned_config(tmp_path: Path, text: str) -> Path:
    config: Final = tmp_path / "proxy_config.yaml"
    config.write_text(text)
    return config


def test_passthrough_basic_completion_spend_row_v1_messages(gateway: Gateway) -> None:
    marker: Final = "pt-basic-" + uuid.uuid4().hex
    prompt: Final = f"say hello {marker}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == _KEY
        body: Final = json.loads(request.body)
        assert body["model"] == _MODEL
        assert body["messages"] == [{"role": "user", "content": prompt}]
        assert "litellm_metadata" not in body
        return Reply(body=json.dumps(_message(f"msg_{marker}")).encode())

    with _upstream(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}],
                "litellm_metadata": {"tags": [f"{marker}-1", f"{marker}-2"], "user": f"end-user-{marker}"},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"] == f"msg_{marker}"
        assert len(_provider_calls(wire)) == 1
        row: Final = _spend_row(f"msg_{marker}")
        _assert_usage_row(row, "anthropic_messages", [f"{marker}-1", f"{marker}-2"])
        assert row["end_user"] == f"end-user-{marker}", row


def test_passthrough_streaming_spend_row_v1_messages(gateway: Gateway) -> None:
    marker: Final = "pt-stream-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.target == "/v1/messages", request.target
        body: Final = json.loads(request.body)
        assert body["model"] == _MODEL
        assert body["stream"] is True
        return Reply(content_type="text/event-stream", chunks=_stream_chunks(f"msg_{marker}"))

    with _upstream(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_KEY)
        text: Final = _stream_text(
            gateway,
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": f"say hello {marker}"}],
                "litellm_metadata": {"tags": [f"{marker}-1", f"{marker}-2"], "user": f"end-user-{marker}"},
            },
        )
        assert "hello stream" in text
        row: Final = _spend_row(f"msg_{marker}")
        _assert_usage_row(row, "anthropic_messages", [f"{marker}-1", f"{marker}-2"])
        assert row["end_user"] == f"end-user-{marker}", row


def test_passthrough_wildcard_model_strips_provider_prefix(gateway: Gateway) -> None:
    marker: Final = "pt-wildcard-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        assert body["model"] == "claude-haiku-4-5-20251001"
        return Reply(body=json.dumps(_message(f"msg_{marker}", "claude-haiku-4-5-20251001")).encode())

    with _upstream(respond) as wire, gateway.scenario() as scenario:
        created: Final = gateway.post(
            "/model/new",
            {
                "model_name": "anthropic/*",
                "litellm_params": {"model": "anthropic/*", "api_base": wire.url, "api_key": _KEY},
            },
        )
        model_info: Final = created["model_info"]
        assert isinstance(model_info, dict), created
        identity: Final = model_info["id"]
        assert isinstance(identity, str), created
        scenario.cleanups.callback(scenario.delete_model, identity)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": "anthropic/claude-haiku-4-5-20251001",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": f"hello wildcard {marker}"}],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["content"][0]["text"] == "hello test"
        assert len(_provider_calls(wire)) == 1


def test_passthrough_thinking_block_round_trips_v1_messages(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        assert body["model"] == "claude-haiku-4-5-20251001"
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 16000}
        assert body["max_tokens"] == 20000
        return Reply(body=json.dumps(_thinking_message("msg_" + uuid.uuid4().hex)).encode())

    with _upstream(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="anthropic/claude-haiku-4-5-20251001", api_base=wire.url, api_key=_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 20000,
                "thinking": {"type": "enabled", "budget_tokens": 16000},
                "messages": [{"role": "user", "content": "Just pinging with thinking enabled"}],
            },
        )
        assert response.status_code == 200, response.text
        content: Final = response.json()["content"]
        assert content[0]["type"] == "thinking"
        assert content[0]["thinking"] == "pondering the joke"


def test_passthrough_bad_request_returns_400_v1_messages(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        assert not _messages_are_objects(body), body
        return _bad_request_reply()

    with _upstream(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_KEY)
        responses: Final = tuple(
            gateway.request(
                "POST",
                "/v1/messages",
                {"model": model, "max_tokens": 10, "stream": stream, "messages": ["hi"]},
            )
            for stream in (False, True)
        )
        assert [response.status_code for response in responses] == [400, 400], [r.text for r in responses]


def test_native_anthropic_route_completion_stream_thinking_and_bad_request(gateway: Gateway, tmp_path: Path) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == _KEY
        body: Final = json.loads(request.body)
        if not _messages_are_objects(body):
            return _bad_request_reply()
        if body.get("stream") is True:
            return Reply(content_type="text/event-stream", chunks=_stream_chunks("msg_" + uuid.uuid4().hex))
        if body.get("thinking") is not None:
            return Reply(body=json.dumps(_thinking_message("msg_" + uuid.uuid4().hex)).encode())
        return Reply(body=json.dumps(_message("msg_" + uuid.uuid4().hex)).encode())

    with (
        _upstream(respond) as wire,
        owned_proxy(
            gateway,
            tmp_path,
            {"ANTHROPIC_API_BASE": wire.url, "ANTHROPIC_API_KEY": _KEY},
            config=_owned_config(tmp_path, _PROXY_CONFIG),
        ) as candidate,
    ):
        completion: Final = candidate.request(
            "POST",
            "/anthropic/v1/messages",
            {"model": _MODEL, "max_tokens": 100, "messages": [{"role": "user", "content": "say hello native"}]},
        )
        assert completion.status_code == 200, completion.text
        assert completion.json()["content"][0]["text"] == "hello test"
        thinking: Final = candidate.request(
            "POST",
            "/anthropic/v1/messages",
            {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 20000,
                "thinking": {"type": "enabled", "budget_tokens": 16000},
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        assert thinking.status_code == 200, thinking.text
        assert thinking.json()["content"][0]["type"] == "thinking"
        assert thinking.json()["content"][0]["thinking"] == "pondering the joke"
        bad: Final = tuple(
            candidate.request(
                "POST",
                "/anthropic/v1/messages",
                {"model": _MODEL, "max_tokens": 10, "stream": stream, "messages": ["hi"]},
            )
            for stream in (False, True)
        )
        assert [response.status_code for response in bad] == [400, 400], [r.text for r in bad]
        text: Final = _stream_text(
            candidate,
            "/anthropic/v1/messages",
            {
                "model": _MODEL,
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hello native stream"}],
            },
        )
        assert "hello stream" in text


def test_native_passthrough_spend_rows_record_usage_tags_and_spend(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "pt-native-" + uuid.uuid4().hex
    completion_id: Final = f"msg_{marker}_completion"
    stream_id: Final = f"msg_{marker}_stream"

    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        assert "litellm_metadata" not in body
        if body.get("stream") is True:
            return Reply(content_type="text/event-stream", chunks=_stream_chunks(stream_id))
        return Reply(body=json.dumps(_message(completion_id)).encode())

    with (
        _upstream(respond) as wire,
        owned_proxy(
            gateway,
            tmp_path,
            {"ANTHROPIC_API_BASE": wire.url, "ANTHROPIC_API_KEY": _KEY},
            config=_owned_config(tmp_path, _PROXY_CONFIG),
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        completion_key: Final = scenario.key()
        stream_key: Final = scenario.key()
        response: Final = candidate.request(
            "POST",
            "/anthropic/v1/messages",
            {
                "model": _MODEL,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Say 'hello test' and nothing else"}],
                "litellm_metadata": {"tags": [f"{marker}-1", f"{marker}-2"]},
            },
            key=completion_key,
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"] == completion_id
        text: Final = _stream_text(
            candidate,
            "/anthropic/v1/messages",
            {
                "model": _MODEL,
                "max_tokens": 10,
                "stream": True,
                "messages": [{"role": "user", "content": "Say 'hello stream test' and nothing else"}],
                "litellm_metadata": {"tags": [f"{marker}-s1", f"{marker}-s2"], "user": f"end-user-{marker}"},
            },
            key=stream_key,
        )
        assert "hello stream" in text
        completion_row: Final = _key_spend_row(completion_key)
        stream_row: Final = _key_spend_row(stream_key)
    assert completion_row["request_id"] == completion_id, completion_row
    assert stream_row["request_id"] == stream_id, stream_row
    _assert_usage_row(completion_row, "pass_through_endpoint", [f"{marker}-1", f"{marker}-2"])
    assert completion_row["api_base"] == f"{wire.url}/v1/messages", completion_row
    assert "claude" in str(completion_row["model"]), completion_row
    _assert_usage_row(stream_row, "pass_through_endpoint", [f"{marker}-s1", f"{marker}-s2"])
    assert stream_row["end_user"] == f"end-user-{marker}", stream_row


def _openai_responses_stream() -> tuple[bytes, ...]:
    response: Final[dict[str, JsonValue]] = {
        "id": "resp_pt1",
        "object": "response",
        "created_at": 1700000000,
        "model": "gpt-4o-mini",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi from openai"}],
            }
        ],
        "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
    }
    return (
        _sse(
            "response.created",
            {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
        ),
        _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": "msg_pto",
                "output_index": 0,
                "content_index": 0,
                "delta": "hi from openai",
            },
        ),
        _sse("response.completed", {"type": "response.completed", "response": response}),
    )


def _openai_chat_stream() -> tuple[bytes, ...]:
    chunk: Final[dict[str, JsonValue]] = {
        "id": "chatcmpl-pt1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o",
    }
    return (
        f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'Hi'}}]})}\n\n".encode(),
        f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode(),
        f"data: {json.dumps({**chunk, 'choices': [], 'usage': {'prompt_tokens': 12, 'completion_tokens': 8, 'total_tokens': 20}})}\n\n".encode(),
        b"data: [DONE]\n\n",
    )


def _delta_usages(text: str) -> list[Mapping[str, JsonValue]]:
    events: Final = [json.loads(line[len("data: ") :]) for line in text.splitlines() if line.startswith("data: ")]
    return [event["usage"] for event in events if event.get("type") == "message_delta" and "usage" in event]


def _cost_config(wire_url: str) -> str:
    return (
        "model_list:\n"
        "  - model_name: amsg\n"
        "    litellm_params:\n"
        f"      model: anthropic/{_MODEL}\n"
        f"      api_base: {wire_url}\n"
        f"      api_key: {_KEY}\n"
        "  - model_name: omsg\n"
        "    litellm_params:\n"
        "      model: openai/gpt-4o-mini\n"
        f"      api_base: {wire_url}\n"
        "      api_key: synthetic-openai-key\n"
        "litellm_settings:\n"
        "  include_cost_in_streaming_usage: true\n"
        "general_settings:\n"
        "  master_key: os.environ/LITELLM_MASTER_KEY\n"
        "  database_url: os.environ/DATABASE_URL\n"
        "  store_model_in_db: true\n"
        "  disable_spend_logs: false\n"
        "  proxy_batch_write_at: 1\n"
    )


def _assert_cost_in_every_delta(gateway: Gateway, model: str) -> None:
    text: Final = _stream_text(
        gateway,
        "/v1/messages",
        {"model": model, "max_tokens": 20, "stream": True, "messages": [{"role": "user", "content": "Say 'Hi'"}]},
    )
    usages: Final = _delta_usages(text)
    assert usages, (model, text)
    costs: Final = [usage.get("cost") for usage in usages]
    assert all(isinstance(cost, (int, float)) and cost > 0 for cost in costs), (model, text)


def test_streaming_cost_injected_into_usage_for_anthropic_and_openai_responses(
    gateway: Gateway, tmp_path: Path
) -> None:
    def respond(request: Request) -> Reply:
        if request.target.endswith("/responses"):
            return Reply(content_type="text/event-stream", chunks=_openai_responses_stream())
        assert request.target == "/v1/messages", request.target
        return Reply(content_type="text/event-stream", chunks=_stream_chunks("msg_" + uuid.uuid4().hex))

    with (
        _upstream(respond) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_owned_config(tmp_path, _cost_config(wire.url))) as candidate,
    ):
        _assert_cost_in_every_delta(candidate, "amsg")
        _assert_cost_in_every_delta(candidate, "omsg")
        targets: Final = [request.target for request in _provider_calls(wire)]
        assert targets == ["/v1/messages", "/responses"], targets


def test_streaming_cost_injected_into_usage_for_openai_chat_completions_bridge(
    gateway: Gateway, tmp_path: Path
) -> None:
    def respond(request: Request) -> Reply:
        assert request.target.endswith("/chat/completions"), request.target
        assert json.loads(request.body)["stream"] is True
        return Reply(content_type="text/event-stream", chunks=_openai_chat_stream())

    with (
        _upstream(respond) as wire,
        owned_proxy(
            gateway,
            tmp_path,
            {"LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES": "true"},
            config=_owned_config(tmp_path, _cost_config(wire.url)),
        ) as candidate,
    ):
        _assert_cost_in_every_delta(candidate, "omsg")
        assert len(_provider_calls(wire)) == 1
