import json
import uuid
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, JsonValue, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_MODEL: Final = "claude-sonnet-4-5-20250929"
_KEY: Final = "synthetic-anthropic-key"
_MESSAGE_ID: Final = "msg_passthrough_migration"

_MESSAGE_BODY: Final = {
    "id": _MESSAGE_ID,
    "type": "message",
    "role": "assistant",
    "model": _MODEL,
    "content": [{"type": "text", "text": "hello test"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 11, "output_tokens": 7},
}

_THINKING_BODY: Final = {
    **_MESSAGE_BODY,
    "content": [
        {"type": "thinking", "thinking": "pondering the joke", "signature": "sig1"},
        {"type": "text", "text": "hello thinking"},
    ],
    "usage": {"input_tokens": 11, "output_tokens": 30},
}

_PROXY_CONFIG: Final = (
    "model_list: []\n"
    "general_settings:\n"
    "  master_key: os.environ/LITELLM_MASTER_KEY\n"
    "  database_url: os.environ/DATABASE_URL\n"
    "  store_model_in_db: true\n"
    "  disable_spend_logs: false\n"
    "  proxy_batch_write_at: 1\n"
)


def _sse(event: str, payload: dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def _stream_chunks() -> tuple[bytes, ...]:
    return (
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": _MESSAGE_ID,
                    "type": "message",
                    "role": "assistant",
                    "model": _MODEL,
                    "content": [],
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            },
        )
        + _sse(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        )
        + _sse(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello stream"}},
        )
        + _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        + _sse(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}},
        )
        + _sse("message_stop", {"type": "message_stop"}),
    )


def _spend_rows(request_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        "SELECT status, prompt_tokens, completion_tokens, spend, request_tags, end_user, call_type "
        'FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
        (request_id,),
    )


def _passthrough_rows(digest: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        "SELECT status, prompt_tokens, completion_tokens, spend, request_tags, end_user, call_type "
        'FROM "LiteLLM_SpendLogs" WHERE api_key=%s AND call_type=%s',
        (digest, "pass_through_endpoint"),
    )


def _tags(row: JsonValue) -> list[JsonValue]:
    raw: Final = dict(row)["request_tags"]
    tags: Final = json.loads(raw) if isinstance(raw, str) else raw
    assert isinstance(tags, list), row
    return tags


def test_passthrough_basic_completion_spend_row_v1_messages(gateway: Gateway) -> None:
    marker: Final = "pt-basic-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == _KEY
        body: Final = json.loads(request.body)
        assert body["model"] == _MODEL
        assert body["messages"] == [{"role": "user", "content": f"say hello {marker}"}]
        return Reply(body=json.dumps({**_MESSAGE_BODY, "id": f"msg_{marker}"}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": f"say hello {marker}"}],
                "litellm_metadata": {"tags": [marker], "user": f"end-user-{marker}"},
            },
            headers={"x-litellm-tags": marker},
        )
        assert response.status_code == 200, response.text
        rows: Final = eventually(lambda: _spend_rows(f"msg_{marker}"), lambda values: len(values) == 1, seconds=90)
        row: Final = rows[0]
        assert row["status"] == "success", rows
        assert row["prompt_tokens"] == 11
        assert row["completion_tokens"] == 7
        assert float(row["spend"]) > 0
        assert marker in _tags(row), row
        assert row["end_user"] == f"end-user-{marker}", row


def test_passthrough_streaming_spend_row_v1_messages(gateway: Gateway) -> None:
    marker: Final = "pt-stream-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.target == "/v1/messages", request.target
        body: Final = json.loads(request.body)
        assert body["model"] == _MODEL
        assert body["stream"] is True
        return Reply(content_type="text/event-stream", chunks=_stream_chunks())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_KEY)
        chunks: Final = []
        with gateway.client.stream(
            "POST",
            f"{gateway.client.base_url}/v1/messages",
            json={
                "model": model,
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": f"say hello {marker}"}],
                "litellm_metadata": {"tags": [marker], "user": f"end-user-{marker}"},
            },
            headers={"Authorization": f"Bearer {gateway.key}", "x-litellm-tags": marker},
        ) as stream:
            for line in stream.iter_text():
                chunks.append(line)
        assert "hello stream" in "".join(chunks)


def test_passthrough_streaming_spend_row_written_v1_messages(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: a successful streamed /v1/messages call returns complete chunks but writes no LiteLLM_SpendLogs "
        "row (non-streamed calls write one); the streamed usage, tags and end_user are never recorded"
    )
    marker: Final = "pt-stream-spend-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        return Reply(content_type="text/event-stream", chunks=_stream_chunks())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_KEY)
        with gateway.client.stream(
            "POST",
            f"{gateway.client.base_url}/v1/messages",
            json={
                "model": model,
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": f"say hello {marker}"}],
                "litellm_metadata": {"tags": [marker], "user": f"end-user-{marker}"},
            },
            headers={"Authorization": f"Bearer {gateway.key}", "x-litellm-tags": marker},
        ) as stream:
            for _line in stream.iter_text():
                pass
        rows: Final = eventually(lambda: _spend_rows(f"msg_{marker}"), lambda values: len(values) == 1, seconds=90)
        row: Final = rows[0]
        assert row["status"] == "success", rows
        assert row["prompt_tokens"] == 11
        assert row["completion_tokens"] == 7
        assert float(row["spend"]) > 0
        assert marker in _tags(row), row
        assert row["end_user"] == f"end-user-{marker}", row


def test_passthrough_wildcard_model_strips_provider_prefix(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        assert body["model"] == "claude-haiku-4-5-20251001"
        return Reply(body=json.dumps(_MESSAGE_BODY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        created: Final = gateway.post(
            "/model/new",
            {
                "model_name": "anthropic/*",
                "litellm_params": {"model": "anthropic/*", "api_base": wire.url, "api_key": _KEY},
            },
        )
        scenario.cleanups.callback(
            scenario.delete_model, created["model_info"]["id"] if isinstance(created.get("model_info"), dict) else ""
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": "anthropic/claude-haiku-4-5-20251001",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello wildcard"}],
            },
        )
        assert response.status_code == 200, response.text


def test_passthrough_thinking_block_round_trips_v1_messages(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        assert body["model"] == "claude-haiku-4-5-20251001"
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 16000}
        return Reply(body=json.dumps(_THINKING_BODY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="anthropic/claude-haiku-4-5-20251001", api_base=wire.url, api_key=_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 20000,
                "thinking": {"type": "enabled", "budget_tokens": 16000},
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        assert response.status_code == 200, response.text
        content: Final = response.json()["content"]
        assert content[0]["type"] == "thinking"
        assert content[0]["thinking"] == "pondering the joke"


def test_passthrough_bad_request_returns_400_v1_messages(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        if not isinstance(body.get("messages"), list) or not all(
            isinstance(message, dict) for message in body["messages"]
        ):
            return Reply(
                status=400,
                body=json.dumps(
                    {"type": "error", "error": {"type": "invalid_request_error", "message": "messages must be objects"}}
                ).encode(),
            )
        return Reply(status=500)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_KEY)
        for stream in (False, True):
            response: Final = gateway.request(
                "POST",
                "/v1/messages",
                {"model": model, "max_tokens": 100, "stream": stream, "messages": ["hi"]},
            )
            assert response.status_code == 400, response.text


def test_native_anthropic_route_completion_stream_and_thinking(gateway: Gateway, tmp_path) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == _KEY
        body: Final = json.loads(request.body)
        if not isinstance(body.get("messages"), list) or not all(
            isinstance(message, dict) for message in body["messages"]
        ):
            return Reply(
                status=400,
                body=json.dumps(
                    {"type": "error", "error": {"type": "invalid_request_error", "message": "messages must be objects"}}
                ).encode(),
            )
        if body.get("stream") is True:
            return Reply(content_type="text/event-stream", chunks=_stream_chunks())
        if body.get("thinking") is not None:
            return Reply(body=json.dumps(_THINKING_BODY).encode())
        return Reply(body=json.dumps(_MESSAGE_BODY).encode())

    config: Final = tmp_path / "proxy_config.yaml"
    config.write_text(_PROXY_CONFIG)
    with wire_server(respond) as wire:
        with owned_proxy(
            gateway,
            tmp_path,
            {"ANTHROPIC_API_BASE": wire.url, "ANTHROPIC_API_KEY": _KEY},
            config=config,
        ) as candidate:
            completion: Final = candidate.request(
                "POST",
                "/anthropic/v1/messages",
                {
                    "model": _MODEL,
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "say hello native"}],
                },
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
            bad: Final = candidate.request(
                "POST",
                "/anthropic/v1/messages",
                {"model": _MODEL, "max_tokens": 100, "messages": ["hi"]},
            )
            assert bad.status_code == 400, bad.text
            chunks: Final = []
            with candidate.client.stream(
                "POST",
                f"{candidate.client.base_url}/anthropic/v1/messages",
                json={
                    "model": _MODEL,
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello native stream"}],
                },
                headers={"Authorization": f"Bearer {candidate.key}"},
            ) as stream:
                for line in stream.iter_text():
                    chunks.append(line)
            assert "hello stream" in "".join(chunks)


def test_native_passthrough_spend_row_records_usage_tags_and_spend(gateway: Gateway, tmp_path) -> None:
    pytest.skip(
        "BUG: pass_through_endpoint rows for successful /anthropic calls record request_tags=[] even with the "
        "documented x-litellm-tags header, no end_user, zero prompt/completion tokens and zero spend"
    )
    marker: Final = "pt-native-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        return Reply(body=json.dumps(_MESSAGE_BODY).encode())

    config: Final = tmp_path / "proxy_config.yaml"
    config.write_text(_PROXY_CONFIG)
    with wire_server(respond) as wire:
        with owned_proxy(
            gateway,
            tmp_path,
            {"ANTHROPIC_API_BASE": wire.url, "ANTHROPIC_API_KEY": _KEY},
            config=config,
        ) as candidate:
            digest: Final = sha256(candidate.key.encode()).hexdigest()
            response: Final = candidate.request(
                "POST",
                "/anthropic/v1/messages",
                {
                    "model": _MODEL,
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": f"say hello {marker}"}],
                    "litellm_metadata": {"user": f"end-user-{marker}"},
                },
                headers={"x-litellm-tags": marker},
            )
            assert response.status_code == 200, response.text
        rows: Final = eventually(lambda: _passthrough_rows(digest), lambda values: len(values) == 1, seconds=90)
        row: Final = rows[0]
        assert row["status"] == "success", rows
        assert row["prompt_tokens"] == 11
        assert row["completion_tokens"] == 7
        assert float(row["spend"]) > 0
        assert marker in _tags(row), row
        assert row["end_user"] == f"end-user-{marker}", row


def _openai_stream_chunks() -> tuple[bytes, ...]:
    response: Final = {
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
        )
        + _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": "msg_pto",
                "output_index": 0,
                "content_index": 0,
                "delta": "hi from openai",
            },
        )
        + _sse("response.completed", {"type": "response.completed", "response": response}),
    )


def _delta_usages(chunks: str) -> list[dict]:
    return [
        json.loads(line.split("data: ", 1)[1])["usage"]
        for line in chunks.splitlines()
        if line.startswith("data: ") and "message_delta" in line and '"usage"' in line
    ]


def test_streaming_cost_injected_into_usage_for_anthropic_and_openai(gateway: Gateway, tmp_path) -> None:
    def respond(request: Request) -> Reply:
        if "/responses" in request.target:
            return Reply(content_type="text/event-stream", chunks=_openai_stream_chunks())
        return Reply(content_type="text/event-stream", chunks=_stream_chunks())

    with wire_server(respond) as wire:
        config: Final = tmp_path / "proxy_config.yaml"
        config.write_text(
            "model_list:\n"
            "  - model_name: amsg\n"
            "    litellm_params:\n"
            "      model: anthropic/claude-sonnet-4-5-20250929\n"
            f"      api_base: {wire.url}\n"
            "      api_key: synthetic-anthropic-key\n"
            "  - model_name: omsg\n"
            "    litellm_params:\n"
            "      model: openai/gpt-4o-mini\n"
            f"      api_base: {wire.url}\n"
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
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            for model in ("amsg", "omsg"):
                chunks: Final = []
                with candidate.client.stream(
                    "POST",
                    f"{candidate.client.base_url}/v1/messages",
                    json={
                        "model": model,
                        "max_tokens": 100,
                        "stream": True,
                        "messages": [{"role": "user", "content": f"hello {model}"}],
                    },
                    headers={"Authorization": f"Bearer {candidate.key}"},
                ) as stream:
                    for line in stream.iter_text():
                        chunks.append(line)
                text: Final = "".join(chunks)
                usages: Final = _delta_usages(text)
                assert usages, (model, text)
                assert all(isinstance(usage.get("cost"), (int, float)) for usage in usages), (model, text)
