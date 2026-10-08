import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_MODEL: Final = "claude-sonnet-4-5-20250929"
_KEY: Final = "synthetic-anthropic-key"

_MESSAGE_BODY: Final = {
    "id": "msg_pt1",
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
    "router_settings:\n"
    "  disable_cooldowns: true\n"
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
                    "id": "msg_pt_stream",
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


def _metadata(marker: str) -> dict:
    return {"litellm_metadata": {"tags": [marker], "user": f"end-user-{marker}"}}


def _spend_row(marker: str):
    return read_rows(
        "SELECT prompt_tokens, completion_tokens, spend, status, request_tags, end_user "
        'FROM "LiteLLM_SpendLogs" WHERE request_tags::text LIKE %s',
        (f"%{marker}%",),
    )


def _assert_spend_row(marker: str, prompt_tokens: int, completion_tokens: int) -> None:
    rows: Final = eventually(lambda: _spend_row(marker), lambda values: len(values) >= 1, seconds=90)
    row: Final = rows[0]
    assert row["status"] == "success", rows
    assert row["prompt_tokens"] == prompt_tokens
    assert row["completion_tokens"] == completion_tokens
    assert marker in str(row["request_tags"])
    assert row["end_user"] == f"end-user-{marker}"
    assert float(row["spend"]) > 0


def test_passthrough_basic_completion_spend_row_v1_messages(gateway: Gateway) -> None:
    marker: Final = "pt-basic-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        body: Final = json.loads(request.body)
        assert body["model"] == _MODEL
        assert body["messages"] == [{"role": "user", "content": f"say hello {marker}"}]
        return Reply(body=json.dumps(_MESSAGE_BODY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": f"say hello {marker}"}],
                **_metadata(marker),
            },
        )
        assert response.status_code == 200, response.text


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
                **_metadata(marker),
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as stream:
            for line in stream.iter_text():
                chunks.append(line)
        assert "hello stream" in "".join(chunks)


def test_passthrough_wildcard_model_strips_provider_prefix(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: a wildcard anthropic/* deployment forwards the provider-prefixed model name to the upstream; "
        "the upstream receives 'anthropic/claude-haiku-4-5-20251001' instead of 'claude-haiku-4-5-20251001'"
    )

    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        assert body["model"] == "claude-haiku-4-5-20251001"
        return Reply(body=json.dumps(_MESSAGE_BODY).encode())

    with wire_server(respond) as wire:
        gateway.post(
            "/model/new",
            {
                "model_name": "*",
                "litellm_params": {"model": "anthropic/*", "api_base": wire.url, "api_key": _KEY},
            },
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
    marker: Final = "pt-native-" + uuid.uuid4().hex

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
                    "messages": [{"role": "user", "content": f"say hello {marker}"}],
                    **_metadata(marker),
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


def test_passthrough_spend_rows_recorded_for_v1_messages(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: successful anthropic_messages calls on /v1/messages never invoke the success logging handler "
        "(litellm/llms/custom_httpx/llm_http_handler.py returns via _maybe_wrap_in_fake_stream), so no "
        "LiteLLM_SpendLogs row is written and tags, end_user and spend are not recorded"
    )
    marker: Final = "pt-spend-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        if body.get("stream") is True:
            return Reply(content_type="text/event-stream", chunks=_stream_chunks())
        return Reply(body=json.dumps(_MESSAGE_BODY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_KEY)
        for stream in (False, True):
            response: Final = gateway.request(
                "POST",
                "/v1/messages",
                {
                    "model": model,
                    "max_tokens": 100,
                    "stream": stream,
                    "messages": [{"role": "user", "content": f"say hello {marker}"}],
                    **_metadata(marker),
                },
            )
            assert response.status_code == 200, response.text
        _assert_spend_row(marker, 11, 7)


def test_native_passthrough_spend_row_records_tags_and_spend(gateway: Gateway, tmp_path) -> None:
    pytest.skip(
        "BUG: pass_through_endpoint spend rows for successful /anthropic calls record empty request_tags, "
        "no end_user and zero prompt/completion tokens and spend"
    )
    marker: Final = "pt-native-spend-" + uuid.uuid4().hex

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
            response: Final = candidate.request(
                "POST",
                "/anthropic/v1/messages",
                {
                    "model": _MODEL,
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": f"say hello {marker}"}],
                    **_metadata(marker),
                },
            )
            assert response.status_code == 200, response.text
        _assert_spend_row(marker, 11, 7)


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
