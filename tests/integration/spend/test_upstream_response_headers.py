import json
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal
from uuid import uuid4

import pytest
from pydantic import JsonValue
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


class AmbiguousFallbackAttempt(AssertionError):
    pass


class MissingDisconnectSpendLog(AssertionError):
    pass


class MissingDisconnectStatus(AssertionError):
    pass


MISSING_DISCONNECT_LOG: Final = pytest.mark.xfail(
    strict=True,
    raises=MissingDisconnectSpendLog,
    reason="PR #44053: downstream disconnect leaves no terminal spend row after upstream headers arrived",
)


def _provider_response(request: Request) -> Reply:
    if request.method == "GET":
        assert request.target.endswith("/models")
        return Reply(body=b'{"object":"list","data":[]}')
    body: Final = json.loads(request.body)
    assert not any("capture" in name for name in request.headers)
    marker: Final = body["model"]
    stream: Final = body.get("stream", False)
    usage: Final = {"input_tokens": 1, "output_tokens": 1}
    headers: Final = MappingProxyType({"x-request-id": f"header-{marker}", "set-cookie": "secret=value"})
    if request.target.endswith("/messages"):
        payload: Final = {
            "id": f"msg_{marker}",
            "type": "message",
            "role": "assistant",
            "model": marker,
            "content": [{"type": "text", "text": "visible response"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": usage,
        }
        return Reply(
            headers=headers,
            content_type="text/event-stream" if stream else "application/json",
            body=(
                f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': payload})}\n\n"
                'event: message_stop\ndata: {"type":"message_stop"}\n\n'
                if stream
                else json.dumps(payload)
            ).encode(),
        )
    if request.target.endswith("/responses"):
        response: Final = {
            "id": f"resp_{marker}",
            "object": "response",
            "created_at": 1,
            "model": marker,
            "status": "completed",
            "output": [],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "usage": {**usage, "total_tokens": 2},
        }
        return Reply(
            headers=headers,
            content_type="text/event-stream" if stream else "application/json",
            body=(
                f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'sequence_number': 0, 'response': response})}\n\n"
                if stream
                else json.dumps(response)
            ).encode(),
        )
    assert request.target.endswith("/chat/completions"), request.target
    assert body["messages"] == [{"role": "user", "content": "header probe"}]
    tool: Final = {"id": "call-probe", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    chat: Final = {
        "id": f"chatcmpl-{marker}",
        "object": "chat.completion.chunk" if stream else "chat.completion",
        "created": 1,
        "model": marker,
        "choices": [
            {
                "index": 0,
                "delta" if stream else "message": {
                    "role": "assistant",
                    "tool_calls": [{**tool, "index": 0}] if stream else [tool],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return Reply(
        headers=headers,
        content_type="text/event-stream" if stream else "application/json",
        body=(f"data: {json.dumps(chat)}\n\ndata: [DONE]\n\n" if stream else json.dumps(chat)).encode(),
    )


@pytest.mark.timeout(240)
@pytest.mark.parametrize("store_prompts", (False, True))
def test_upstream_headers_survive_routes_streaming_and_spend_log_readback(
    gateway: Gateway, tmp_path: Path, store_prompts: bool
) -> None:
    config: Final = tmp_path / "proxy.yaml"
    config.write_text(
        "model_list: []\ngeneral_settings:\n  master_key: os.environ/LITELLM_MASTER_KEY\n"
        "  database_url: os.environ/DATABASE_URL\n  store_model_in_db: true\n"
        "  proxy_batch_write_at: 1\n  proxy_batch_polling_interval: 1\n"
    )
    with owned_proxy(
        gateway, tmp_path, {"STORE_PROMPTS_IN_SPEND_LOGS": str(store_prompts).lower()}, config=config
    ) as proxy:
        with wire_server(_provider_response) as wire, proxy.scenario() as scenario:
            for route in ("chat/completions", "responses", "messages"):
                for stream in (False, True):
                    marker: Final = f"probe-{uuid4().hex}"
                    model: Final = scenario.model(
                        model=f"{'anthropic' if route == 'messages' else 'openai'}/{marker}",
                        api_base=wire.url if route == "messages" else f"{wire.url}/v1",
                        input_cost_per_token=0,
                        output_cost_per_token=0,
                    )
                    result: Final = proxy.request(
                        "POST",
                        f"/v1/{route}",
                        {
                            "model": model,
                            "stream": stream,
                            **(
                                {"input": "header probe"}
                                if route == "responses"
                                else {
                                    "messages": [{"role": "user", "content": "header probe"}],
                                    "max_tokens": 16,
                                }
                            ),
                        },
                    )
                    assert result.status_code == 200, result.text
                    rows: Final = eventually(
                        lambda selected_model=model: read_rows(
                            'SELECT request_id, response, metadata FROM "LiteLLM_SpendLogs" WHERE model_group=%s',
                            (selected_model,),
                        ),
                        lambda values: len(values) == 1,
                        seconds=70,
                    )
                    metadata: Final = object_value(rows[0]["metadata"])
                    captured: Final = metadata["upstream_responses"]
                    assert isinstance(captured, list) and len(captured) == 1, rows
                    response_metadata: Final = object_value(captured[0])
                    assert response_metadata["status_code"] == 200
                    assert ["x-request-id", f"header-{marker}"] in response_metadata["headers"]
                    assert ["set-cookie", "[REDACTED]"] in response_metadata["headers"]
                    assert "secret=value" not in json.dumps(rows)
                    stored_body: Final = rows[0]["response"]
                    assert (
                        bool(json.loads(stored_body) if isinstance(stored_body, str) else stored_body) == store_prompts
                    )
                    readback: Final = proxy.request(
                        "GET", "/spend/logs", params={"request_id": str(rows[0]["request_id"])}
                    )
                    assert readback.status_code == 200, readback.text
                    assert f"header-{marker}" in readback.text


@pytest.mark.timeout(150)
@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize(
    "attribute_attempts",
    (
        pytest.param(False, id="full-history"),
        pytest.param(True, id="attempt-attribution"),
    ),
)
def test_fallback_spend_log_retains_history_and_attributes_attempts(
    gateway: Gateway, tmp_path: Path, stream: bool, attribute_attempts: bool
) -> None:
    import yaml

    marker: Final = uuid4().hex
    primary: Final = f"primary-{marker}"
    fallback: Final = f"fallback-{marker}"

    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            assert request.target.endswith("/models")
            return Reply(body=b'{"object":"list","data":[]}')
        model: Final = json.loads(request.body)["model"]
        if model == primary:
            return Reply(
                status=503,
                headers=MappingProxyType({"x-request-id": "failed-" + marker}),
                body=b'{"error":{"message":"unavailable","type":"server_error"}}',
            )
        return _provider_response(request)

    with wire_server(upstream) as wire:
        config: Final = tmp_path / "fallback.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": name,
                            "litellm_params": {
                                "model": "openai/" + name,
                                "api_key": "synthetic",
                                "api_base": wire.url + "/v1",
                                "input_cost_per_token": 0,
                                "output_cost_per_token": 0,
                                "max_retries": 0,
                            },
                        }
                        for name in (primary, fallback)
                    ],
                    "general_settings": {
                        "master_key": "os.environ/LITELLM_MASTER_KEY",
                        "database_url": "os.environ/DATABASE_URL",
                        "proxy_batch_write_at": 1,
                        "proxy_batch_polling_interval": 1,
                    },
                    "router_settings": {
                        "num_retries": 0,
                        "disable_cooldowns": True,
                        "fallbacks": [{primary: [fallback]}],
                    },
                }
            )
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as proxy:
            result: Final = proxy.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": primary,
                    "stream": stream,
                    "messages": [{"role": "user", "content": "header probe"}],
                },
            )
            assert result.status_code == 200, result.text
            assert fallback in result.text
            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s', ("chatcmpl-" + fallback,)
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
            captured: Final = object_value(rows[0]["metadata"])["upstream_responses"]
            assert isinstance(captured, list) and captured
            final_response: Final = object_value(captured[-1])
            assert final_response["status_code"] == 200
            assert ["x-request-id", "header-" + fallback] in final_response["headers"]
            assert "failed-" + marker not in json.dumps(final_response)
            assert len(captured) == 2, captured
            failed_response: Final = object_value(captured[0])
            assert failed_response["status_code"] == 503
            assert ["x-request-id", "failed-" + marker] in failed_response["headers"]
            if attribute_attempts and failed_response["attempt_id"] == final_response["attempt_id"]:
                raise AmbiguousFallbackAttempt(f"Failed and serving attempts share {final_response['attempt_id']!r}")


def _disconnect_frames(route: str, marker: str) -> tuple[bytes, ...]:
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        (
            {
                "id": "chatcmpl-" + marker,
                "object": "chat.completion.chunk",
                "created": 1,
                "model": marker,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "first"}, "finish_reason": None}],
            },
        )
        if route == "chat/completions"
        else (
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": "resp_" + marker,
                    "object": "response",
                    "created_at": 1,
                    "model": marker,
                    "status": "in_progress",
                    "output": [],
                    "tools": [],
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                },
            },
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "msg_" + marker,
                "output_index": 0,
                "content_index": 0,
                "delta": "first",
            },
        )
        if route == "responses"
        else (
            {
                "type": "message_start",
                "message": {
                    "id": "msg_" + marker,
                    "type": "message",
                    "role": "assistant",
                    "model": marker,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "first"}},
        )
    )
    return tuple(
        ((f"event: {event['type']}\n" if "type" in event else "") + f"data: {json.dumps(event)}\n\n").encode()
        for event in events
    )


@pytest.mark.timeout(150)
@pytest.mark.parametrize(
    ("route", "before_first_chunk"),
    (
        pytest.param("chat/completions", True, id="chat-before-first-chunk", marks=MISSING_DISCONNECT_LOG),
        pytest.param("responses", True, id="responses-before-first-chunk", marks=MISSING_DISCONNECT_LOG),
        pytest.param("messages", True, id="messages-before-first-chunk", marks=MISSING_DISCONNECT_LOG),
        pytest.param("chat/completions", False, id="chat-mid-stream"),
        pytest.param("responses", False, id="responses-mid-stream", marks=MISSING_DISCONNECT_LOG),
        pytest.param(
            "messages",
            False,
            id="messages-mid-stream",
            marks=pytest.mark.xfail(
                strict=True,
                raises=MissingDisconnectStatus,
                reason="PR #44053: Messages spend row retains headers but omits downstream disconnect status",
            ),
        ),
    ),
)
@pytest.mark.parametrize("store_prompts", (False, True), ids=("body-off", "body-on"))
def test_downstream_disconnect_closes_upstream_and_persists_received_headers(
    gateway: Gateway,
    tmp_path: Path,
    before_first_chunk: bool,
    store_prompts: bool,
    route: Literal["chat/completions", "responses", "messages"],
) -> None:
    import socket
    import threading

    import yaml

    marker: Final = "disconnect-" + uuid4().hex
    headers_sent: Final = threading.Event()
    frames: Final = _disconnect_frames(route, marker)

    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            assert request.target == "/v1/models"
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.target == f"/v1/{route}"
        assert json.loads(request.body)["model"] == marker
        assert json.loads(request.body)["stream"] is True
        return Reply(
            headers=MappingProxyType({"x-request-id": marker, "set-cookie": "secret=value"}),
            content_type="text/event-stream",
            chunks=frames,
            headers_sent=headers_sent,
            wait_for_disconnect_after=0 if before_first_chunk else len(frames),
        )

    with wire_server(upstream) as wire:
        config: Final = tmp_path / "disconnect.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": marker,
                            "litellm_params": {
                                "model": f"{'anthropic' if route == 'messages' else 'openai'}/{marker}",
                                "api_key": "synthetic",
                                "api_base": wire.url if route == "messages" else wire.url + "/v1",
                                "input_cost_per_token": 0,
                                "output_cost_per_token": 0,
                                "max_retries": 0,
                                "timeout": 60,
                            },
                        }
                    ],
                    "general_settings": {
                        "master_key": "os.environ/LITELLM_MASTER_KEY",
                        "database_url": "os.environ/DATABASE_URL",
                        "proxy_batch_write_at": 1,
                        "proxy_batch_polling_interval": 1,
                    },
                    "router_settings": {"num_retries": 0},
                }
            )
        )
        with owned_proxy(
            gateway, tmp_path, {"STORE_PROMPTS_IN_SPEND_LOGS": str(store_prompts).lower()}, config=config
        ) as proxy:
            address: Final = proxy.client.base_url
            payload: Final = json.dumps(
                {
                    "model": marker,
                    **(
                        {"input": "header probe"}
                        if route == "responses"
                        else {
                            "messages": [{"role": "user", "content": "header probe"}],
                            "max_tokens": 16,
                        }
                    ),
                    "stream": True,
                }
            ).encode()
            with socket.create_connection((address.host, address.port), timeout=10) as downstream:
                downstream.sendall(
                    (
                        f"POST /v1/{route} HTTP/1.1\r\nHost: {address.host}\r\n"
                        f"Authorization: Bearer {proxy.key}\r\nContent-Type: application/json\r\n"
                        f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
                    ).encode()
                    + payload
                )
                assert headers_sent.wait(timeout=10), "Provider response headers were never sent"
                if not before_first_chunk:
                    with downstream.makefile("rb") as response:
                        assert response.readline().split()[1] == b"200"
                        assert any(b'"first"' in line for line in response if line.startswith(b"data: "))
                downstream.shutdown(socket.SHUT_RDWR)
            closed: Final = eventually(
                wire.disconnected.qsize, lambda count: count == 1, seconds=6, return_last_on_timeout=True
            )
            rows: Final = eventually(
                lambda: read_rows('SELECT metadata, response FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (marker,)),
                bool,
                seconds=20,
                return_last_on_timeout=True,
            )
            assert closed == 1, "Proxy did not close the provider connection after downstream disconnect"
            assert len(tuple(request for request in wire.drain() if request.method == "POST")) == 1
        final_rows: Final = read_rows(
            'SELECT metadata, response FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (marker,)
        )
        if not rows and not final_rows:
            raise MissingDisconnectSpendLog(
                "No terminal spend row before or after shutdown; upstream socket was closed"
            )
        assert len(rows) == len(final_rows) == 1, {"before_shutdown": rows, "after_shutdown": final_rows}
        assert final_rows == rows, "Shutdown changed the terminal spend snapshot"
        metadata: Final = object_value(final_rows[0]["metadata"])
        captured: Final = metadata.get("upstream_responses")
        assert isinstance(captured, list) and len(captured) == 1, metadata
        received: Final = object_value(captured[0])
        assert received["status_code"] == 200
        assert ["x-request-id", marker] in received["headers"]
        assert ["set-cookie", "[REDACTED]"] in received["headers"]
        assert "secret=value" not in json.dumps(final_rows)
        stored_body: Final = final_rows[0]["response"]
        if not store_prompts:
            assert not (json.loads(stored_body) if isinstance(stored_body, str) else stored_body), final_rows
        elif not before_first_chunk:
            assert "first" in str(stored_body), final_rows
        error_information: Final = metadata.get("error_information")
        if error_information is None:
            raise MissingDisconnectStatus("Received headers persisted, but the terminal row lacks disconnect status")
        assert object_value(error_information)["error_code"] == "499", metadata
