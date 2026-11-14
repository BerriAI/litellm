import json
import uuid
from collections.abc import Callable
from typing import Final, Literal

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.shares_worker_state

_BACKEND: Final = "openai/gpt-4o-mini"
_PROMPT: Final = "upstream provider trace"
_PROVIDER: Final = "serving-site-a"
_ANSWER: Final = "hello"
_USAGE: Final[dict[str, JsonValue]] = {
    "prompt_tokens": 5,
    "completion_tokens": 1,
    "total_tokens": 6,
    "cost": 0.00015,
}


def _reply(
    request_id: str,
    *,
    response_provider: bool,
    stream_provider_mode: Literal["all", "last", "none"],
) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target.split("?", 1)[0].endswith("/chat/completions"), request.target
        body: Final = JSON_OBJECT.validate_json(request.body)
        assert body["messages"] == [{"role": "user", "content": _PROMPT}], body
        if body.get("stream") is True:
            common: Final = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": 1,
                "model": _BACKEND,
            }
            all_provider: Final[dict[str, JsonValue]] = {"provider": _PROVIDER}
            first_provider: Final = all_provider if stream_provider_mode == "all" else {}
            finish_provider: Final = (
                all_provider if stream_provider_mode in ("all", "last") else {}
            )
            usage_provider: Final = all_provider if stream_provider_mode == "all" else {}
            chunks: Final[tuple[dict[str, JsonValue], ...]] = (
                {
                    **common,
                    **first_provider,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": _ANSWER}, "finish_reason": None}],
                },
                {
                    **common,
                    **finish_provider,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
                {**common, **usage_provider, "choices": [], "usage": _USAGE},
            )
            return Reply(
                content_type="text/event-stream",
                chunks=tuple(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks)
                + (b"data: [DONE]\n\n",),
            )

        provider: Final[dict[str, JsonValue]] = {"provider": _PROVIDER} if response_provider else {}
        return Reply(
            body=json.dumps(
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": 1,
                    "model": _BACKEND,
                    **provider,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": _ANSWER},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": _USAGE,
                }
            ).encode()
        )

    return respond


def _stream_chunks(response: httpx.Response) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _send(
    gateway: Gateway,
    scenario: Scenario,
    wire_url: str,
    request_id: str,
    *,
    openrouter: bool,
    stream: bool,
) -> httpx.Response:
    model: Final = scenario.model(
        model=f"{'openrouter' if openrouter else 'openai'}/{_BACKEND}",
        api_base=wire_url,
        api_key="synthetic-provider-key",
    )
    key: Final = scenario.key(models=[model])
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": _PROMPT}],
            "stream": stream,
            **({"stream_options": {"include_usage": True}} if stream else {}),
        },
        key=key,
    )
    assert response.status_code == 200, response.text
    returned_id: Final = (
        string_value(_stream_chunks(response)[0]["id"])
        if stream
        else string_value(JSON_OBJECT.validate_json(response.content)["id"])
    )
    assert returned_id == request_id, response.text
    return response


def _spend_metadata(request_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (request_id,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return object_value(rows[0]["metadata"])


@pytest.mark.parametrize(
    ("stream", "stream_provider_mode"),
    [
        pytest.param(False, "none", id="nonstream"),
        pytest.param(True, "all", id="stream-every-chunk"),
        pytest.param(True, "last", id="stream-later-chunk"),
    ],
)
def test_openrouter_provider_is_persisted_in_spend_metadata_regression(
    gateway: Gateway, stream: bool, stream_provider_mode: Literal["all", "last", "none"]
) -> None:
    request_id: Final = f"chatcmpl-{uuid.uuid4().hex}"
    with (
        wire_server(
            _reply(
                request_id,
                response_provider=True,
                stream_provider_mode=stream_provider_mode,
            )
        ) as wire,
        gateway.scenario() as scenario,
    ):
        _send(gateway, scenario, wire.url, request_id, openrouter=True, stream=stream)
        metadata: Final = _spend_metadata(request_id)

    assert metadata.get("upstream_provider") == _PROVIDER, metadata


def test_openrouter_client_response_keeps_provider_in_body_guard(gateway: Gateway) -> None:
    request_id: Final = f"chatcmpl-{uuid.uuid4().hex}"
    with (
        wire_server(
            _reply(
                request_id,
                response_provider=True,
                stream_provider_mode="none",
            )
        ) as wire,
        gateway.scenario() as scenario,
    ):
        response: Final = _send(gateway, scenario, wire.url, request_id, openrouter=True, stream=False)

    body: Final = JSON_OBJECT.validate_json(response.content)
    choices: Final = body["choices"]
    assert isinstance(choices, list)
    message: Final = object_value(object_value(choices[0])["message"])
    assert body["provider"] == _PROVIDER
    assert "upstream_provider" not in body
    assert message["content"] == _ANSWER


def test_non_openrouter_spend_metadata_has_no_provider_guard(gateway: Gateway) -> None:
    request_id: Final = f"chatcmpl-{uuid.uuid4().hex}"
    with (
        wire_server(
            _reply(
                request_id,
                response_provider=False,
                stream_provider_mode="none",
            )
        ) as wire,
        gateway.scenario() as scenario,
    ):
        _send(gateway, scenario, wire.url, request_id, openrouter=False, stream=False)
        metadata: Final = _spend_metadata(request_id)

    assert "upstream_provider" not in metadata, metadata
