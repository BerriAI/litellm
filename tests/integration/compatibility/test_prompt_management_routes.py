import json
import uuid
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, wire_server
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk
from pydantic import JsonValue

_OPENAI_KEY: Final = "synthetic-openai-key"
_ANSWER: Final = "scripted response"
_DISCOVERY_PROBE: Final = ("GET", "/v1/models")


def _is_discovery_probe(request: Request) -> bool:
    return (request.method, request.target) == _DISCOVERY_PROBE


def _content(model: str, marker: str) -> str:
    return f"---\nmodel: {model}\ntemperature: 0.2\n---\nSystem: {marker}\nUser: Hi {{{{name}}}}"


def _delete_prompt(gateway: Gateway, prompt_id: str) -> None:
    deleted: Final = gateway.request("DELETE", f"/prompts/{prompt_id}")
    assert deleted.status_code == 200, deleted.text


def _create_prompt(
    gateway: Gateway,
    prompt_id: str,
    model: str,
    environment: str,
    marker: str,
) -> None:
    response: Final = gateway.request(
        "POST",
        "/prompts",
        {
            "prompt_id": prompt_id,
            "litellm_params": {
                "prompt_id": prompt_id,
                "prompt_integration": "dotprompt",
                "dotprompt_content": _content(model, marker),
            },
            "prompt_info": {"prompt_type": "db", "environment": environment},
        },
    )
    assert response.status_code == 200, response.text


def _client(gateway: Gateway, key: str, base_path: str = "/v1") -> OpenAI:
    return OpenAI(
        api_key=key,
        base_url=f"{str(gateway.client.base_url).rstrip('/')}{base_path}",
        http_client=httpx.Client(trust_env=False),
    )


def _chat_response() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-integration",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": _ANSWER},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        ).encode()
    )


def _chunk(content: str, finish_reason: str | None = None) -> bytes:
    return (
        b"data: "
        + json.dumps(
            {
                "id": "chatcmpl-integration",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content} if content else {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
        ).encode()
        + b"\n\n"
    )


def _stream_response() -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _chunk("scripted "),
            _chunk("response"),
            _chunk("", "stop"),
            _usage_chunk(),
            b"data: [DONE]\n\n",
        ),
    )


def _usage_chunk() -> bytes:
    return (
        b"data: "
        + json.dumps(
            {
                "id": "chatcmpl-integration",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "gpt-4o-mini",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        ).encode()
        + b"\n\n"
    )


def _responses_stream_response() -> Reply:
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": "resp-integration",
                "object": "response",
                "created_at": 1700000000,
                "status": "in_progress",
                "model": "gpt-4o-mini",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg-integration",
            "output_index": 0,
            "content_index": 0,
            "delta": "scripted ",
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 2,
            "item_id": "msg-integration",
            "output_index": 0,
            "content_index": 0,
            "delta": "response",
        },
        {
            "type": "response.completed",
            "sequence_number": 3,
            "response": {
                "id": "resp-integration",
                "object": "response",
                "created_at": 1700000000,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "id": "msg-integration",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": _ANSWER, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 2,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 12,
                },
            },
        },
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _prompt_setup(
    gateway: Gateway,
    scenario: Scenario,
    model: str,
    prompt_id: str,
) -> None:
    _create_prompt(gateway, prompt_id, model, "staging", "staging one")
    scenario.cleanups.callback(_delete_prompt, gateway, prompt_id)
    _create_prompt(gateway, prompt_id, model, "staging", "staging two")
    _create_prompt(gateway, prompt_id, model, "production", "production one")


@pytest.mark.parametrize(
    ("version", "environment", "marker", "base_path"),
    (
        (2, "staging", "staging two", "/v1"),
        ("1", "staging", "staging one", "/v1"),
        (1, "staging", "staging one", "/v1"),
        (None, "staging", "staging two", "/v1"),
        (None, None, "production one", "/v1"),
        ("1", "staging", "staging one", ""),
        (None, None, "production one", ""),
    ),
)
def test_chat_completion_resolves_prompt_id_by_version_and_environment(
    gateway: Gateway,
    version: int | str | None,
    environment: str | None,
    marker: str,
    base_path: str,
) -> None:
    def respond(request: Request) -> Reply:
        if _is_discovery_probe(request):
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_OPENAI_KEY}"
        body: Final[dict[str, JsonValue]] = json.loads(request.body)
        assert body == {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": marker},
                {"role": "user", "content": "Hi x"},
                {"role": "user", "content": "client turn"},
            ],
            "temperature": 0.2,
        }, body
        return _chat_response()

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", api_key=_OPENAI_KEY)
        key: Final = scenario.key(models=[model])
        prompt_id: Final = f"prompt-{uuid.uuid4().hex}"
        _prompt_setup(gateway, scenario, model, prompt_id)
        extras: Final[dict[str, JsonValue]] = {
            "prompt_id": prompt_id,
            "prompt_variables": {"name": "x"},
            "prompt_label": "release",
            **({"prompt_version": version} if version is not None else {}),
            **({"prompt_environment": environment} if environment is not None else {}),
        }
        with _client(gateway, key, base_path) as client:
            response: Final = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "client turn"}],
                extra_body=extras,
            )
        assert response.choices[0].message.content == _ANSWER, response.model_dump_json()
        requests: Final = [request for request in wire.drain() if not _is_discovery_probe(request)]
        assert [(request.method, request.target) for request in requests] == [("POST", "/v1/chat/completions")]


def test_chat_completion_resolves_versioned_prompt_id_returned_by_create(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: prompt_id '<id>.v1' (the id POST /prompts returns) with prompt_environment=staging renders the "
        "latest staging version 'staging two' instead of version 1 'staging one'"
    )

    def respond(request: Request) -> Reply:
        if _is_discovery_probe(request):
            return Reply(body=b'{"object":"list","data":[]}')
        assert (request.method, request.target) == ("POST", "/v1/chat/completions"), request
        body: Final[dict[str, JsonValue]] = json.loads(request.body)
        assert body == {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "staging one"},
                {"role": "user", "content": "Hi x"},
                {"role": "user", "content": "client turn"},
            ],
            "temperature": 0.2,
        }, body
        return _chat_response()

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", api_key=_OPENAI_KEY)
        key: Final = scenario.key(models=[model])
        prompt_id: Final = f"prompt-{uuid.uuid4().hex}"
        _prompt_setup(gateway, scenario, model, prompt_id)
        with _client(gateway, key) as client:
            response: Final = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "client turn"}],
                extra_body={
                    "prompt_id": f"{prompt_id}.v1",
                    "prompt_variables": {"name": "x"},
                    "prompt_environment": "staging",
                },
            )
        assert response.choices[0].message.content == _ANSWER, response.model_dump_json()
        requests: Final = [request for request in wire.drain() if not _is_discovery_probe(request)]
        assert [(request.method, request.target) for request in requests] == [("POST", "/v1/chat/completions")]


def test_streaming_chat_completion_resolves_prompt_and_preserves_streaming(
    gateway: Gateway,
) -> None:
    def respond(request: Request) -> Reply:
        if _is_discovery_probe(request):
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_OPENAI_KEY}"
        body: Final[dict[str, JsonValue]] = json.loads(request.body)
        assert body == {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "staging two"},
                {"role": "user", "content": "Hi x"},
                {"role": "user", "content": "client turn"},
            ],
            "temperature": 0.2,
            "stream": True,
            "stream_options": {"include_usage": True},
        }, body
        return _stream_response()

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", api_key=_OPENAI_KEY)
        key: Final = scenario.key(models=[model])
        prompt_id: Final = f"prompt-{uuid.uuid4().hex}"
        _prompt_setup(gateway, scenario, model, prompt_id)
        with _client(gateway, key) as client:
            stream: Final = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "client turn"}],
                stream=True,
                extra_body={
                    "prompt_id": prompt_id,
                    "prompt_variables": {"name": "x"},
                    "prompt_version": 2,
                    "prompt_environment": "staging",
                },
            )
            chunks: Final = tuple(stream)
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == _ANSWER, chunks
        requests: Final = [request for request in wire.drain() if not _is_discovery_probe(request)]
        assert [(request.method, request.target) for request in requests] == [("POST", "/v1/chat/completions")]


_TEXT_INPUT: Final = "client turn"
_MESSAGE_INPUT: Final[list[JsonValue]] = [{"role": "user", "content": "client turn"}]
_TEXT_TURN: Final[dict[str, JsonValue]] = {"role": "user", "content": "client turn"}
_TYPED_TURN: Final[dict[str, JsonValue]] = {
    "type": "message",
    "role": "user",
    "content": [{"type": "input_text", "text": "client turn"}],
}
_TYPED_ITEM_INPUT: Final[list[JsonValue]] = [_TYPED_TURN]


@pytest.mark.parametrize(
    ("version", "environment", "client_input", "client_turn", "marker", "stream", "base_path"),
    (
        (2, "staging", _TEXT_INPUT, _TEXT_TURN, "staging two", False, "/v1"),
        (2, "staging", _TEXT_INPUT, _TEXT_TURN, "staging two", True, "/v1"),
        ("2", "staging", _TEXT_INPUT, _TEXT_TURN, "staging two", False, ""),
        ("1", "staging", _TEXT_INPUT, _TEXT_TURN, "staging one", True, ""),
        (None, None, _TEXT_INPUT, _TEXT_TURN, "production two", False, "/v1"),
        (None, None, _TEXT_INPUT, _TEXT_TURN, "production two", True, ""),
        ("2", "staging", _MESSAGE_INPUT, _TEXT_TURN, "staging two", False, "/v1"),
        (2, "staging", _TYPED_ITEM_INPUT, _TYPED_TURN, "staging two", True, "/v1"),
    ),
    ids=(
        "int-version-staging",
        "int-version-staging-stream",
        "string-version-staging-root-path",
        "string-version-one-stream-root-path",
        "default-environment-latest",
        "default-environment-latest-stream-root-path",
        "message-list-input",
        "typed-item-list-input-stream",
    ),
)
def test_responses_create_resolves_prompt_and_shapes_template_messages(
    gateway: Gateway,
    version: int | str | None,
    environment: str | None,
    client_input: str | list[JsonValue],
    client_turn: dict[str, JsonValue],
    marker: str,
    stream: bool,
    base_path: str,
) -> None:
    def respond(request: Request) -> Reply:
        if _is_discovery_probe(request):
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST"
        assert request.target == "/v1/responses"
        assert request.headers["authorization"] == f"Bearer {_OPENAI_KEY}"
        body: Final[dict[str, JsonValue]] = json.loads(request.body)
        assert body == {
            "model": "gpt-4o-mini",
            "input": [
                {"role": "system", "content": marker},
                {"role": "user", "content": "Hi x"},
                client_turn,
            ],
            "temperature": 0.2,
            "stream": stream,
        }, body
        if stream:
            return _responses_stream_response()
        return Reply(
            body=json.dumps(
                {
                    "id": "resp-integration",
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "completed",
                    "model": "gpt-4o-mini",
                    "output": [
                        {
                            "id": "msg-integration",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": _ANSWER, "annotations": []}],
                        }
                    ],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", api_key=_OPENAI_KEY)
        key: Final = scenario.key(models=[model])
        prompt_id: Final = f"prompt-{uuid.uuid4().hex}"
        _prompt_setup(gateway, scenario, model, prompt_id)
        _create_prompt(gateway, prompt_id, model, "production", "production two")
        with _client(gateway, key, base_path) as client:
            response: Final = client.responses.create(
                model=model,
                input=client_input,  # pyright: ignore[reportArgumentType]  # the SDK param union rejects JsonValue
                stream=stream,
                extra_body={
                    "prompt_id": prompt_id,
                    "prompt_variables": {"name": "x"},
                    "prompt_label": "release",
                    **({"prompt_version": version} if version is not None else {}),
                    **({"prompt_environment": environment} if environment is not None else {}),
                },
            )
            if stream:
                assert isinstance(response, Stream)
                events: Final = tuple(response)
                assert (
                    "".join(event.delta for event in events if event.type == "response.output_text.delta") == _ANSWER
                ), events
                assert events[-1].type == "response.completed", events
                assert events[-1].response.id == events[0].response.id, events
                assert events[-1].response.id.startswith("resp_"), events
            else:
                assert not isinstance(response, Stream)
                assert response.output_text == _ANSWER, response.model_dump_json()
        requests: Final = [request for request in wire.drain() if not _is_discovery_probe(request)]
        assert [(request.method, request.target) for request in requests] == [("POST", "/v1/responses")]


@pytest.mark.parametrize(
    ("history", "expected_messages"),
    (
        (
            [{"role": "user", "content": "prior"}],
            [{"role": "system", "content": "s"}, {"role": "user", "content": "prior"}],
        ),
        (None, [{"role": "system", "content": "s"}, {"role": "user", "content": "Hi x"}]),
    ),
)
def test_prompt_test_streams_rendered_prompt_with_optional_history(
    gateway: Gateway,
    history: list[dict[str, JsonValue]] | None,
    expected_messages: list[dict[str, JsonValue]],
) -> None:
    def respond(request: Request) -> Reply:
        if _is_discovery_probe(request):
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_OPENAI_KEY}"
        body: Final[dict[str, JsonValue]] = json.loads(request.body)
        assert body == {
            "model": "gpt-4o-mini",
            "messages": expected_messages,
            "temperature": 0.2,
            "stream": True,
            "stream_options": {"include_usage": True},
        }, body
        return _stream_response()

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", api_key=_OPENAI_KEY)
        template: Final = f"---\nmodel: {model}\ntemperature: 0.2\n---\nSystem: s\nUser: Hi {{{{name}}}}"
        body: Final[dict[str, JsonValue]] = {
            "dotprompt_content": template,
            "prompt_variables": {"name": "x"},
            **({"conversation_history": history} if history is not None else {}),
        }
        response: Final = gateway.request("POST", "/prompts/test", body)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream"), response.text
        events: Final = tuple(
            line.removeprefix("data: ").strip() for line in response.text.splitlines() if line.startswith("data:")
        )
        assert events[-1] == "[DONE]", response.text
        chunks: Final = tuple(ChatCompletionChunk.model_validate_json(event) for event in events[:-1])
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == _ANSWER, response.text
        requests: Final = [request for request in wire.drain() if not _is_discovery_probe(request)]
        assert [(request.method, request.target) for request in requests] == [("POST", "/v1/chat/completions")]
