import json
import uuid
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, wire_server
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
from pydantic import JsonValue

_OPENAI_KEY: Final = "synthetic-openai-key"
_ANSWER: Final = "scripted response"


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


def _client(gateway: Gateway, key: str) -> OpenAI:
    return OpenAI(
        api_key=key,
        base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
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
    ("version", "environment", "marker"),
    (
        (2, "staging", "staging two"),
        ("2", "staging", "staging two"),
        (1, "staging", "staging one"),
        (None, "staging", "staging two"),
        (None, None, "production one"),
    ),
)
def test_chat_completion_resolves_prompt_id_by_version_and_environment(
    gateway: Gateway,
    version: int | str | None,
    environment: str | None,
    marker: str,
) -> None:
    def respond(request: Request) -> Reply:
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
            **({"prompt_version": version} if version is not None else {}),
            **({"prompt_environment": environment} if environment is not None else {}),
        }
        with _client(gateway, key) as client:
            response: Final = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "client turn"}],
                extra_body=extras,
            )
        assert response.choices[0].message.content == _ANSWER, response.model_dump_json()
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/chat/completions")]


def test_streaming_chat_completion_resolves_prompt_and_preserves_streaming(
    gateway: Gateway,
) -> None:
    def respond(request: Request) -> Reply:
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
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/chat/completions")]


def test_responses_create_resolves_prompt_and_shapes_template_messages(
    gateway: Gateway,
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/responses"
        assert request.headers["authorization"] == f"Bearer {_OPENAI_KEY}"
        body: Final[dict[str, JsonValue]] = json.loads(request.body)
        assert body == {
            "model": "gpt-4o-mini",
            "input": [
                {"role": "system", "content": "staging two"},
                {"role": "user", "content": "Hi x"},
                {"role": "user", "content": "client turn"},
            ],
            "temperature": 0.2,
        }, body
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
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": _ANSWER, "annotations": []}],
                        }
                    ],
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", api_key=_OPENAI_KEY)
        key: Final = scenario.key(models=[model])
        prompt_id: Final = f"prompt-{uuid.uuid4().hex}"
        _prompt_setup(gateway, scenario, model, prompt_id)
        with _client(gateway, key) as client:
            response: Final = client.responses.create(
                model=model,
                input="client turn",
                extra_body={
                    "prompt_id": prompt_id,
                    "prompt_variables": {"name": "x"},
                    "prompt_version": 2,
                    "prompt_environment": "staging",
                },
            )
        assert response.output_text == _ANSWER, response.model_dump_json()
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/responses")]


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
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/chat/completions")]
