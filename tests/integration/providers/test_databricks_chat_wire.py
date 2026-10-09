import json
import uuid
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_BACKEND: Final = "databricks-glm-5-2"
_API_KEY: Final = "synthetic-databricks-key"
_PROMPT: Final = "Summarise the cached briefing in one sentence."
_PROVIDER_USAGE: Final[Mapping[str, JsonValue]] = {
    "prompt_tokens": 12011,
    "completion_tokens": 8,
    "total_tokens": 12019,
    "cache_read_input_tokens": 12002,
    "cache_creation_input_tokens": 0,
}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class _PromptTokensDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cached_tokens: int | None = None


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: _PromptTokensDetails | None = None


class _Delta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    content: str | None = None


class _Choice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    delta: _Delta


class _Chunk(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    choices: tuple[_Choice, ...]
    usage: _Usage | None = None


def _frame(identity: str, choices: list[Mapping[str, object]], usage: Mapping[str, JsonValue] | None = None) -> bytes:
    value: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": _BACKEND,
        "choices": choices,
        **({} if usage is None else {"usage": usage}),
    }
    return b"data: " + json.dumps(value).encode() + b"\n\n"


@pytest.mark.covers("other.provider_wire.databricks.stream_usage_and_cache_reads_reach_client_and_spend_log")
def test_databricks_stream_final_usage_chunk_reaches_client_and_spend_log(gateway: Gateway) -> None:
    identity: Final = f"databricks-stream-{uuid.uuid4().hex}"
    frames: Final = (
        _frame(
            identity, [{"index": 0, "delta": {"role": "assistant", "content": "The briefing "}, "finish_reason": None}]
        ),
        _frame(identity, [{"index": 0, "delta": {"content": "is short."}, "finish_reason": None}]),
        _frame(identity, [{"index": 0, "delta": {}, "finish_reason": "stop"}]),
        _frame(identity, [], usage=_PROVIDER_USAGE),
        b"data: [DONE]\n\n",
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _BACKEND
        assert body["messages"] == [{"role": "user", "content": _PROMPT}]
        assert body["stream"] is True
        return Reply(content_type="text/event-stream", chunks=frames)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"databricks/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": _PROMPT}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            assert response.status_code == 200, response.read()
            lines: Final = tuple(line for line in response.iter_lines() if line.startswith("data: "))
        assert lines[-1] == "data: [DONE]", lines
        chunks: Final = tuple(_Chunk.model_validate_json(line.removeprefix("data: ")) for line in lines[:-1])
        assert {chunk.id for chunk in chunks} == {identity}
        assert (
            "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
            == "The briefing is short."
        )
        usages: Final = tuple(chunk.usage for chunk in chunks if chunk.usage is not None)
        assert len(usages) == 1, lines
        assert (
            usages[0].prompt_tokens,
            usages[0].completion_tokens,
            usages[0].total_tokens,
            usages[0].prompt_tokens_details.cached_tokens if usages[0].prompt_tokens_details is not None else None,
        ) == (12011, 8, 12019, 12002), lines
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT prompt_tokens, completion_tokens, total_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (identity,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"], rows[0]["total_tokens"]) == (12011, 8, 12019)


def _shape_reply(identity: str, stream: bool) -> Reply:
    if stream:
        chunks: Final = (
            _frame(
                identity,
                [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
            ),
            _frame(identity, [{"index": 0, "delta": {}, "finish_reason": "stop"}]),
            b"data: [DONE]\n\n",
        )
        return Reply(content_type="text/event-stream", chunks=chunks)

    body: Final = {
        "id": identity,
        "object": "chat.completion",
        "created": 1,
        "model": _BACKEND,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return Reply(body=json.dumps(body).encode())


@pytest.mark.parametrize(
    "case,endpoint,provider_model,request_body,expected_messages,responses_mode",
    [
        pytest.param(
            "chat_nonstream",
            "/v1/chat/completions",
            "databricks-meta-llama-3-3-70b-instruct",
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply in JSON"}]}],
                "response_format": {"type": "json_object"},
            },
            [{"role": "user", "content": "Reply in JSON"}],
            False,
            id="A1-chat-nonstream",
        ),
        pytest.param(
            "chat_stream",
            "/v1/chat/completions",
            "databricks-meta-llama-3-3-70b-instruct",
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply in JSON"}]}],
                "response_format": {"type": "json_object"},
                "stream": True,
            },
            [{"role": "user", "content": "Reply in JSON"}],
            False,
            id="A2-chat-stream",
        ),
        pytest.param(
            "responses-mode-bridge",
            "/v1/chat/completions",
            "databricks-meta-llama-3-1-8b-instruct",
            {
                "messages": [{"role": "user", "content": "Reply in JSON"}],
                "response_format": {"type": "json_object"},
            },
            [{"role": "user", "content": "Reply in JSON"}],
            True,
            id="A3-responses-mode",
        ),
        pytest.param(
            "messages-endpoint",
            "/v1/messages",
            "databricks-meta-llama-3-3-70b-instruct",
            {
                "max_tokens": 32,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply in JSON"}]}],
            },
            [{"role": "user", "content": "Reply in JSON"}],
            False,
            id="A4-messages",
        ),
        pytest.param(
            "responses-endpoint",
            "/v1/responses",
            "databricks-meta-llama-3-3-70b-instruct",
            {"input": [{"role": "user", "content": [{"type": "input_text", "text": "Reply in JSON"}]}]},
            [{"role": "user", "content": "Reply in JSON"}],
            False,
            id="A5-responses-list-input",
        ),
    ],
)
def test_databricks_non_claude_single_text_block_reaches_upstream_as_string(
    gateway: Gateway,
    case: str,
    endpoint: str,
    provider_model: str,
    request_body: dict[str, JsonValue],
    expected_messages: list[JsonValue],
    responses_mode: bool,
) -> None:
    identity: Final = f"databricks-single-text-{case}-{uuid.uuid4().hex}"
    streaming: Final = request_body.get("stream") is True

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["messages"] == expected_messages, body["messages"]
        return _shape_reply(identity, streaming)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"databricks/{provider_model}",
            api_base=wire.url,
            api_key=_API_KEY,
            model_info={"mode": "responses"} if responses_mode else None,
        )
        headers: Final = {"anthropic-version": "2023-06-01"} if endpoint == "/v1/messages" else {}
        body: Final = {"model": model, **request_body}
        if streaming:
            with gateway.client.stream(
                "POST", endpoint, json=body, headers={"Authorization": f"Bearer {gateway.key}", **headers}
            ) as response:
                assert response.status_code == 200, response.read()
                lines: Final = tuple(line for line in response.iter_lines() if line.startswith("data: "))
            assert lines[-1] == "data: [DONE]", lines
            assert any('"content":"ok"' in line for line in lines), lines
        else:
            response: Final = gateway.request("POST", endpoint, body, headers=headers)
            assert response.status_code == 200, response.text
            assert _JSON_OBJECT.validate_json(response.content)["id"]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]


@pytest.mark.parametrize(
    "case,provider_model,content,expected_content",
    [
        pytest.param(
            "cache-control",
            "databricks-meta-llama-3-3-70b-instruct",
            [{"type": "text", "text": "Reply in JSON", "cache_control": {"type": "ephemeral"}}],
            [{"type": "text", "text": "Reply in JSON", "cache_control": {"type": "ephemeral"}}],
            id="B1-unchanged-by-this-pr-cache-control",
        ),
        pytest.param(
            "two-text-blocks",
            "databricks-meta-llama-3-3-70b-instruct",
            [{"type": "text", "text": "Reply"}, {"type": "text", "text": " in JSON"}],
            [{"type": "text", "text": "Reply"}, {"type": "text", "text": " in JSON"}],
            id="B2-unchanged-by-this-pr-two-text-blocks",
        ),
        pytest.param(
            "text-and-image",
            "databricks-meta-llama-3-3-70b-instruct",
            [
                {"type": "text", "text": "Reply in JSON"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            ],
            [
                {"type": "text", "text": "Reply in JSON"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            ],
            id="B3-unchanged-by-this-pr-text-and-image",
        ),
        pytest.param(
            "claude",
            "databricks-claude-opus-5-5",
            [{"type": "text", "text": "Reply in JSON"}],
            [{"type": "text", "text": "Reply in JSON"}],
            id="B4-unchanged-by-this-pr-claude",
        ),
    ],
)
def test_databricks_list_content_unchanged_by_this_pr(
    gateway: Gateway,
    case: str,
    provider_model: str,
    content: list[JsonValue],
    expected_content: list[JsonValue],
) -> None:
    identity: Final = f"databricks-list-content-{case}-{uuid.uuid4().hex}"
    expected_messages: Final = [{"role": "user", "content": expected_content}]

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["messages"] == expected_messages, body["messages"]
        return _shape_reply(identity, False)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"databricks/{provider_model}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": content}]},
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["id"]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]
