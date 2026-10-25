import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Final
from urllib.parse import urlsplit

import pytest
from integration._support.client import JSON_OBJECT, Gateway, held
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery, responses_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_DEPLOYMENT: Final = "relay-deployment"
_API_VERSION: Final = "2024-10-21"
_INPUT_COST: Final = 0.001
_OUTPUT_COST: Final = 0.002
_USAGE: Final[dict[str, JsonValue]] = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


def _marker() -> str:
    return "azure-relay-" + uuid.uuid4().hex


def _frame(payload: Mapping[str, JsonValue] | str) -> bytes:
    return b"data: " + (payload if isinstance(payload, str) else json.dumps(payload)).encode() + b"\n\n"


def _chunk(identity: str, choices: Sequence[JsonValue], **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt",
        "choices": list(choices),
        **extra,
    }


def _chat_stream(identity: str, with_usage: bool) -> tuple[bytes, ...]:
    return (
        _frame(_chunk(identity, [{"index": 0, "delta": {"content": "Hello there"}, "finish_reason": None}])),
        _frame(_chunk(identity, [{"index": 0, "delta": {}, "finish_reason": "stop"}])),
        *((_frame(_chunk(identity, [], usage=_USAGE)),) if with_usage else ()),
        _frame("[DONE]"),
    )


def _chat_completion(identity: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": "gpt",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
            "usage": _USAGE,
        }
    ).encode()


def _embedding() -> bytes:
    return json.dumps(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5]}],
            "model": "embed",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
    ).encode()


def _asks_usage(body: Mapping[str, JsonValue]) -> bool:
    options: Final = body.get("stream_options")
    return isinstance(options, dict) and options.get("include_usage") is True


def _respond(marker: str) -> Callable[[Request], Reply]:
    identity: Final = f"chatcmpl-{marker}"

    def respond(request: Request) -> Reply:
        body: Final = JSON_OBJECT.validate_json(request.body)
        assert marker in json.dumps(body), body
        path: Final = urlsplit(request.target).path
        if path.endswith("/embeddings"):
            return Reply(body=_embedding())
        if path.endswith("/responses"):
            return responses_reply(f"resp_{marker}", "gpt", "Hello", stream=body.get("stream") is True)
        if body.get("stream") is True:
            return Reply(content_type="text/event-stream", chunks=_chat_stream(identity, _asks_usage(body)))
        return Reply(body=_chat_completion(identity))

    return answering_model_discovery(respond)


@contextmanager
def _deployment(gateway: Gateway, marker: str, provider: str) -> Iterator[tuple[Wire, str]]:
    with wire_server(_respond(marker)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"{provider}/{_DEPLOYMENT}",
            api_base=wire.url,
            api_key="sk-azure-relay",
            api_version=_API_VERSION,
            input_cost_per_token=_INPUT_COST,
            output_cost_per_token=_OUTPUT_COST,
        )
        yield wire, model


def _posted(wire: Wire) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    return tuple(
        (urlsplit(request.target).path, JSON_OBJECT.validate_json(request.body))
        for request in wire.drain()
        if request.method == "POST"
    )


def _one_row(model: str) -> dict[str, JsonValue]:
    rows: Final = held(
        lambda: read_rows(
            "SELECT request_id, status, custom_llm_provider, prompt_tokens, completion_tokens, spend "
            'FROM "LiteLLM_SpendLogs" WHERE model_group=%s',
            (model,),
        ),
        lambda found: len(found) >= 1,
        holding=2,
        seconds=70,
    )
    assert len(rows) == 1, rows
    return rows[0]


def _assert_row(row: Mapping[str, JsonValue], provider: str, prompt: int, completion: int) -> None:
    assert (row["status"], row["custom_llm_provider"]) == ("success", provider), row
    assert (row["prompt_tokens"], row["completion_tokens"]) == (prompt, completion), row
    assert row["spend"] == pytest.approx(prompt * _INPUT_COST + completion * _OUTPUT_COST), row


def _chat_path(model: str, route: str) -> str:
    return (
        f"/azure/openai/deployments/{model}/chat/completions?api-version={_API_VERSION}"
        if route == "azure"
        else f"/azure_ai/{model}/chat/completions"
    )


def _chat_body(marker: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"messages": [{"role": "user", "content": marker}], **extra}


@pytest.mark.parametrize("provider", ["azure", "azure_ai"])
def test_router_model_chat_non_stream_writes_the_upstream_usage(gateway: Gateway, provider: str) -> None:
    marker: Final = _marker()
    with _deployment(gateway, marker, provider) as (wire, model):
        response: Final = gateway.request("POST", _chat_path(model, provider), _chat_body(marker))
        assert response.status_code == 200, response.text
        assert response.content == _chat_completion(f"chatcmpl-{marker}"), response.text
        ((path, _),) = _posted(wire)
        assert path.endswith("/chat/completions"), path
        row: Final = _one_row(model)
        assert row["request_id"] == f"chatcmpl-{marker}", row
        _assert_row(row, provider, 5, 3)


@pytest.mark.parametrize("provider", ["azure", "azure_ai"])
def test_router_model_chat_stream_with_usage_is_relayed_verbatim_and_logged(gateway: Gateway, provider: str) -> None:
    marker: Final = _marker()
    with _deployment(gateway, marker, provider) as (wire, model):
        sent: Final = _chat_body(marker, stream=True, stream_options={"include_usage": True})
        response: Final = gateway.request("POST", _chat_path(model, provider), sent)
        assert response.status_code == 200, response.text
        assert response.content == b"".join(_chat_stream(f"chatcmpl-{marker}", True)), response.text
        ((_, posted),) = _posted(wire)
        assert posted == sent, posted
        _assert_row(_one_row(model), provider, 5, 3)


@pytest.mark.parametrize(
    "extra",
    [pytest.param({}, id="plain"), pytest.param({"prompt": 5}, id="stray_prompt_field")],
)
def test_azure_chat_stream_without_usage_is_forwarded_untouched_and_counted_from_the_messages(
    gateway: Gateway, extra: Mapping[str, JsonValue]
) -> None:
    marker: Final = _marker()
    with _deployment(gateway, marker, "azure") as (wire, model):
        sent: Final = _chat_body(marker, stream=True, **extra)
        response: Final = gateway.request("POST", _chat_path(model, "azure"), sent)
        assert response.status_code == 200, response.text
        assert response.content == b"".join(_chat_stream(f"chatcmpl-{marker}", False)), response.text
        ((_, posted),) = _posted(wire)
        assert posted == sent, posted
        row: Final = _one_row(model)
        prompt, completion = row["prompt_tokens"], row["completion_tokens"]
        assert row["status"] == "success", row
        assert isinstance(prompt, int) and prompt > 0, row
        assert isinstance(completion, int) and completion > 0, row
        assert row["spend"] == pytest.approx(prompt * _INPUT_COST + completion * _OUTPUT_COST), row


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_azure_responses_body_model_writes_the_upstream_usage(gateway: Gateway, stream: bool) -> None:
    marker: Final = _marker()
    with _deployment(gateway, marker, "azure") as (wire, model):
        sent: Final = {"model": model, "input": marker, **({"stream": True} if stream else {})}
        response: Final = gateway.request("POST", f"/azure/openai/responses?api-version={_API_VERSION}", sent)
        assert response.status_code == 200, response.text
        assert f"resp_{marker}" in response.text, response.text
        ((path, _),) = _posted(wire)
        assert path.endswith("/responses"), path
        _assert_row(_one_row(model), "azure", 5, 3)


def test_azure_embeddings_write_the_upstream_prompt_tokens(gateway: Gateway) -> None:
    marker: Final = _marker()
    with _deployment(gateway, marker, "azure") as (wire, model):
        response: Final = gateway.request(
            "POST", f"/azure/openai/deployments/{model}/embeddings?api-version={_API_VERSION}", {"input": marker}
        )
        assert response.status_code == 200, response.text
        assert response.content == _embedding(), response.text
        ((path, _),) = _posted(wire)
        assert path == f"/openai/deployments/{_DEPLOYMENT}/embeddings", path
        row: Final = _one_row(model)
        assert (row["status"], row["custom_llm_provider"]) == ("success", "azure"), row
        assert (row["prompt_tokens"], row["completion_tokens"]) == (5, 0), row
        assert row["spend"] == pytest.approx(5 * _INPUT_COST), row
