"""Scaleway, an OpenAI-compatible provider, through the proxy against a scripted upstream (#34503).

`test_regression_*` fails on a build without the fix, `test_guard_*` already holds and must keep holding.
"""

import base64
import json
import mimetypes
import struct
import uuid
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import OpenAI

_SCALEWAY_KEY: Final = "synthetic-scaleway-secret"
_PINSTRIPES_KEY: Final = "synthetic-pinstripes-key"
_VECTOR: Final = (0.25, 0.5, 0.75)
_PROMPT_TOKENS: Final = 7
_EMBEDDING_MODELS: Final = ("qwen3-embedding-8b", "bge-multilingual-gemma2")
# The cost map prices each bare Scaleway embedding id under its vendor-namespaced row
_EMBEDDING_COST_MAP_KEYS: Final = (
    ("qwen3-embedding-8b", "scaleway/qwen/qwen3-embedding-8b"),
    ("bge-multilingual-gemma2", "scaleway/BAAI/bge-multilingual-gemma2"),
)
_WAV_BYTES: Final = (
    b"RIFF\x30\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x40\x1f\x00\x00\x40\x1f\x00\x00\x01\x00\x08\x00data\x08\x00\x00\x00"
    b"\x00\x01\x02\x03\x04\x05\x06\x07"
)
_WAV_CONTENT_TYPE: Final = mimetypes.guess_type("a.wav")[0] or "application/octet-stream"
_TRANSCRIPT: Final = "bonjour le monde"
_OVERFLOW_MESSAGE: Final = (
    "This model's maximum context length is 32768 tokens. However, you requested 40000 tokens. "
    "Please reduce the length of the messages."
)


def _embedding_reply(model: str, encoding_format: str) -> Reply:
    embedding: Final = (
        base64.b64encode(struct.pack("<3f", *_VECTOR)).decode("ascii") if encoding_format == "base64" else list(_VECTOR)
    )
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": embedding}],
                "model": model,
                "usage": {"prompt_tokens": _PROMPT_TOKENS, "total_tokens": _PROMPT_TOKENS},
            }
        ).encode()
    )


def _error_reply(status: int, message: str, code: str | None = None) -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"error": {"message": message, "type": "invalid_request_error", "code": code}}).encode(),
    )


def _completion_reply(model: str, text: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-scaleway",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        ).encode()
    )


def _calls(wire: Wire) -> list[tuple[str, str, str, dict[str, object]]]:
    return [
        (
            request.method,
            request.target,
            request.headers.get("authorization", ""),
            JSON_OBJECT.validate_json(request.body),
        )
        for request in wire.drain()
    ]


def _error_message(response: httpx.Response) -> str:
    return str(object_value(JSON_OBJECT.validate_json(response.content)["error"])["message"])


def _multipart_parts(request: Request) -> tuple[Message, ...]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return tuple(parsed.iter_parts())


def _text_fields(request: Request) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (part.get_param("name", header="content-disposition"), part.get_payload(decode=True).decode())
            for part in _multipart_parts(request)
            if part.get_filename() is None
        )
    )


def _file_fields(request: Request) -> tuple[tuple[str, str, bytes], ...]:
    return tuple(
        (part.get_param("name", header="content-disposition"), part.get_filename(), part.get_payload(decode=True))
        for part in _multipart_parts(request)
        if part.get_filename() is not None
    )


def _sdk(gateway: Gateway) -> OpenAI:
    return OpenAI(
        base_url=str(gateway.client.base_url).rstrip("/") + "/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )


@pytest.mark.parametrize("upstream_model", _EMBEDDING_MODELS)
def test_regression_scaleway_embedding_forwards_bare_model_and_options_and_returns_the_vectors(
    gateway: Gateway, upstream_model: str
) -> None:
    """a scaleway/<model> embedding reaches <api_base>/embeddings with the bare model id"""
    with wire_server(lambda _: _embedding_reply(upstream_model, "float")) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"scaleway/{upstream_model}", api_base=f"{wire.url}/v1", api_key=_SCALEWAY_KEY
        )
        response: Final = gateway.request(
            "POST",
            "/v1/embeddings",
            {"model": model, "input": ["hello"], "encoding_format": "float", "dimensions": 256, "user": "caller-7"},
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body["data"] == [{"object": "embedding", "index": 0, "embedding": list(_VECTOR)}], response.text
        usage: Final = object_value(body["usage"])
        assert (usage["prompt_tokens"], usage["total_tokens"]) == (_PROMPT_TOKENS, _PROMPT_TOKENS), response.text
        assert _calls(wire) == [
            (
                "POST",
                "/v1/embeddings",
                f"Bearer {_SCALEWAY_KEY}",
                {
                    "model": upstream_model,
                    "input": ["hello"],
                    "encoding_format": "float",
                    "dimensions": 256,
                    "user": "caller-7",
                },
            )
        ]


def test_regression_openai_sdk_default_base64_embedding_reaches_scaleway_and_decodes(gateway: Gateway) -> None:
    """the OpenAI SDK's default encoding_format=base64 reaches Scaleway and the caller gets floats"""
    with (
        wire_server(lambda _: _embedding_reply("qwen3-embedding-8b", "base64")) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="scaleway/qwen3-embedding-8b", api_base=f"{wire.url}/v1", api_key=_SCALEWAY_KEY
        )
        with _sdk(gateway) as sdk:
            response: Final = sdk.embeddings.create(model=model, input="hello")
        assert [item.embedding for item in response.data] == [list(_VECTOR)], response.model_dump_json()
        assert _calls(wire) == [
            (
                "POST",
                "/v1/embeddings",
                f"Bearer {_SCALEWAY_KEY}",
                {"model": "qwen3-embedding-8b", "input": "hello", "encoding_format": "base64"},
            )
        ]


@pytest.mark.parametrize(("upstream_model", "cost_map_key"), _EMBEDDING_COST_MAP_KEYS)
def test_regression_scaleway_embedding_is_charged_at_the_scaleway_per_token_price(
    gateway: Gateway, upstream_model: str, cost_map_key: str
) -> None:
    """scaleway/<model> embedding spend is priced at the cost map's Scaleway rate, not zero"""
    input_cost_per_token: Final = float(
        object_value(
            JSON_OBJECT.validate_json(Path("model_prices_and_context_window.json").read_bytes())[cost_map_key]
        )["input_cost_per_token"]
    )
    with wire_server(lambda _: _embedding_reply(upstream_model, "float")) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"scaleway/{upstream_model}", api_base=f"{wire.url}/v1", api_key=_SCALEWAY_KEY
        )
        response: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": ["price control"]})
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(
            _PROMPT_TOKENS * input_cost_per_token
        ), dict(response.headers)
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/embeddings")]


@pytest.mark.parametrize("status", (400, 401, 429))
def test_regression_scaleway_embedding_upstream_error_keeps_its_status(gateway: Gateway, status: int) -> None:
    """an upstream 4xx on a Scaleway embedding reaches the caller with the same status"""
    upstream_message: Final = f"scaleway rejected the embedding {uuid.uuid4().hex}"
    with wire_server(lambda _: _error_reply(status, upstream_message)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="scaleway/qwen3-embedding-8b", api_base=f"{wire.url}/v1", api_key=_SCALEWAY_KEY
        )
        response: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": ["hello"]})
        assert (response.status_code, upstream_message in _error_message(response)) == (status, True), response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/embeddings")]


@pytest.mark.parametrize("response_format", ("json", "text"))
def test_guard_scaleway_transcription_forwards_every_option_and_returns_the_transcript(
    gateway: Gateway, response_format: str
) -> None:
    """Scaleway audio transcription reaches <api_base>/audio/transcriptions with every form field"""
    wav: Final = _WAV_BYTES + uuid.uuid4().bytes
    upstream_text: Final = _TRANSCRIPT if response_format == "json" else f"{_TRANSCRIPT}\n"
    reply: Final = (
        Reply(body=json.dumps({"text": upstream_text}).encode())
        if response_format == "json"
        else Reply(body=upstream_text.encode(), content_type="text/plain")
    )
    with wire_server(lambda _: reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="scaleway/whisper-large-v3", api_base=f"{wire.url}/v1", api_key=_SCALEWAY_KEY
        )
        with _sdk(gateway) as sdk:
            response: Final = sdk.audio.transcriptions.with_raw_response.create(
                model=model,
                file=("a.wav", wav, _WAV_CONTENT_TYPE),
                language="fr",
                prompt="salutations",
                temperature=0.2,
                response_format=response_format,
            )
        assert (response.status_code, JSON_OBJECT.validate_json(response.content)) == (
            200,
            {"text": upstream_text, "usage": None},
        ), response.text
        requests: Final = wire.drain()
        assert [(request.method, request.target, request.headers["authorization"]) for request in requests] == [
            ("POST", "/v1/audio/transcriptions", f"Bearer {_SCALEWAY_KEY}")
        ]
        assert _text_fields(requests[0]) == (
            ("language", "fr"),
            ("model", "whisper-large-v3"),
            ("prompt", "salutations"),
            ("response_format", response_format),
            ("temperature", "0.2"),
        )
        assert _file_fields(requests[0]) == (("file", "a.wav", wav),)


def test_guard_scaleway_chat_completion_forwards_the_bare_model_and_returns_the_answer(gateway: Gateway) -> None:
    """Scaleway chat completions reach <api_base>/chat/completions unchanged"""
    with (
        wire_server(lambda _: _completion_reply("llama-3.1-8b-instruct", "salut")) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="scaleway/llama-3.1-8b-instruct", api_base=f"{wire.url}/v1", api_key=_SCALEWAY_KEY
        )
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hello"}]}
        )
        assert response.status_code == 200, response.text
        choices: Final = list_value(JSON_OBJECT.validate_json(response.content)["choices"])
        assert [object_value(object_value(choice)["message"])["content"] for choice in choices] == ["salut"], (
            response.text
        )
        assert _calls(wire) == [
            (
                "POST",
                "/v1/chat/completions",
                f"Bearer {_SCALEWAY_KEY}",
                {"model": "llama-3.1-8b-instruct", "messages": [{"role": "user", "content": "hello"}]},
            )
        ]


def test_guard_scaleway_rerank_forwards_query_and_documents_and_returns_the_ranking(gateway: Gateway) -> None:
    """Scaleway rerank reaches <api_base>/rerank unchanged"""
    ranking: Final = [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]
    reply: Final = Reply(
        body=json.dumps(
            {"id": "rerank-scaleway", "results": ranking, "usage": {"prompt_tokens": 5, "total_tokens": 5}}
        ).encode()
    )
    with wire_server(lambda _: reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="scaleway/bge-reranker-v2-m3", api_base=f"{wire.url}/v1", api_key=_SCALEWAY_KEY
        )
        response: Final = gateway.request(
            "POST", "/v1/rerank", {"model": model, "query": "capital", "documents": ["lyon", "paris"], "top_n": 2}
        )
        assert response.status_code == 200, response.text
        assert [
            (object_value(item)["index"], object_value(item)["relevance_score"])
            for item in list_value(JSON_OBJECT.validate_json(response.content)["results"])
        ] == [(1, 0.9), (0, 0.1)], response.text
        assert _calls(wire) == [
            (
                "POST",
                "/v1/rerank",
                f"Bearer {_SCALEWAY_KEY}",
                {"model": "bge-reranker-v2-m3", "query": "capital", "documents": ["lyon", "paris"], "top_n": 2},
            )
        ]


def test_regression_scaleway_chat_context_overflow_reaches_the_caller_as_context_window_exceeded(
    gateway: Gateway,
) -> None:
    """a Scaleway 400 naming the context length surfaces as a context-window-exceeded error"""
    reply: Final = _error_reply(400, _OVERFLOW_MESSAGE, "context_length_exceeded")
    with wire_server(lambda _: reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="scaleway/llama-3.1-8b-instruct", api_base=f"{wire.url}/v1", api_key=_SCALEWAY_KEY
        )
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "long"}]}
        )
        assert response.status_code == 400, response.text
        assert _error_message(response).startswith("litellm.ContextWindowExceededError"), response.text
        assert _OVERFLOW_MESSAGE in _error_message(response), response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/chat/completions")]


def test_guard_json_provider_without_embeddings_still_rejects_embeddings(gateway: Gateway) -> None:
    """a JSON-configured provider that does not declare embeddings never sends one upstream"""
    with (
        wire_server(lambda _: _completion_reply("m", "ok")) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model="pinstripes/m", api_base=f"{wire.url}/v1", api_key=_PINSTRIPES_KEY)
        embedding: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": ["hello"]})
        chat: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hello"}]}
        )
        assert (embedding.status_code, JSON_OBJECT.validate_json(embedding.content)) == (
            400,
            {
                "error": {
                    "message": "litellm.BadRequestError: Unmapped LLM provider for this endpoint. You passed model=m, "
                    "custom_llm_provider=pinstripes. Check supported provider and route: "
                    "https://docs.litellm.ai/docs/providers\n\n"
                    f"LiteLLM: model group '{model}' failed with the error above. No fallback was attempted.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "400",
                }
            },
        ), embedding.text
        assert chat.status_code == 200, chat.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/chat/completions")]
