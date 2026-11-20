import json
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
from typing import Final

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
MESSAGES: Final = ({"role": "user", "content": "Prove it"},)

HOP_1: Final = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "hello"}]},
            "finishReason": "CONTINUATION",
            "continuationToken": "dG9rLTE=",
        }
    ],
    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 30, "totalTokenCount": 40},
}
HOP_2: Final = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": " world"}]}, "finishReason": "STOP"}],
    "usageMetadata": {
        "promptTokenCount": 10,
        "cachedContentTokenCount": 9,
        "candidatesTokenCount": 20,
        "totalTokenCount": 30,
    },
}
STREAM_HOP_1_END: Final = {
    "candidates": [{**HOP_1["candidates"][0], "content": {"role": "model", "parts": [{"text": "lo"}]}}],
    "usageMetadata": HOP_1["usageMetadata"],
}


def _replay(hops: Sequence[httpx.Response], sent: list[dict[str, object]]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return hops[len(sent) - 1]

    return handler


def _sse(*chunks: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        content="".join(f"data: {json.dumps(chunk)}\r\n\r\n" for chunk in chunks).encode(),
        headers={"content-type": "text/event-stream"},
    )


NO_CITATIONS_OR_URL_CONTEXT: Final = (
    pytest.param({}, id="missing"),
    pytest.param({"citationMetadata": None, "urlContextMetadata": None}, id="null"),
    pytest.param({"citationMetadata": {}, "urlContextMetadata": {}}, id="empty"),
)


STREAM_FIRST_CHUNK: Final = {"candidates": [{"content": {"role": "model", "parts": [{"text": "hel"}]}}]}


def _stream_hops() -> tuple[httpx.Response, httpx.Response]:
    return (_sse(STREAM_FIRST_CHUNK, STREAM_HOP_1_END), _sse(HOP_2))


def _stream_text(chunks: list) -> str:
    return "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)


def _stream_finish_reasons(chunks: list) -> list[str]:
    return [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]


def _delta_fields(chunks: list) -> list[Mapping[str, object]]:
    return [
        choice.delta.provider_specific_fields
        for chunk in chunks
        for choice in chunk.choices  # comprehension-ok: flattening stream chunks into their choices
        if choice.delta.provider_specific_fields
    ]


def _client_tokens(chunks: list) -> list[object]:
    return [fields["continuation_token"] for fields in _delta_fields(chunks) if "continuation_token" in fields]


@dataclass(frozen=True, slots=True)
class _TokenEndpoint:
    credentials: str
    down: threading.Event


def _service_account_json(token_uri: str) -> str:
    private_key: Final = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        .decode()
    )
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "p",
            "private_key_id": "k",
            "private_key": private_key,
            "client_email": "sa@p.iam.gserviceaccount.com",
            "token_uri": token_uri,
        }
    )


@pytest.fixture
def token_endpoint() -> Iterator[_TokenEndpoint]:
    down: Final = threading.Event()
    minted: Final = count(1)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            refused: Final = down.is_set()
            payload: Final = json.dumps(
                {"error": "invalid_grant"} if refused else {"access_token": f"token-{next(minted)}", "expires_in": 0}
            ).encode()
            self.send_response(400 if refused else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args: object) -> None:
            return None

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield _TokenEndpoint(_service_account_json(f"http://127.0.0.1:{server.server_port}/token"), down)
    server.shutdown()
    server.server_close()


def _google_taking_each_access_token_once(
    hops: Sequence[httpx.Response], used: list[str], on_answer: Callable[[], object]
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        token: Final = request.headers["authorization"]
        if token in used:
            return httpx.Response(401, json={"error": {"code": 401, "status": "UNAUTHENTICATED"}})
        used.append(token)
        on_answer()
        return hops[len(used) - 1]

    return handler


async def _vertex_answer(
    is_async: bool,
    stream: bool,
    handler: Callable[[httpx.Request], httpx.Response],
    credentials: str,
    extra_headers: Mapping[str, str] | None = None,
) -> tuple[str, list[str], list[object]]:
    kwargs: Final = {
        "model": "vertex_ai/gemini-3.8-flash",
        "extra_headers": extra_headers,
        "messages": list(MESSAGES),
        "max_tokens": 100_000,
        "stream": stream,
        "vertex_project": "p",
        "vertex_location": "global",
        "vertex_credentials": credentials,
    }
    transport: Final = httpx.MockTransport(handler)
    response: Final = (
        await litellm.acompletion(**kwargs, client=AsyncHTTPHandler(transport=transport))
        if is_async
        else litellm.completion(**kwargs, client=HTTPHandler(client=httpx.Client(transport=transport)))
    )
    if stream:
        chunks: Final = [chunk async for chunk in response] if is_async else list(response)
        return _stream_text(chunks), _stream_finish_reasons(chunks), _client_tokens(chunks)
    choice: Final = response.choices[0]
    fields: Final = choice.message.provider_specific_fields or {}
    return choice.message.content, [choice.finish_reason], [fields["continuation_token"]] if fields else []


def _two_hops(stream: bool) -> tuple[httpx.Response, httpx.Response]:
    return _stream_hops() if stream else (httpx.Response(200, json=HOP_1), httpx.Response(200, json=HOP_2))


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_vertex_long_decode_finishes_when_each_continuation_outlives_the_access_token_before_it(
    is_async: bool, stream: bool, token_endpoint: _TokenEndpoint
):
    used: Final[list[str]] = []
    google: Final = _google_taking_each_access_token_once(_two_hops(stream), used, lambda: None)

    answer: Final = await _vertex_answer(is_async, stream, google, token_endpoint.credentials)

    assert answer == ("hello world", ["stop"], [])
    assert len(used) == 2


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_vertex_long_decode_returns_the_answer_so_far_with_its_token_when_no_new_access_token_can_be_had(
    is_async: bool, stream: bool, token_endpoint: _TokenEndpoint
):
    used: Final[list[str]] = []
    google: Final = _google_taking_each_access_token_once(_two_hops(stream), used, token_endpoint.down.set)

    answer: Final = await _vertex_answer(is_async, stream, google, token_endpoint.credentials)

    assert answer == ("hello", ["length"], ["dG9rLTE="])
    assert len(used) == 1


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_vertex_long_decode_keeps_the_callers_own_authorization_header_on_every_continuation(
    is_async: bool, stream: bool, token_endpoint: _TokenEndpoint
):
    hops: Final = _two_hops(stream)
    seen: Final[list[str]] = []

    def google(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return hops[len(seen) - 1]

    answer: Final = await _vertex_answer(
        is_async, stream, google, token_endpoint.credentials, {"Authorization": "Bearer mine"}
    )

    assert answer == ("hello world", ["stop"], [])
    assert seen == ["Bearer mine", "Bearer mine"]


BUDGET_ENV: Final = "LITELLM_GEMINI_DEFAULT_MAX_OUTPUT_TOKENS"
ARGON: Final = "gemini/gemini-4-argon"
OUTPUT_LIMIT_KEYS: Final = ("max_output_tokens", "maxOutputTokens")


def _output_limits(request: Mapping[str, object]) -> dict[str, object]:
    generation_config: Final = request.get("generationConfig", {})
    assert isinstance(generation_config, dict)
    return {key: value for key, value in generation_config.items() if key in OUTPUT_LIMIT_KEYS}


def _cost_map_max_output_tokens(model: str) -> int:
    provider, name = model.split("/", 1)
    max_output_tokens: Final = litellm.get_model_info(model=name, custom_llm_provider=provider)["max_output_tokens"]
    assert isinstance(max_output_tokens, int)
    return max_output_tokens


def _gemini_chat_models_answering_no_text() -> tuple[str, ...]:
    rows: Final = (
        (name, row)
        for name, row in litellm.model_cost.items()
        if row.get("mode") == "chat" and "text" not in row.get("supported_output_modalities", ("text",))
    )
    return tuple(
        name if name.startswith("gemini/") else f"vertex_ai/{name}"
        for name, row in rows
        if row.get("litellm_provider") in ("gemini", "vertex_ai-language-models")
    )


@pytest.mark.asyncio
async def test_env_budget_skips_every_chat_model_the_cost_map_lists_as_answering_no_text(
    monkeypatch: pytest.MonkeyPatch, token_endpoint: _TokenEndpoint
):
    monkeypatch.setenv(BUDGET_ENV, "50")
    models: Final = _gemini_chat_models_answering_no_text()
    sent: Final[list[dict[str, object]]] = []
    replies: Final = tuple(httpx.Response(200, json=HOP_2) for _ in models)
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_replay(replies, sent)))

    for model in models:
        await litellm.acompletion(
            model=model,
            messages=list(MESSAGES),
            api_key="fake-gemini-key",
            vertex_project="p",
            vertex_location="us-central1",
            vertex_credentials=token_endpoint.credentials,
            client=client,
        )

    assert models
    assert {model: _output_limits(request) for model, request in zip(models, sent, strict=True)} == dict.fromkeys(
        models, {}
    )


@pytest.mark.parametrize("model", ["gemini/gemini-2.5-flash", "vertex_ai/gemini-2.5-flash", ARGON])
@pytest.mark.asyncio
async def test_env_budget_above_a_models_output_limit_sends_that_limit_instead(
    monkeypatch: pytest.MonkeyPatch, token_endpoint: _TokenEndpoint, model: str
):
    budget: Final = 2 * _cost_map_max_output_tokens(model)
    monkeypatch.setenv(BUDGET_ENV, str(budget))
    sent: Final[list[dict[str, object]]] = []

    await litellm.acompletion(
        model=model,
        messages=list(MESSAGES),
        api_key="fake-gemini-key",
        vertex_project="p",
        vertex_location="us-central1",
        vertex_credentials=token_endpoint.credentials,
        client=AsyncHTTPHandler(transport=httpx.MockTransport(_replay((httpx.Response(200, json=HOP_2),), sent))),
    )

    assert _output_limits(sent[0]) == {"max_output_tokens": _cost_map_max_output_tokens(model)}
