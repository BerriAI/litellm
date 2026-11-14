import base64
import json
from collections.abc import Callable, Mapping, Sequence
from itertools import chain
from typing import Final

import httpx
import pytest

import litellm
from litellm.google_genai import agenerate_content, agenerate_content_stream, generate_content, generate_content_stream
from litellm.litellm_core_utils.prompt_templates.factory import get_dummy_thought_signature
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.gemini.rejected_thought_signature_retry import with_placeholder_thought_signatures
from litellm.types.llms.vertex_ai import RequestBody
from litellm.types.utils import Choices, ModelResponse, ModelResponseStream
from litellm.utils import CustomStreamWrapper

MODEL: Final = "gemini/gemini-3.8-flash"
VERTEX_MINTED: Final = base64.b64encode(b"minted-by-vertex").decode()
VERTEX_MINTED_TEXT: Final = base64.b64encode(b"minted-by-vertex-for-text").decode()
AI_STUDIO_MINTED: Final = base64.b64encode(b"minted-by-ai-studio").decode()
HISTORY_WITH_FOREIGN_SIGNATURES: Final = (
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                "provider_specific_fields": {"thought_signature": VERTEX_MINTED},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "18C and sunny"},
    {
        "role": "assistant",
        "content": "It is 18C and sunny in Paris.",
        "provider_specific_fields": {"thought_signatures": [VERTEX_MINTED_TEXT]},
    },
    {"role": "user", "content": "And in Berlin?"},
)
HISTORY_WITHOUT_SIGNATURES: Final = (
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "18C and sunny"},
    {"role": "user", "content": "And in Berlin?"},
)
TOOLS: Final = (
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    },
)
ANSWER: Final = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "Berlin is 12C."}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 6, "totalTokenCount": 46},
}
# Wording observed from Google on 2026-10-08 when a signature minted by the other backend is replayed
AI_STUDIO_REJECTION: Final = "Corrupted thought signature."
VERTEX_REJECTION: Final = "Invalid thought signature."
UNRELATED_REJECTION: Final = 'Invalid JSON payload received. Unknown name "foo".'


def _rejected(message: str) -> httpx.Response:
    return httpx.Response(400, json={"error": {"code": 400, "message": message, "status": "INVALID_ARGUMENT"}})


def _sse(*chunks: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        content="".join(f"data: {json.dumps(chunk)}\r\n\r\n" for chunk in chunks).encode(),
        headers={"content-type": "text/event-stream"},
    )


def _replay(
    answers: Sequence[httpx.Response], sent: list[dict[str, object]]
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return answers[len(sent) - 1]

    return handler


def _signatures_of(body: Mapping[str, object], key: str = "thoughtSignature") -> list[str]:
    contents: Final = body["contents"]
    assert isinstance(contents, list)
    parts: Final = chain.from_iterable(content["parts"] for content in contents)
    return [part[key] for part in parts if key in part]


def _text_of(response: ModelResponse) -> str:
    choice: Final = response.choices[0]
    assert isinstance(choice, Choices), choice
    assert isinstance(choice.message.content, str), choice.message
    return choice.message.content


def _delta_text_of(chunk: object) -> str:
    assert isinstance(chunk, ModelResponseStream), chunk
    return chunk.choices[0].delta.content or ""


async def _complete(
    messages: Sequence[Mapping[str, object]],
    backend: Callable[[httpx.Request], httpx.Response],
    is_async: bool,
    stream: bool,
) -> str:
    kwargs: Final = {
        "model": MODEL,
        "messages": list(messages),
        "tools": list(TOOLS),
        "stream": stream,
        "api_key": "unit-test-key",
    }
    response: Final = (
        await litellm.acompletion(**kwargs, client=AsyncHTTPHandler(transport=httpx.MockTransport(backend)))
        if is_async
        else litellm.completion(**kwargs, client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(backend))))
    )
    if isinstance(response, ModelResponse):
        return _text_of(response)
    assert isinstance(response, CustomStreamWrapper), response
    if is_async:
        return "".join([_delta_text_of(chunk) async for chunk in response])
    return "".join(_delta_text_of(chunk) for chunk in response)


def _answer(stream: bool) -> httpx.Response:
    return _sse(ANSWER) if stream else httpx.Response(200, json=ANSWER)


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("rejection", [AI_STUDIO_REJECTION, VERTEX_REJECTION])
@pytest.mark.asyncio
async def test_signature_rejected_by_the_other_backend_is_resent_once_with_the_placeholder(
    is_async: bool, stream: bool, rejection: str
):
    sent: Final[list[dict[str, object]]] = []
    answer: Final = await _complete(
        HISTORY_WITH_FOREIGN_SIGNATURES, _replay((_rejected(rejection), _answer(stream)), sent), is_async, stream
    )
    assert answer == "Berlin is 12C."
    assert len(sent) == 2
    first_signatures, second_signatures = _signatures_of(sent[0]), _signatures_of(sent[1])
    assert set(first_signatures) == {VERTEX_MINTED, VERTEX_MINTED_TEXT}
    assert second_signatures == [get_dummy_thought_signature()] * len(first_signatures)
    assert {**sent[1], "contents": None} == {**sent[0], "contents": None}


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_placeholder_rejected_too_is_raised_after_the_second_post(is_async: bool, stream: bool):
    sent: Final[list[dict[str, object]]] = []
    with pytest.raises(litellm.BadRequestError, match="thought signature"):
        await _complete(
            HISTORY_WITH_FOREIGN_SIGNATURES,
            _replay((_rejected(AI_STUDIO_REJECTION), _rejected(AI_STUDIO_REJECTION)), sent),
            is_async,
            stream,
        )
    assert len(sent) == 2


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_unrelated_400_is_raised_without_a_resend(is_async: bool, stream: bool):
    sent: Final[list[dict[str, object]]] = []
    with pytest.raises(litellm.BadRequestError, match="Unknown name"):
        await _complete(HISTORY_WITH_FOREIGN_SIGNATURES, _replay((_rejected(UNRELATED_REJECTION),), sent), is_async, stream)
    assert len(sent) == 1


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_signature_400_on_a_history_carrying_only_the_placeholder_is_raised_without_a_resend(
    is_async: bool, stream: bool
):
    sent: Final[list[dict[str, object]]] = []
    with pytest.raises(litellm.BadRequestError, match="thought signature"):
        await _complete(HISTORY_WITHOUT_SIGNATURES, _replay((_rejected(AI_STUDIO_REJECTION),), sent), is_async, stream)
    assert len(sent) == 1
    assert set(_signatures_of(sent[0])) <= {get_dummy_thought_signature()}


def test_placeholder_body_keeps_the_original_untouched():
    body: Final[RequestBody] = {
        "contents": [
            {"role": "model", "parts": [{"text": "hi", "thoughtSignature": VERTEX_MINTED}, {"text": "unsigned"}]},
            {"role": "user", "parts": [{"text": "again"}]},
        ],
        "generationConfig": {"temperature": 0.1},
    }
    resend: Final = with_placeholder_thought_signatures(body)
    assert resend is not None
    assert _signatures_of(resend) == [get_dummy_thought_signature()]
    assert _signatures_of(body) == [VERTEX_MINTED]
    assert resend.get("generationConfig") == body.get("generationConfig")


def test_body_without_a_foreign_signature_has_nothing_to_resend():
    body: Final[RequestBody] = {
        "contents": [
            {"role": "model", "parts": [{"text": "hi", "thoughtSignature": get_dummy_thought_signature()}]},
            {"role": "user", "parts": [{"text": "again"}]},
        ]
    }
    assert with_placeholder_thought_signatures(body) is None


CACHE_CONTROL: Final = {"type": "ephemeral"}


def _cached_history(
    tag: str, cached_signature: str = VERTEX_MINTED, later_signatures: tuple[str, ...] = ()
) -> tuple[Mapping[str, object], ...]:
    later_answer: Final = (
        {"role": "assistant", "content": "Noted.", "provider_specific_fields": {"thought_signatures": [*later_signatures]}}
        if later_signatures
        else {"role": "assistant", "content": "Noted."}
    )
    prefix: Final = " ".join(f"Reference paragraph {i} for {tag}: weather values are in Celsius." for i in range(600))
    return (
        {"role": "system", "content": [{"type": "text", "text": prefix, "cache_control": CACHE_CONTROL}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the weather in Paris?", "cache_control": CACHE_CONTROL}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Calling the tool.", "cache_control": CACHE_CONTROL}],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    "provider_specific_fields": {"thought_signature": cached_signature},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "text", "text": "18C and sunny", "cache_control": CACHE_CONTROL}],
        },
        {"role": "user", "content": [{"type": "text", "text": "Got it.", "cache_control": CACHE_CONTROL}]},
        later_answer,
        {"role": "user", "content": "And in Berlin?"},
    )


def _google_ai_studio(
    stream: bool, generated: list[dict[str, object]], minted: frozenset[str] = frozenset()
) -> Callable[[httpx.Request], httpx.Response]:
    """Rejects a generateContent call whose own contents or referenced cache carry a signature AI Studio did not mint,
    the way the live backend answered on 2026-11-02."""
    caches: Final[dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/cachedContents"):
            if request.method == "GET":
                return httpx.Response(200, json={})
            caches["cachedContents/prefix"] = json.loads(request.content)
            return httpx.Response(
                200, json={"name": "cachedContents/prefix", "usageMetadata": {"totalTokenCount": 9000}}
            )
        body: Final = json.loads(request.content)
        generated.append(body)
        cache: Final = caches.get(body.get("cachedContent"), {"contents": []})
        assert isinstance(cache, dict)
        signatures: Final = {*_signatures_of(body), *_signatures_of(cache)}
        if signatures - {get_dummy_thought_signature(), *minted}:
            return _rejected(AI_STUDIO_REJECTION)
        return _answer(stream)

    return handler


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_signature_inside_a_cached_prefix_is_resent_inline_with_the_placeholder(is_async: bool, stream: bool):
    generated: Final[list[dict[str, object]]] = []
    answer: Final = await _complete(
        _cached_history(f"async={is_async} stream={stream}"),
        _google_ai_studio(stream, generated),
        is_async,
        stream,
    )
    assert answer == "Berlin is 12C."
    assert len(generated) == 2
    assert generated[0]["cachedContent"] == "cachedContents/prefix"
    assert _signatures_of(generated[0]) == []
    assert "cachedContent" not in generated[1]
    assert _signatures_of(generated[1]) == [get_dummy_thought_signature()]


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_signature_after_a_cache_the_backend_accepts_is_resent_with_the_cache_kept(is_async: bool, stream: bool):
    generated: Final[list[dict[str, object]]] = []
    answer: Final = await _complete(
        _cached_history(
            f"own cache async={is_async} stream={stream}", AI_STUDIO_MINTED, later_signatures=(VERTEX_MINTED_TEXT,)
        ),
        _google_ai_studio(stream, generated, minted=frozenset({AI_STUDIO_MINTED})),
        is_async,
        stream,
    )
    assert answer == "Berlin is 12C."
    assert [body.get("cachedContent") for body in generated] == ["cachedContents/prefix"] * 2
    assert _signatures_of(generated[0]) == [VERTEX_MINTED_TEXT]
    assert _signatures_of(generated[1]) == [get_dummy_thought_signature()]


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_signature_inside_and_after_the_cached_prefix_goes_inline_once_the_cache_kept_resend_is_rejected(
    is_async: bool, stream: bool
):
    generated: Final[list[dict[str, object]]] = []
    answer: Final = await _complete(
        _cached_history(f"both async={is_async} stream={stream}", later_signatures=(VERTEX_MINTED_TEXT,)),
        _google_ai_studio(stream, generated),
        is_async,
        stream,
    )
    assert answer == "Berlin is 12C."
    assert [body.get("cachedContent") for body in generated] == ["cachedContents/prefix"] * 2 + [None]
    assert _signatures_of(generated[1]) == [get_dummy_thought_signature()]
    assert _signatures_of(generated[2]) == [get_dummy_thought_signature()] * 2


def _backend_validating_cached_signatures(
    stream: bool, created: list[dict[str, object]], generated: list[dict[str, object]]
) -> Callable[[httpx.Request], httpx.Response]:
    """Rejects creating a cache whose contents carry a signature it did not mint, the way Vertex answered on
    2026-11-02."""

    def handler(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content) if request.content else {}
        if request.url.path.endswith("/cachedContents"):
            if request.method == "GET":
                return httpx.Response(200, json={})
            created.append(body)
        else:
            generated.append(body)
        if set(_signatures_of(body)) - {get_dummy_thought_signature()}:
            return _rejected(VERTEX_REJECTION)
        if request.url.path.endswith("/cachedContents"):
            return httpx.Response(
                200, json={"name": "cachedContents/prefix", "usageMetadata": {"totalTokenCount": 9000}}
            )
        return _answer(stream)

    return handler


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_signature_rejected_while_creating_the_cache_is_sent_inline_with_the_placeholder(
    is_async: bool, stream: bool
):
    created: Final[list[dict[str, object]]] = []
    generated: Final[list[dict[str, object]]] = []
    answer: Final = await _complete(
        _cached_history(f"cache creation async={is_async} stream={stream}"),
        _backend_validating_cached_signatures(stream, created, generated),
        is_async,
        stream,
    )
    assert answer == "Berlin is 12C."
    assert [_signatures_of(cache) for cache in created] == [[VERTEX_MINTED]]
    assert len(generated) == 1
    assert "cachedContent" not in generated[0]
    assert _signatures_of(generated[0]) == [get_dummy_thought_signature()]


NATIVE_QUESTION: Final = {"role": "user", "parts": [{"text": "What is the weather in Paris?"}]}
NATIVE_TOOL_RESULT: Final = {
    "role": "user",
    "parts": [{"functionResponse": {"name": "get_weather", "response": {"result": "18C and sunny"}}}],
}


def _native_contents(signature_key: str) -> list[Mapping[str, object]]:
    return [
        NATIVE_QUESTION,
        {
            "role": "model",
            "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}, signature_key: VERTEX_MINTED}],
        },
        NATIVE_TOOL_RESULT,
    ]


async def _generate(
    contents: list[Mapping[str, object]],
    backend: Callable[[httpx.Request], httpx.Response],
    is_async: bool,
    stream: bool,
) -> str:
    kwargs: Final = {"model": MODEL, "contents": contents, "api_key": "unit-test-key"}
    sync_client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(backend)))
    async_client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(backend))
    match is_async, stream:
        case True, True:
            chunks = await agenerate_content_stream(**kwargs, client=async_client)
            return b"".join([chunk async for chunk in chunks]).decode()
        case True, False:
            return json.dumps((await agenerate_content(**kwargs, client=async_client)).model_dump(mode="json"))
        case False, True:
            return b"".join(generate_content_stream(**kwargs, client=sync_client)).decode()
        case False, False:
            return json.dumps(generate_content(**kwargs, client=sync_client).model_dump(mode="json"))


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("signature_key", ["thoughtSignature", "thought_signature"])
@pytest.mark.asyncio
async def test_native_generate_content_resends_a_rejected_signature_with_the_placeholder(
    is_async: bool, stream: bool, signature_key: str
):
    sent: Final[list[dict[str, object]]] = []
    output: Final = await _generate(
        _native_contents(signature_key),
        _replay((_rejected(AI_STUDIO_REJECTION), _answer(stream)), sent),
        is_async,
        stream,
    )
    assert "Berlin is 12C." in output
    assert len(sent) == 2
    assert _signatures_of(sent[0], signature_key) == [VERTEX_MINTED]
    assert _signatures_of(sent[1], signature_key) == [get_dummy_thought_signature()]
    assert {**sent[1], "contents": None} == {**sent[0], "contents": None}


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_native_generate_content_raises_an_unrelated_400_without_a_resend(is_async: bool, stream: bool):
    sent: Final[list[dict[str, object]]] = []
    with pytest.raises(litellm.BadRequestError, match="Unknown name"):
        await _generate(
            _native_contents("thoughtSignature"), _replay((_rejected(UNRELATED_REJECTION),), sent), is_async, stream
        )
    assert len(sent) == 1


def _history_signed_after_the_cached_prefix(tag: str) -> tuple[Mapping[str, object], ...]:
    prefix: Final = " ".join(f"Reference paragraph {i} for {tag}: weather values are in Celsius." for i in range(600))
    return (
        {"role": "system", "content": [{"type": "text", "text": prefix, "cache_control": CACHE_CONTROL}]},
        {"role": "user", "content": [{"type": "text", "text": "Hello.", "cache_control": CACHE_CONTROL}]},
        *HISTORY_WITH_FOREIGN_SIGNATURES,
    )


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_signature_after_the_cached_prefix_is_resent_with_the_cache_kept(is_async: bool, stream: bool):
    generated: Final[list[dict[str, object]]] = []
    answer: Final = await _complete(
        _history_signed_after_the_cached_prefix(f"async={is_async} stream={stream}"),
        _google_ai_studio(stream, generated),
        is_async,
        stream,
    )
    assert answer == "Berlin is 12C."
    assert len(generated) == 2
    assert generated[1]["cachedContent"] == generated[0]["cachedContent"] == "cachedContents/prefix"
    assert set(_signatures_of(generated[0])) == {VERTEX_MINTED, VERTEX_MINTED_TEXT}
    assert _signatures_of(generated[1]) == [get_dummy_thought_signature()] * 2
