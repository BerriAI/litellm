import base64
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import get_dummy_thought_signature
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.gemini.rejected_thought_signature_retry import with_placeholder_thought_signatures
from litellm.types.llms.vertex_ai import RequestBody
from litellm.types.utils import Choices, ModelResponse, ModelResponseStream
from litellm.utils import CustomStreamWrapper

MODEL: Final = "gemini/gemini-3.8-flash"
VERTEX_MINTED: Final = base64.b64encode(b"minted-by-vertex").decode()
VERTEX_MINTED_TEXT: Final = base64.b64encode(b"minted-by-vertex-for-text").decode()
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


def _signatures_of(body: Mapping[str, object]) -> list[str]:
    contents: Final = body["contents"]
    assert isinstance(contents, list)
    parts: Final = [part for content in contents for part in content["parts"]]
    return [part["thoughtSignature"] for part in parts if "thoughtSignature" in part]


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
    answers: Sequence[httpx.Response],
    sent: list[dict[str, object]],
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
    if is_async:
        response = await litellm.acompletion(
            **kwargs, client=AsyncHTTPHandler(transport=httpx.MockTransport(_replay(answers, sent)))
        )
    else:
        response = litellm.completion(
            **kwargs, client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_replay(answers, sent))))
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
        HISTORY_WITH_FOREIGN_SIGNATURES, (_rejected(rejection), _answer(stream)), sent, is_async, stream
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
            (_rejected(AI_STUDIO_REJECTION), _rejected(AI_STUDIO_REJECTION)),
            sent,
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
        await _complete(HISTORY_WITH_FOREIGN_SIGNATURES, (_rejected(UNRELATED_REJECTION),), sent, is_async, stream)
    assert len(sent) == 1


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_signature_400_on_a_history_carrying_only_the_placeholder_is_raised_without_a_resend(
    is_async: bool, stream: bool
):
    sent: Final[list[dict[str, object]]] = []
    with pytest.raises(litellm.BadRequestError, match="thought signature"):
        await _complete(HISTORY_WITHOUT_SIGNATURES, (_rejected(AI_STUDIO_REJECTION),), sent, is_async, stream)
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
