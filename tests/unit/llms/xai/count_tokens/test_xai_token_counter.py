import json
from collections.abc import Callable, Mapping
from typing import Final

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.xai.count_tokens.token_counter import XAITokenCounter
from litellm.types.utils import TokenCountResponse

XAI_DEPLOYMENT: Final = {"litellm_params": {"model": "xai/grok-4.7", "api_key": "xai-deployment-key"}}
AZURE_DEPLOYMENT: Final = {
    "litellm_params": {
        "model": "azure_ai/grok-4-1-fast-reasoning",
        "api_key": "azure-key",
        "api_base": "https://grok.services.ai.azure.com/models",
    }
}
HI: Final = [{"role": "user", "content": "hi"}]


class _XAI:
    def __init__(self, reply: Callable[[httpx.Request], httpx.Response]) -> None:
        self.reply: Final = reply
        self.requests: tuple[httpx.Request, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        return self.reply(request)

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(json.loads(request.content)["text"] for request in self.requests)


def _tokens(count: int) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(
        200,
        json={"token_ids": [{"token_id": i, "string_token": "t", "token_bytes": [116]} for i in range(count)]},
        request=request,
    )


def _counter(xai: _XAI) -> XAITokenCounter:
    return XAITokenCounter(client=AsyncHTTPHandler(transport=httpx.MockTransport(xai)))


@pytest.fixture(autouse=True)
def _no_ambient_xai_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_BASE", raising=False)


@pytest.mark.asyncio
async def test_counts_the_grok_chat_template_render_with_the_deployment_key() -> None:
    xai: Final = _XAI(_tokens(31))

    result: Final = await _counter(xai).count_tokens(
        model_to_use="xai/grok-4.7",
        messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "  first  "},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "quote", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "42"},
        ],
        contents=None,
        deployment=XAI_DEPLOYMENT,
        request_model="grok",
    )

    assert result == TokenCountResponse(
        total_tokens=31, request_model="grok", model_used="grok-4.7", tokenizer_type="xai_api"
    )
    assert xai.texts == (
        "System: Be terse.<|separator|>\n\n"
        "Human: first\nsecond<|separator|>\n\n"
        "Assistant: quote({})<|separator|>\n\n"
        "Human: 42<|separator|>\n\n",
    )
    assert str(xai.requests[0].url) == "https://api.x.ai/v1/tokenize-text"
    assert xai.requests[0].headers["Authorization"] == "Bearer xai-deployment-key"
    assert json.loads(xai.requests[0].content)["model"] == "grok-4.7"


@pytest.mark.asyncio
async def test_anthropic_request_renders_tools_and_system_ahead_of_the_turns() -> None:
    xai: Final = _XAI(_tokens(9))

    result: Final = await _counter(xai).count_tokens(
        model_to_use="grok-4.7",
        messages=[{"role": "user", "content": [{"type": "text", "text": "price?"}]}],
        contents=None,
        deployment=XAI_DEPLOYMENT,
        tools=[{"name": "quote", "description": "Stock quote", "input_schema": {"type": "object"}}],
        system="Be terse.",
        messages_format="anthropic",
    )

    assert result is not None and result.total_tokens == 9
    tool_turn, system_turn, user_turn, trailer = xai.texts[0].split("<|separator|>\n\n")
    assert json.loads(tool_turn.removeprefix("System: "))[0]["function"]["name"] == "quote"
    assert (system_turn, user_turn, trailer) == ("System: Be terse.", "Human: price?", "")


@pytest.mark.parametrize(
    "deployment",
    [
        AZURE_DEPLOYMENT,
        {"litellm_params": {"model": "my-grok", "api_key": "unknown-key", "api_base": "https://my-grok.internal"}},
        {"litellm_params": {"api_key": "modelless-key"}},
        None,
    ],
    ids=["azure_deployment", "unresolvable_model", "no_model", "no_deployment"],
)
@pytest.mark.asyncio
async def test_a_non_xai_deployment_key_and_base_are_never_sent_to_xai(
    monkeypatch: pytest.MonkeyPatch, deployment: Mapping[str, object] | None
) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-env-key")
    xai: Final = _XAI(_tokens(4))

    result: Final = await _counter(xai).count_tokens(
        model_to_use="xai/grok-4-1-fast-reasoning", messages=HI, contents=None, deployment=deployment
    )

    assert result is not None and result.total_tokens == 4
    assert str(xai.requests[0].url) == "https://api.x.ai/v1/tokenize-text"
    assert xai.requests[0].headers["Authorization"] == "Bearer xai-env-key"


@pytest.mark.asyncio
async def test_an_xai_deployment_api_base_receives_the_count() -> None:
    xai: Final = _XAI(_tokens(4))

    await _counter(xai).count_tokens(
        model_to_use="xai/grok-4.7",
        messages=HI,
        contents=None,
        deployment={"litellm_params": {**XAI_DEPLOYMENT["litellm_params"], "api_base": "https://xai.internal/v1/"}},
    )

    assert str(xai.requests[0].url) == "https://xai.internal/v1/tokenize-text"


@pytest.mark.asyncio
async def test_without_an_xai_key_it_answers_401_and_sends_nothing() -> None:
    xai: Final = _XAI(_tokens(4))

    result: Final = await _counter(xai).count_tokens(
        model_to_use="xai/grok-4-1-fast-reasoning",
        messages=HI,
        contents=None,
        deployment=AZURE_DEPLOYMENT,
        request_model="azure-grok",
    )

    assert result is not None
    assert (result.error, result.status_code) == (True, 401)
    assert result.error_message is not None and "XAI_API_KEY" in result.error_message
    assert xai.requests == ()


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadError("connection reset", request=request)


@pytest.mark.parametrize(
    ("reply", "status_code", "message"),
    [
        (lambda request: httpx.Response(429, text="slow down", request=request), 429, "slow down"),
        (lambda request: httpx.Response(503, request=request), 503, "answered 503"),
        (lambda request: httpx.Response(200, json={"tokens": []}, request=request), 502, "ValidationError"),
        (lambda request: httpx.Response(200, text="<html>", request=request), 502, "JSONDecodeError"),
        (_refuse, 502, "ReadError"),
    ],
    ids=["provider_error", "empty_error_body", "unexpected_shape", "not_json", "unreachable"],
)
@pytest.mark.asyncio
async def test_a_failed_count_answers_an_error_with_its_status(
    reply: Callable[[httpx.Request], httpx.Response], status_code: int, message: str
) -> None:
    result: Final = await _counter(_XAI(reply)).count_tokens(
        model_to_use="xai/grok-4.7", messages=HI, contents=None, deployment=XAI_DEPLOYMENT
    )

    assert result is not None
    assert (result.error, result.status_code, result.total_tokens) == (True, status_code, 0)
    assert result.error_message is not None and message in result.error_message


@pytest.mark.parametrize(
    ("messages", "contents"),
    [
        ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}], None),
        (HI, [{"parts": [{"text": "hi"}]}]),
        (None, None),
        ([], None),
    ],
    ids=["image_part", "gemini_contents", "no_messages", "empty_messages"],
)
@pytest.mark.asyncio
async def test_requests_it_cannot_render_fall_back_to_the_local_estimate(
    messages: list[Mapping[str, object]] | None, contents: list[Mapping[str, object]] | None
) -> None:
    xai: Final = _XAI(_tokens(4))

    result: Final = await _counter(xai).count_tokens(
        model_to_use="xai/grok-4.7", messages=messages, contents=contents, deployment=XAI_DEPLOYMENT
    )

    assert result is None
    assert xai.requests == ()


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": ["hi"]}],
        [{"content": "hi"}],
        [{"role": "user", "content": [{"type": "image", "source": "not-a-source"}]}],
    ],
    ids=["adapter_rejects_a_bare_string_block", "adapter_rejects_a_missing_role", "nothing_left_after_translation"],
)
@pytest.mark.asyncio
async def test_anthropic_requests_it_cannot_translate_fall_back_to_the_local_estimate(
    messages: list[Mapping[str, object]],
) -> None:
    xai: Final = _XAI(_tokens(4))

    result: Final = await _counter(xai).count_tokens(
        model_to_use="xai/grok-4.7",
        messages=messages,
        contents=None,
        deployment=XAI_DEPLOYMENT,
        messages_format="anthropic",
    )

    assert result is None
    assert xai.requests == ()
