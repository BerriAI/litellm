import json
from collections.abc import AsyncIterator
from typing import Final

import httpx
import pytest
import pytest_asyncio
import respx

from litellm import Choices, ModelResponse
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.noma.noma import NomaBlockedMessage, NomaGuardrail
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


@pytest_asyncio.fixture
async def guardrail() -> AsyncIterator[NomaGuardrail]:
    instance: Final = NomaGuardrail(
        api_key="test-key",
        api_base="https://noma.test",
        guardrail_name="test-noma",
        event_hook="post_call",
        default_on=True,
        anonymize_input=True,
    )
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient() as client:
        handler.client = client
        instance.async_handler = handler
        yield instance


def classify(request: httpx.Request) -> httpx.Response:
    text: Final[str] = json.loads(request.content)["input"][0]["content"][0]["text"]
    return httpx.Response(200, json={"aggregatedScanResult": text == "blocked", "scanResult": []})


@pytest.mark.asyncio
async def test_later_choice_is_checked(guardrail: NomaGuardrail, respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://noma.test/ai-dr/v2/prompt/scan").mock(side_effect=classify)
    response: Final = ModelResponse(
        choices=[
            Choices(index=0, message={"role": "assistant", "content": "allowed"}),
            Choices(index=1, message={"role": "assistant", "content": "blocked"}),
        ]
    )
    with pytest.raises(NomaBlockedMessage):
        await guardrail.async_post_call_success_hook({}, UserAPIKeyAuth(), response)
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_anonymization_stays_with_its_choice(guardrail: NomaGuardrail, respx_mock: respx.MockRouter) -> None:
    def anonymize(request: httpx.Request) -> httpx.Response:
        text: Final[str] = json.loads(request.content)["input"][0]["content"][0]["text"]
        return httpx.Response(
            200,
            json={
                "aggregatedScanResult": False,
                "scanResult": [
                    {"role": "assistant", "results": {"anonymizedContent": {"anonymized": text + " redacted"}}}
                ],
            },
        )

    route: Final = respx_mock.post("https://noma.test/ai-dr/v2/prompt/scan").mock(side_effect=anonymize)
    response: Final = ModelResponse(
        choices=[
            Choices(index=0, message={"role": "assistant", "content": "first"}),
            Choices(index=1, message={"role": "assistant", "content": "second"}),
            Choices(index=2, message={"role": "assistant", "content": None}),
        ]
    )
    result: Final = await guardrail.async_post_call_success_hook({}, UserAPIKeyAuth(), response)
    assert [choice.message.content for choice in result.choices] == ["first redacted", "second redacted", None]
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_stream_checks_later_choice_before_yielding(
    guardrail: NomaGuardrail, respx_mock: respx.MockRouter
) -> None:
    respx_mock.post("https://noma.test/ai-dr/v2/prompt/scan").mock(side_effect=classify)

    async def chunks() -> AsyncIterator[ModelResponseStream]:
        for index, text in enumerate(("allowed", "blocked")):
            yield ModelResponseStream(
                choices=[StreamingChoices(index=index, delta=Delta(content=text), finish_reason="stop")]
            )

    stream: Final = guardrail.async_post_call_streaming_iterator_hook(UserAPIKeyAuth(), chunks(), {})
    with pytest.raises(NomaBlockedMessage):
        await anext(stream)
