import asyncio
from collections.abc import Awaitable, Callable
from types import MappingProxyType
from typing import Final

import openai
import pytest
from integration._support.idle_gateway import idle_gateway
from integration.providers._idle_upstream_support import (
    DROP_AFTER_SECONDS,
    IDLE_SECONDS,
    PROVIDER_KEY,
    answer,
    expected_text,
    landed,
    new_marker,
    stay_idle,
)

import litellm
from litellm.types.utils import Choices, ModelResponse


def _text(response: object) -> str:
    assert isinstance(response, ModelResponse), response
    choice: Final = response.choices[0]
    assert isinstance(choice, Choices), choice
    return choice.message.content or ""


async def _chat(model: str, api_base: str, marker: str) -> str:
    return _text(
        await litellm.acompletion(
            model=model,
            api_base=api_base,
            api_key=PROVIDER_KEY,
            messages=[{"role": "user", "content": marker}],
            num_retries=0,
            max_retries=0,
        )
    )


async def openai_chat(api_base: str, marker: str) -> str:
    return await _chat("openai/idle-upstream", api_base, marker)


async def anthropic_chat(api_base: str, marker: str) -> str:
    return await _chat("anthropic/claude-idle-upstream", api_base, marker)


async def mistral_ocr_rust_bridge(api_base: str, marker: str) -> str:
    response: Final = await litellm.aocr(
        model="mistral/mistral-ocr-latest",
        document={"type": "document_url", "document_url": f"https://documents.invalid/{marker}.pdf"},
        api_base=api_base,
        api_key=PROVIDER_KEY,
    )
    return response.pages[0].markdown


CALLS: Final[MappingProxyType[str, Callable[[str, str], Awaitable[str]]]] = MappingProxyType(
    {
        "acompletion-openai": openai_chat,
        "acompletion-anthropic": anthropic_chat,
        "aocr-mistral-rust-bridge": mistral_ocr_rust_bridge,
    }
)


async def _outcome(call: Callable[[str, str], Awaitable[str]], api_base: str, marker: str) -> str:
    try:
        return await call(api_base, marker)
    except openai.APIError as error:
        return f"raised {type(error).__name__}: {error}"


@pytest.mark.timeout(int(IDLE_SECONDS + 120))
@pytest.mark.parametrize("name", list(CALLS))
async def test_sdk_call_after_upstream_idle_past_keepalive_opens_a_fresh_connection(name: str) -> None:
    call: Final = CALLS[name]
    warm_marker: Final = new_marker()
    probe_marker: Final = new_marker()
    with idle_gateway(answer, drop_after=DROP_AFTER_SECONDS) as upstream:
        warm: Final = await _outcome(call, upstream.url, warm_marker)
        warmed: Final = upstream.drain()
        await asyncio.to_thread(stay_idle, (upstream,), IDLE_SECONDS)
        probe: Final = await _outcome(call, upstream.url, probe_marker)
        arrivals: Final = upstream.drain()
    assert (warm, probe) == (expected_text(warm_marker), expected_text(probe_marker)), (warm, probe)
    assert landed((*warmed, *arrivals)) == tuple(sorted(((warm_marker, False), (probe_marker, False)))), (
        warmed,
        arrivals,
    )
