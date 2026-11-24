"""Customer-owned providers for the proxy's `custom_provider_map`, one for each way generateContent reaches them."""

from collections.abc import Mapping
from typing import Final

import httpx

from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ModelResponse


async def _relay(api_base: object, route: str, model: str) -> Mapping[str, object]:
    async with httpx.AsyncClient(trust_env=False) as client:
        reply: Final = await client.post(f"{api_base}/{route}", json={"model": model})
    return reply.json()


class RelayChatProvider(CustomLLM):
    async def acompletion(self, model: str, messages: list[object], api_base: str, **kwargs: object) -> ModelResponse:
        return ModelResponse(**await _relay(api_base, "complete", model))


class RelayGenerateContentProvider(CustomLLM):
    async def agenerate_content(self, model: str, contents: object, **kwargs: object) -> Mapping[str, object]:
        return await _relay(kwargs.get("api_base"), "generate", model)


CHAT_HANDLER: Final = RelayChatProvider()
GENERATE_CONTENT_HANDLER: Final = RelayGenerateContentProvider()
