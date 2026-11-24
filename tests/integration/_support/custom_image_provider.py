"""A customer-owned image provider for the proxy's `custom_provider_map` that relays what it was handed to its host."""

from collections.abc import Mapping, Sequence
from typing import Final

import httpx

from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ImageResponse

_RELAYED: Final = ("n", "size", "quality", "response_format", "user", "studio_preset")


async def _relay(
    route: str, api_base: str | None, model: str, prompt: str | None, optional_params: Mapping[str, object]
) -> ImageResponse:
    received: Final = {key: optional_params[key] for key in _RELAYED if key in optional_params}
    async with httpx.AsyncClient() as client:
        reply: Final = await client.post(
            f"{api_base}/{route}",
            json={"model": model, "prompt": prompt, "received": received, "has_mask": "mask" in optional_params},
        )
    return ImageResponse(created=0, data=reply.json()["data"])


class RelayImageProvider(CustomLLM):
    async def aimage_generation(
        self, model: str, prompt: str, api_base: str | None, optional_params: Mapping[str, object], **kwargs: object
    ) -> ImageResponse:
        return await _relay("render", api_base, model, prompt, optional_params)

    async def aimage_edit(
        self,
        model: str,
        image: Sequence[object],
        prompt: str | None,
        api_base: str | None,
        optional_params: Mapping[str, object],
        **kwargs: object,
    ) -> ImageResponse:
        return await _relay("retouch", api_base, model, prompt, optional_params)


IMAGE_HANDLER: Final = RelayImageProvider()
