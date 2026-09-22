"""Live e2e: POST /v1/videos on fal.ai MiniMax H3 accepts the request shapes the
translation layer claims to support.

Registers `fal_ai/minimax/h3/text-to-video` at runtime and drives it through the
real OpenAI SDK. `seconds="auto"` and an oversized `size` both map to a provider
request, so anything other than a queued job is a proxy crash rather than a
provider rejection: before LIT-8339 neither request reached fal.ai.
"""

from __future__ import annotations

import pytest
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from openai import APIStatusError
from openai.types import Video
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e

H3_BACKEND = "fal_ai/minimax/h3/text-to-video"
PROMPT = "A paper boat drifting down a quiet stream at dawn"
OVERSIZED_SIDE = "9" * 30


def _register_h3(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str]:
    model = f"e2e-fal-h3-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(model=H3_BACKEND, api_key="os.environ/FAL_AI_API_KEY"),
        provider_live=True,
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _create_video(sdk: SdkClients, key: str, model: str, seconds: str, size: str) -> Video:
    try:
        return sdk.openai(key).videos.create(
            model=model,
            prompt=PROMPT,
            seconds=seconds,  # pyright: ignore[reportArgumentType]  # the proxy accepts "auto", the SDK literal does not
            size=size,  # pyright: ignore[reportArgumentType]  # arbitrary WxH is the shape under test
        )
    except APIStatusError as error:
        pytest.fail(
            f"seconds={seconds!r} size={size!r}: /v1/videos returned {error.status_code}: {error.message[:300]}"
        )


class TestFalAiH3Videos:
    @pytest.mark.covers("llm.videos.fal_ai.basic.nonstream.works")
    def test_auto_duration_queues_h3_job(self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients) -> None:
        model, key = _register_h3(proxy, resources)
        video = _create_video(sdk, key, model, seconds="auto", size="854x480")
        assert video.id, f"/v1/videos queued no job for seconds='auto': {video!r}"

    @pytest.mark.covers("llm.videos.fal_ai.basic.nonstream.works")
    def test_oversized_size_falls_through_to_top_resolution_tier(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register_h3(proxy, resources)
        video = _create_video(sdk, key, model, seconds="5", size=f"{OVERSIZED_SIDE}x{OVERSIZED_SIDE}")
        assert video.id, f"/v1/videos queued no job for an oversized size: {video!r}"
