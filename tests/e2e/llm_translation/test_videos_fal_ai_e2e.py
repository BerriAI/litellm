"""Live e2e: POST /v1/videos on fal.ai MiniMax H3 accepts the request shapes the
translation layer claims to support, and fal.ai then runs the job.

Registers `fal_ai/minimax/h3/text-to-video` at runtime and drives it through the
real OpenAI SDK. fal.ai validates the body after queueing, so a queued id alone
proves nothing: each test also waits for the job to leave `queued` without failing.
Before LIT-8339 `seconds="auto"` and an oversized `size` both crashed the proxy.
"""

from __future__ import annotations

import time

import pytest
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from openai import APIStatusError, OpenAI
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


def _create_video(client: OpenAI, model: str, seconds: str, size: str) -> Video:
    try:
        return client.videos.create(
            model=model,
            prompt=PROMPT,
            seconds=seconds,  # pyright: ignore[reportArgumentType]  # the proxy accepts "auto", the SDK literal does not
            size=size,  # pyright: ignore[reportArgumentType]  # arbitrary WxH is the shape under test
        )
    except APIStatusError as error:
        pytest.fail(
            f"seconds={seconds!r} size={size!r}: /v1/videos returned {error.status_code}: {error.message[:300]}"
        )


def _assert_job_runs(client: OpenAI, proxy: ProxyClient, video: Video, context: str) -> None:
    deadline = time.monotonic() + proxy.poll_timeout
    while True:
        current = client.videos.retrieve(video.id)
        assert current.status != "failed", f"{context}: fal.ai failed the job: {current.error!r}"
        if current.status != "queued":
            return
        assert time.monotonic() < deadline, f"{context}: job still queued after {proxy.poll_timeout}s: {current!r}"
        time.sleep(proxy.poll_interval)


class TestFalAiH3Videos:
    @pytest.mark.covers("llm.videos.fal_ai.basic.nonstream.works")
    def test_auto_duration_runs_h3_job(self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients) -> None:
        model, key = _register_h3(proxy, resources)
        client = sdk.openai(key)
        video = _create_video(client, model, seconds="auto", size="854x480")
        _assert_job_runs(client, proxy, video, 'seconds="auto"')

    @pytest.mark.covers("llm.videos.fal_ai.basic.nonstream.works")
    def test_oversized_size_runs_h3_job_at_top_resolution_tier(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register_h3(proxy, resources)
        client = sdk.openai(key)
        video = _create_video(client, model, seconds="5", size=f"{OVERSIZED_SIDE}x{OVERSIZED_SIDE}")
        _assert_job_runs(client, proxy, video, "oversized size")
