"""Live e2e: POST /v1/videos creates a video and serves its content.

Registers a fal.ai Seedance deployment at runtime, polls the queued video until it
completes, and asserts the generated content is returned as binary data.
"""

from __future__ import annotations

import time
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import require_successful_call, unwrap
from endpoints_client import EndpointsClient, VideoObject
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

_POLL_INTERVAL_SECONDS: Final[float] = 5.0
_POLL_TIMEOUT_SECONDS: Final[float] = 600.0


def _wait_for_completion(
    endpoints_client: EndpointsClient, key: str, created: VideoObject
) -> VideoObject:
    deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status = unwrap(endpoints_client.video_status(key, created.id))
        assert status.id == created.id
        if status.status == "completed":
            return status
        if status.status == "failed":
            pytest.fail(f"fal.ai video generation failed: {status}")
        time.sleep(_POLL_INTERVAL_SECONDS)
    pytest.fail(f"fal.ai video {created.id!r} did not complete within {_POLL_TIMEOUT_SECONDS}s")


class TestVideoGeneration:
    @pytest.mark.covers("llm.videos.fal_ai.basic.nonstream.works")
    def test_fal_seedance_video_completes_and_downloads(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-fal-video-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="fal_ai/bytedance/seedance-2.5/text-to-video",
                api_key="os.environ/FAL_AI_API_KEY",
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.videos(
            key, model, "a red fox running through snow at dawn"
        )
        require_successful_call(result)
        created = VideoObject.model_validate_json(result.body)
        assert created.id
        assert created.model

        _wait_for_completion(endpoints_client, key, created)

        content = endpoints_client.video_content(key, created.id)
        require_successful_call(content)
        assert len(content.body) > 0
        assert not (content.content_type or "").startswith("application/json")
