from __future__ import annotations

import pytest
from e2e_config import unique_marker
from e2e_http import Success, UnknownApiError
from guardrails_client import GuardrailsClient, poll_until_blocked
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

CHAT_MODEL = "gemini-2.5-flash"
VIDEO_BACKEND = "vertex_ai/veo-3.1-fast-generate-001"


def _video_prompt_with(banned_keyword: str) -> str:
    return f"A short clip of a paper boat floating down a stream. {banned_keyword}"


def _create_video_model(client: GuardrailsClient, resources: ResourceManager) -> str:
    model_name = f"e2e-guard-video-{unique_marker()}"
    model_id = client.proxy.create_model(
        model_name,
        LiteLLMParamsBody(
            model=VIDEO_BACKEND,
            vertex_project="os.environ/VERTEXAI_PROJECT",
            vertex_location="os.environ/VERTEXAI_LOCATION",
            vertex_credentials="os.environ/VERTEXAI_CREDENTIALS",
        ),
        provider_live=True,
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model_name


class TestKeyAttachedGuardrailOnVideos:
    @pytest.mark.covers(
        "guardrail.litellm_content_filter.pre_call.blocks_video",
        exercised_on=["videos"],
    )
    def test_key_attached_content_filter_blocks_banned_video_prompt(
        self, client: GuardrailsClient, resources: ResourceManager
    ) -> None:
        banned = unique_marker()
        guardrail_name = f"e2e-video-filter-{banned}"
        guardrail_id = client.create_content_filter_guardrail(guardrail_name, banned, default_on=False)
        resources.defer(lambda: client.delete_guardrail(guardrail_id))
        key = client.create_key_with_guardrails(resources, [guardrail_name])
        model = _create_video_model(client, resources)

        synced = poll_until_blocked(lambda: client.chat(key, CHAT_MODEL, _video_prompt_with(banned)))
        assert isinstance(synced, UnknownApiError) and synced.status_code == 400, (
            f"key guardrail {guardrail_name!r} never synced to the proxy on /chat/completions: {synced}"
        )

        result = client.create_video(key, model, _video_prompt_with(banned))
        match result:
            case UnknownApiError(status_code=status, body=body):
                assert status == 400, f"expected a 400 guardrail block, got {status}: {body[:300]}"
                assert "content blocked" in body.lower() or banned in body, (
                    f"block response missing content-filter reason: {body[:300]}"
                )
            case Success(data=video):
                pytest.fail(
                    f"key-attached guardrail {guardrail_name!r} was skipped on /v1/videos: "
                    f"the banned prompt reached the provider and started video job {video.id}"
                )
            case _:
                pytest.fail(f"unexpected /v1/videos outcome for a banned prompt: {result}")
