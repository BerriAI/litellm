"""deAPI `/v1/videos`: OpenAI's video jobs API. A queued job has no cost yet, and the status read of a
completed job reports the settled charge in `usage.cost` and in `X-DeAPI-Cost`."""

import httpx
import pytest

import litellm
from litellm.llms.deapi.videos.transformation import DeapiVideoConfig
from litellm.types.utils import LlmProviders
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider
from litellm.utils import ProviderConfigManager

DEAPI_VIDEOS_URL = "https://oai.deapi.ai/v1/videos"
MODEL = "deapi/Ltx2_5_22B_Dist_INT8"
SELLER_MODEL = "Ltx2_5_22B_Dist_INT8"
JOB_ID = "video_7d0bb073aa6c416b"
SETTLED_COST = 0.046998
PROMPT = "a red ball rolling on a wooden table"
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42"


def _video(status: str = "queued", **overrides: object) -> dict:
    return {
        "id": JOB_ID,
        "object": "video",
        "status": status,
        "progress": 100 if status == "completed" else 0,
        "created_at": 1795859000,
        "completed_at": 1795859060 if status == "completed" else None,
        "model": SELLER_MODEL,
        "seconds": "4",
        "size": "768x512",
        **overrides,
    }


def _encoded() -> str:
    return encode_video_id_with_provider(JOB_ID, "deapi", SELLER_MODEL)


class TestRegistration:
    def test_deapi_is_a_native_video_provider(self):
        config = ProviderConfigManager.get_provider_video_config(model=SELLER_MODEL, provider=LlmProviders.DEAPI)

        assert isinstance(config, DeapiVideoConfig)

    def test_missing_key_is_an_authentication_error_before_any_request(self, no_deapi_key, respx_mock):
        with pytest.raises(litellm.AuthenticationError, match="DEAPI_API_KEY"):
            litellm.video_generation(model=MODEL, prompt=PROMPT)
        assert not respx_mock.calls


class TestCreate:
    def test_posts_the_job_with_the_bearer_key_and_routes_the_id_back_to_deapi(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_VIDEOS_URL).mock(return_value=httpx.Response(200, json=_video()))

        response = litellm.video_generation(model=MODEL, prompt=PROMPT, seconds="4")

        request = respx_mock.calls.last.request
        assert request.headers["Authorization"] == f"Bearer {deapi_key}"
        assert PROMPT.encode() in request.content
        assert SELLER_MODEL.encode() in request.content
        decoded = decode_video_id_with_provider(response.id)
        assert (decoded["video_id"], decoded["custom_llm_provider"]) == (JOB_ID, "deapi")
        assert "provider_reported_cost_usd" not in (response.usage or {})


class TestStatus:
    def test_a_completed_job_surfaces_the_settled_cost(self, deapi_key, respx_mock):
        respx_mock.get(f"{DEAPI_VIDEOS_URL}/{JOB_ID}").mock(
            return_value=httpx.Response(
                200,
                json=_video(status="completed", usage={"seconds": 4, "cost": SETTLED_COST}),
                headers={"X-DeAPI-Cost": f"{SETTLED_COST:.8f}"},
            )
        )

        response = litellm.video_status(video_id=_encoded())

        assert respx_mock.calls.last.request.headers["Authorization"] == f"Bearer {deapi_key}"
        assert (response.status, response.progress) == ("completed", 100)
        assert response.usage["provider_reported_cost_usd"] == SETTLED_COST

    def test_a_running_job_has_no_cost_yet(self, deapi_key, respx_mock):
        respx_mock.get(f"{DEAPI_VIDEOS_URL}/{JOB_ID}").mock(
            return_value=httpx.Response(200, json=_video(status="in_progress"))
        )

        response = litellm.video_status(video_id=_encoded())

        assert "provider_reported_cost_usd" not in (response.usage or {})

    def test_an_unknown_job_is_an_error(self, deapi_key, respx_mock):
        envelope = {"error": {"message": "Video not found.", "type": "invalid_request_error", "code": "not_found"}}
        respx_mock.get(f"{DEAPI_VIDEOS_URL}/{JOB_ID}").mock(return_value=httpx.Response(404, json=envelope))

        with pytest.raises(Exception, match="404|not found|Not Found"):
            litellm.video_status(video_id=_encoded())


class TestContent:
    def test_downloads_the_finished_file(self, deapi_key, respx_mock):
        respx_mock.get(f"{DEAPI_VIDEOS_URL}/{JOB_ID}/content").mock(
            return_value=httpx.Response(200, content=MP4_BYTES, headers={"content-type": "video/mp4"})
        )

        assert litellm.video_content(video_id=_encoded()) == MP4_BYTES
