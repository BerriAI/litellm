"""Eden AI `/v3/videos`: OpenAI's video jobs API served by Eden's gateway, which reports `cost` as 0
while a job is queued and the settled amount on the status read once it completes."""

import json
from io import BytesIO

import httpx
import pytest

import litellm
from litellm.llms.edenai.videos.transformation import EdenAIVideoConfig
from litellm.types.utils import LlmProviders
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider
from litellm.utils import ProviderConfigManager

EDEN_BASE = "https://api.edenai.run/v3"
EDEN_VIDEOS_URL = f"{EDEN_BASE}/videos"
MODEL = "edenai/pruna/p-video"
SELLER_MODEL = "pruna/p-video"
JOB_ID = "fcd74ecd-23df-4eea-a372-478a1e842d42"
SETTLED_COST = 0.08
FILE_URL = "https://files.example.net/60b11f54/video.mp4"
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42"
PROMPT = "a red ball rolling on a wooden table"


def _eden_video(status: str = "queued", cost: float = 0.0, **overrides: object) -> dict:
    """Live `/v3/videos` body: OpenAI's video object plus Eden's top-level `provider` and `cost`."""
    return {
        "id": JOB_ID,
        "object": "video",
        "status": status,
        "progress": 100 if status == "completed" else 0,
        "created_at": 1789067483,
        "completed_at": 1789067493 if status == "completed" else None,
        "expires_at": None,
        "model": SELLER_MODEL,
        "seconds": "4",
        "size": "1280x720",
        "remixed_from_video_id": None,
        "error": None,
        "provider": "pruna",
        "cost": cost,
        **overrides,
    }


def _encoded(job_id: str = JOB_ID) -> str:
    return encode_video_id_with_provider(job_id, "edenai", SELLER_MODEL)


def _request_body(respx_mock) -> dict:
    return json.loads(respx_mock.calls.last.request.content)


class TestRegistration:
    def test_eden_is_a_native_video_provider(self):
        config = ProviderConfigManager.get_provider_video_config(model=SELLER_MODEL, provider=LlmProviders.EDENAI)

        assert isinstance(config, EdenAIVideoConfig)


class TestAuthentication:
    def test_missing_key_is_an_authentication_error_before_any_request(self, no_eden_key, respx_mock):
        with pytest.raises(litellm.AuthenticationError, match="EDENAI_API_KEY"):
            litellm.video_generation(model=MODEL, prompt=PROMPT)
        assert not respx_mock.calls


class TestCreate:
    def test_posts_json_to_eden_with_the_bearer_key_and_the_seller_model_id(self, eden_key, respx_mock):
        respx_mock.post(EDEN_VIDEOS_URL).mock(return_value=httpx.Response(200, json=_eden_video()))

        response = litellm.video_generation(model=MODEL, prompt=PROMPT, seconds="4", size="1280x720")

        assert isinstance(response, VideoObject)
        assert response.status == "queued"
        request = respx_mock.calls.last.request
        assert request.headers["Authorization"] == f"Bearer {eden_key}"
        assert request.headers["Content-Type"] == "application/json"
        assert json.loads(request.content) == {
            "model": SELLER_MODEL,
            "prompt": PROMPT,
            "seconds": "4",
            "size": "1280x720",
        }

    def test_the_returned_id_routes_later_calls_back_to_eden(self, eden_key, respx_mock):
        respx_mock.post(EDEN_VIDEOS_URL).mock(return_value=httpx.Response(200, json=_eden_video()))

        response = litellm.video_generation(model=MODEL, prompt=PROMPT)

        assert decode_video_id_with_provider(response.id) == {
            "custom_llm_provider": "edenai",
            "model_id": SELLER_MODEL,
            "video_id": JOB_ID,
        }

    def test_eden_extensions_go_through_as_kwargs_and_extra_body(self, eden_key, respx_mock):
        respx_mock.post(EDEN_VIDEOS_URL).mock(return_value=httpx.Response(200, json=_eden_video()))

        litellm.video_generation(model=MODEL, prompt=PROMPT, seed=7, extra_body={"provider_params": {"guidance": 2}})

        body = _request_body(respx_mock)
        assert (body["seed"], body["provider_params"]) == (7, {"guidance": 2})

    def test_a_reference_image_file_makes_the_request_multipart(self, eden_key, respx_mock):
        respx_mock.post(EDEN_VIDEOS_URL).mock(return_value=httpx.Response(200, json=_eden_video()))
        reference = BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

        litellm.video_generation(model=MODEL, prompt="animate this", input_reference=reference, seconds="4")

        request = respx_mock.calls.last.request
        assert request.headers["Content-Type"].startswith("multipart/form-data")
        assert b'name="input_reference"; filename="input_reference.png"' in request.content
        assert b'name="model"\r\n\r\n' + SELLER_MODEL.encode() in request.content
        assert b'name="seconds"\r\n\r\n4' in request.content

    def test_a_reference_image_url_stays_in_the_json_body(self, eden_key, respx_mock):
        respx_mock.post(EDEN_VIDEOS_URL).mock(return_value=httpx.Response(200, json=_eden_video()))

        litellm.video_generation(
            model=MODEL, prompt="animate this", input_reference={"image_url": "https://img.example.net/start.png"}
        )

        request = respx_mock.calls.last.request
        assert request.headers["Content-Type"] == "application/json"
        assert json.loads(request.content)["input_reference"] == {"image_url": "https://img.example.net/start.png"}

    def test_a_queued_job_reports_edens_zero_cost_and_the_requested_duration(self, eden_key, respx_mock):
        respx_mock.post(EDEN_VIDEOS_URL).mock(return_value=httpx.Response(200, json=_eden_video()))

        response = litellm.video_generation(model=MODEL, prompt=PROMPT, seconds="4")

        assert response.usage == {"duration_seconds": 4.0, "provider_reported_cost_usd": 0.0}

    @pytest.mark.asyncio
    async def test_a_queued_job_bills_nothing_until_it_settles(
        self, eden_key, httpx_transport, respx_mock, spend_capture
    ):
        respx_mock.post(EDEN_VIDEOS_URL).mock(return_value=httpx.Response(200, json=_eden_video()))

        await litellm.avideo_generation(model=MODEL, prompt=PROMPT, seconds="4", litellm_call_id=spend_capture.call_id)
        await spend_capture.settle()

        assert spend_capture.costs == [0.0]

    @pytest.mark.asyncio
    async def test_a_cost_settled_on_the_create_response_is_billed(
        self, eden_key, httpx_transport, respx_mock, spend_capture
    ):
        respx_mock.post(EDEN_VIDEOS_URL).mock(
            return_value=httpx.Response(200, json=_eden_video(status="completed", cost=SETTLED_COST))
        )

        await litellm.avideo_generation(model=MODEL, prompt=PROMPT, seconds="4", litellm_call_id=spend_capture.call_id)
        await spend_capture.settle()

        assert spend_capture.costs == [SETTLED_COST]


class TestStatus:
    def test_reads_the_job_with_the_bearer_key_and_surfaces_the_settled_cost(self, eden_key, respx_mock):
        respx_mock.get(f"{EDEN_VIDEOS_URL}/{JOB_ID}").mock(
            return_value=httpx.Response(
                200, json=_eden_video(status="completed", cost=SETTLED_COST, seconds=None, size=None)
            )
        )

        response = litellm.video_status(video_id=_encoded())

        assert respx_mock.calls.last.request.headers["Authorization"] == f"Bearer {eden_key}"
        assert (response.status, response.progress) == ("completed", 100)
        assert response.usage == {"provider_reported_cost_usd": SETTLED_COST}
        assert decode_video_id_with_provider(response.id)["video_id"] == JOB_ID

    @pytest.mark.asyncio
    async def test_polling_a_finished_job_does_not_bill_it_again(
        self, eden_key, httpx_transport, respx_mock, spend_capture
    ):
        respx_mock.get(f"{EDEN_VIDEOS_URL}/{JOB_ID}").mock(
            return_value=httpx.Response(200, json=_eden_video(status="completed", cost=SETTLED_COST))
        )

        await litellm.avideo_status(video_id=_encoded(), litellm_call_id=spend_capture.call_id)
        await spend_capture.settle()

        assert len(spend_capture.costs) == 1
        assert not spend_capture.costs[0]

    def test_an_unknown_job_is_a_not_found_error(self, eden_key, respx_mock):
        respx_mock.get(f"{EDEN_VIDEOS_URL}/{JOB_ID}").mock(
            return_value=httpx.Response(
                404,
                json={
                    "error": {
                        "message": f"Video {JOB_ID} not found",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "model_not_found",
                    }
                },
            )
        )

        with pytest.raises(litellm.NotFoundError, match="not found"):
            litellm.video_status(video_id=_encoded())


class TestContent:
    def test_follows_edens_redirect_to_the_file_without_forwarding_the_key(self, eden_key, respx_mock):
        respx_mock.get(f"{EDEN_VIDEOS_URL}/{JOB_ID}/content").mock(
            return_value=httpx.Response(302, headers={"location": FILE_URL})
        )
        respx_mock.get(FILE_URL).mock(
            return_value=httpx.Response(200, content=MP4_BYTES, headers={"content-type": "binary/octet-stream"})
        )

        video = litellm.video_content(video_id=_encoded())

        assert video == MP4_BYTES
        eden_request, file_request = (call.request for call in respx_mock.calls)
        assert eden_request.headers["Authorization"] == f"Bearer {eden_key}"
        assert "Authorization" not in file_request.headers

    @pytest.mark.asyncio
    async def test_async_download_follows_the_same_redirect(self, eden_key, httpx_transport, respx_mock):
        respx_mock.get(f"{EDEN_VIDEOS_URL}/{JOB_ID}/content").mock(
            return_value=httpx.Response(302, headers={"location": FILE_URL})
        )
        respx_mock.get(FILE_URL).mock(return_value=httpx.Response(200, content=MP4_BYTES))

        assert await litellm.avideo_content(video_id=_encoded()) == MP4_BYTES


class TestList:
    def test_lists_jobs_newest_first_with_encoded_ids_and_their_costs(self, eden_key, httpx_transport, respx_mock):
        """The sync entry point runs the async handler, so the client must sit on httpx for respx to see it."""
        older = "d544c281-9099-487e-b537-5f2291b603c8"
        respx_mock.get(host="api.edenai.run", path="/v3/videos").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        _eden_video(status="completed", cost=0.02, seconds=None, size=None),
                        _eden_video(status="completed", cost=0.1, id=older, seconds=None, size=None),
                    ],
                    "first_id": JOB_ID,
                    "last_id": older,
                    "has_more": True,
                },
            )
        )

        page = litellm.video_list(custom_llm_provider="edenai", limit=2)

        assert respx_mock.calls.last.request.url.params["limit"] == "2"
        assert [decode_video_id_with_provider(video["id"])["video_id"] for video in page["data"]] == [JOB_ID, older]
        assert [video["cost"] for video in page["data"]] == [0.02, 0.1]
        assert decode_video_id_with_provider(page["last_id"]) == {
            "custom_llm_provider": "edenai",
            "model_id": SELLER_MODEL,
            "video_id": older,
        }


class TestErrors:
    def test_middleware_401_maps_to_authentication_error(self, eden_key, respx_mock):
        respx_mock.post(EDEN_VIDEOS_URL).mock(return_value=httpx.Response(401, json={"detail": "Invalid token"}))

        with pytest.raises(litellm.AuthenticationError, match="Invalid token"):
            litellm.video_generation(model=MODEL, prompt=PROMPT)

    def test_a_401_on_a_read_is_an_authentication_error_too(self, eden_key, httpx_transport, respx_mock):
        respx_mock.get(host="api.edenai.run", path="/v3/videos").mock(
            return_value=httpx.Response(401, json={"detail": "Invalid token"})
        )
        respx_mock.get(f"{EDEN_VIDEOS_URL}/{JOB_ID}/content").mock(
            return_value=httpx.Response(401, json={"detail": "Invalid token"})
        )

        with pytest.raises(litellm.AuthenticationError, match="Invalid token"):
            litellm.video_list(custom_llm_provider="edenai")
        with pytest.raises(litellm.AuthenticationError, match="Invalid token"):
            litellm.video_content(video_id=_encoded())

    def test_an_openai_param_eden_does_not_accept_yet_is_forwarded_and_eden_answers(self, eden_key, respx_mock):
        """OpenAI's full video param set goes through untouched, so Eden's own validation is what a caller
        sees today and nothing here needs to change once Eden accepts these fields."""
        respx_mock.post(EDEN_VIDEOS_URL).mock(
            return_value=httpx.Response(
                422,
                json={
                    "error": {
                        "message": "Extra inputs are not permitted",
                        "type": "invalid_request_error",
                        "param": "user",
                        "code": "invalid_parameter",
                    }
                },
            )
        )

        with pytest.raises(litellm.BadRequestError, match="Extra inputs"):
            litellm.video_generation(model=MODEL, prompt=PROMPT, user="u1")
        assert _request_body(respx_mock)["user"] == "u1"
