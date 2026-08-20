"""Tests for WaveSpeed AI video generation transformation."""

from unittest.mock import Mock

import httpx
import pytest
import respx

import litellm

from litellm.llms.wavespeed.common_utils import WaveSpeedError
from litellm.llms.wavespeed.videos.transformation import WaveSpeedVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import extract_original_video_id

MODEL = "bytedance/seedance-2.5/text-to-video"
API_BASE = "https://api.wavespeed.ai"
OUTPUT_URL = "https://cdn.wavespeed.ai/pred-123.mp4"


def envelope(data):
    return {"code": 200, "message": "success", "data": data}


def prediction(status, **extra):
    return envelope({"id": "pred-123", "status": status, "created_at": "2026-08-20T10:00:00Z", **extra})


class TestWaveSpeedVideoTransformation:
    def setup_method(self):
        self.config = WaveSpeedVideoConfig()
        self.logging_obj = Mock()

    def test_transform_video_create_request(self):
        data, files, url = self.config.transform_video_create_request(
            model=MODEL,
            prompt="a red panda skateboarding",
            api_base=API_BASE,
            video_create_optional_request_params={"size": "1280*720", "duration": 5},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data == {"prompt": "a red panda skateboarding", "size": "1280*720", "duration": 5}
        assert files == []
        assert url == f"{API_BASE}/api/v3/{MODEL}"

    def test_map_openai_params(self):
        assert self.config.map_openai_params(
            {"size": "1280x720", "seconds": "5", "input_reference": "https://example.com/a.png", "guidance": 3},
            MODEL,
            False,
        ) == {"size": "1280*720", "duration": 5, "image": "https://example.com/a.png", "guidance": 3}

    def test_create_response_maps_to_queued_video_object(self):
        raw = httpx.Response(200, json=prediction("created"))

        video = self.config.transform_video_create_response(
            model=MODEL, raw_response=raw, logging_obj=self.logging_obj, custom_llm_provider="wavespeed"
        )

        assert video.status == "queued"
        assert video.object == "video"
        assert extract_original_video_id(video.id) == "pred-123"
        assert video.model == MODEL

    def test_status_retrieve_response_maps_terminal_statuses(self):
        completed = self.config.transform_video_status_retrieve_response(
            raw_response=httpx.Response(200, json=prediction("completed", outputs=[OUTPUT_URL])),
            logging_obj=self.logging_obj,
        )
        assert completed.status == "completed"

        failed = self.config.transform_video_status_retrieve_response(
            raw_response=httpx.Response(200, json=prediction("failed", error="upstream rejected the prompt")),
            logging_obj=self.logging_obj,
        )
        assert failed.status == "failed"
        assert failed.error["message"] == "upstream rejected the prompt"

    def test_status_retrieve_request_url(self):
        url, params = self.config.transform_video_status_retrieve_request(
            video_id="pred-123", api_base=API_BASE, litellm_params=GenericLiteLLMParams(), headers={}
        )
        assert url == f"{API_BASE}/api/v3/predictions/pred-123/result"
        assert params == {}

    def test_content_request_url(self):
        url, params = self.config.transform_video_content_request(
            video_id="pred-123", api_base=API_BASE, litellm_params=GenericLiteLLMParams(), headers={}
        )
        assert url == f"{API_BASE}/api/v3/predictions/pred-123/result"
        assert params == {}

    def test_content_download_raises_while_still_processing(self):
        raw = httpx.Response(200, json=prediction("processing"))
        with pytest.raises(WaveSpeedError, match="still processing"):
            self.config.transform_video_content_response(raw_response=raw, logging_obj=self.logging_obj)

    def test_content_download_raises_on_failed_prediction(self):
        raw = httpx.Response(200, json=prediction("failed", error="upstream rejected the prompt"))
        with pytest.raises(WaveSpeedError, match="upstream rejected the prompt"):
            self.config.transform_video_content_response(raw_response=raw, logging_obj=self.logging_obj)

    def test_validate_environment_sets_bearer_and_attribution_headers(self):
        headers = self.config.validate_environment({}, MODEL, api_key="sk-test")
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["X-Client-Name"] == "litellm"


class TestWaveSpeedVideoContentDownload:
    def setup_method(self):
        self.config = WaveSpeedVideoConfig()
        self.logging_obj = Mock()

    @respx.mock
    def test_content_response_downloads_the_output(self):
        download = respx.get(OUTPUT_URL).mock(return_value=httpx.Response(200, content=b"mp4-bytes"))

        content = self.config.transform_video_content_response(
            raw_response=httpx.Response(200, json=prediction("completed", outputs=[OUTPUT_URL])),
            logging_obj=self.logging_obj,
        )

        assert content == b"mp4-bytes"
        assert download.call_count == 1

    @respx.mock
    def test_content_response_raises_on_a_dead_output_url(self):
        respx.get(OUTPUT_URL).mock(return_value=httpx.Response(404))

        with pytest.raises(httpx.HTTPStatusError):
            self.config.transform_video_content_response(
                raw_response=httpx.Response(200, json=prediction("completed", outputs=[OUTPUT_URL])),
                logging_obj=self.logging_obj,
            )

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_content_response_downloads_the_output(self, monkeypatch):
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        download = respx.get(OUTPUT_URL).mock(return_value=httpx.Response(200, content=b"mp4-bytes"))

        content = await self.config.async_transform_video_content_response(
            raw_response=httpx.Response(200, json=prediction("completed", outputs=[OUTPUT_URL])),
            logging_obj=self.logging_obj,
        )

        assert content == b"mp4-bytes"
        assert download.call_count == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_content_response_raises_while_still_processing(self, monkeypatch):
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        with pytest.raises(WaveSpeedError, match="still created"):
            await self.config.async_transform_video_content_response(
                raw_response=httpx.Response(200, json=prediction("created")),
                logging_obj=self.logging_obj,
            )


class TestWaveSpeedVideoMisc:
    def setup_method(self):
        self.config = WaveSpeedVideoConfig()

    def test_get_complete_url_defaults_and_overrides(self):
        assert self.config.get_complete_url(MODEL, None, {}) == API_BASE
        assert self.config.get_complete_url(MODEL, "https://proxy.internal/", {}) == "https://proxy.internal"

    def test_status_retrieve_response_encodes_the_provider_into_the_id(self):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=httpx.Response(200, json=prediction("processing")),
            logging_obj=Mock(),
            custom_llm_provider="wavespeed",
        )

        assert video.status == "in_progress"
        assert extract_original_video_id(video.id) == "pred-123"

    def test_unknown_status_falls_back_to_queued(self):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=httpx.Response(200, json=prediction("something-new")),
            logging_obj=Mock(),
        )
        assert video.status == "queued"

    @pytest.mark.parametrize(
        "created_at, expected",
        [("2026-08-20T10:00:00Z", 1787220000), (None, 0), ("", 0), ("not-a-date", 0)],
    )
    def test_created_at_parsing(self, created_at, expected):
        payload = envelope({"id": "pred-123", "status": "created", "created_at": created_at})
        video = self.config.transform_video_status_retrieve_response(
            raw_response=httpx.Response(200, json=payload), logging_obj=Mock()
        )
        assert video.created_at == expected

    @pytest.mark.parametrize(
        "seconds, expected_duration",
        [("5", 5), (5, 5), (5.9, 5), ("5.9", 5), (None, None), ("abc", None), (True, None), (object(), None)],
    )
    def test_seconds_coercion(self, seconds, expected_duration):
        mapped = self.config.map_openai_params({"seconds": seconds}, MODEL, False)
        assert mapped.get("duration") == expected_duration

    def test_size_without_an_x_is_left_alone(self):
        assert "size" not in self.config.map_openai_params({"size": "720p"}, MODEL, False)

    def test_get_error_class(self):
        error = self.config.get_error_class("boom", 503, {})
        assert isinstance(error, WaveSpeedError)
        assert error.status_code == 503

    @pytest.mark.parametrize(
        "call",
        [
            lambda c: c.transform_video_remix_request("v", "p", API_BASE, GenericLiteLLMParams(), {}),
            lambda c: c.transform_video_remix_response(httpx.Response(200), Mock()),
            lambda c: c.transform_video_list_request(API_BASE, GenericLiteLLMParams(), {}),
            lambda c: c.transform_video_list_response(httpx.Response(200), Mock()),
            lambda c: c.transform_video_delete_request("v", API_BASE, GenericLiteLLMParams(), {}),
            lambda c: c.transform_video_delete_response(httpx.Response(200), Mock()),
            lambda c: c.transform_video_create_character_request("n", object(), API_BASE, GenericLiteLLMParams(), {}),
            lambda c: c.transform_video_create_character_response(httpx.Response(200), Mock()),
            lambda c: c.transform_video_get_character_request("c", API_BASE, GenericLiteLLMParams(), {}),
            lambda c: c.transform_video_get_character_response(httpx.Response(200), Mock()),
            lambda c: c.transform_video_edit_request("p", "v", API_BASE, GenericLiteLLMParams(), {}),
            lambda c: c.transform_video_edit_response(httpx.Response(200), Mock()),
            lambda c: c.transform_video_extension_request("p", "v", "5", API_BASE, GenericLiteLLMParams(), {}),
            lambda c: c.transform_video_extension_response(httpx.Response(200), Mock()),
        ],
    )
    def test_unsupported_surfaces_raise_not_implemented(self, call):
        with pytest.raises(NotImplementedError):
            call(self.config)
