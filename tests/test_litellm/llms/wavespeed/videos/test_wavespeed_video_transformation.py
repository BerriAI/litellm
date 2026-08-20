"""Tests for WaveSpeed AI video generation transformation."""

from unittest.mock import Mock

import httpx
import pytest

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

    def test_unsupported_surfaces_raise_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.config.transform_video_list_request(API_BASE, GenericLiteLLMParams(), {})
        with pytest.raises(NotImplementedError):
            self.config.transform_video_delete_request("pred-123", API_BASE, GenericLiteLLMParams(), {})
        with pytest.raises(NotImplementedError):
            self.config.transform_video_remix_request("pred-123", "x", API_BASE, GenericLiteLLMParams(), {})
