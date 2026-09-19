from unittest.mock import Mock

import httpx
import pytest

import litellm
import litellm.llms.fal_ai.videos.transformation as fal_video_module
from litellm.cost_calculator import default_video_cost_calculator
from litellm.llms.fal_ai.videos.transformation import (
    FalAIVideoConfig,
    FalAIVideoError,
    _queue_request_base_path,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.utils import decode_video_id_with_provider
from litellm.utils import ProviderConfigManager

MODEL = "bytedance/seedance-2.5/text-to-video"


class TestFalAIVideoTransformation:
    def setup_method(self):
        self.config = FalAIVideoConfig()
        self.logging_obj = Mock()

    def test_map_openai_params(self):
        mapped = self.config.map_openai_params(
            {
                "seconds": "5",
                "size": "1280x720",
                "input_reference": "https://example.com/image.png",
                "user": "user-123",
                "generate_audio": False,
            },
            MODEL,
            False,
        )

        assert mapped == {
            "duration": "5",
            "resolution": "720p",
            "aspect_ratio": "16:9",
            "image_url": "https://example.com/image.png",
            "end_user_id": "user-123",
            "generate_audio": False,
        }

        assert self.config.map_openai_params({"size": "1080x1080"}, MODEL, False) == {
            "resolution": "1080p",
            "aspect_ratio": "1:1",
        }
        assert self.config.map_openai_params({"size": "720p"}, MODEL, False) == {"resolution": "720p"}
        assert self.config.map_openai_params({"size": "720x1280"}, MODEL, False) == {
            "resolution": "720p",
            "aspect_ratio": "9:16",
        }
        assert self.config.map_openai_params({"size": "1080x1920"}, MODEL, False) == {
            "resolution": "1080p",
            "aspect_ratio": "9:16",
        }

    def test_map_openai_params_rejects_non_url_input_reference(self):
        with pytest.raises(ValueError, match="public image URL"):
            self.config.map_openai_params({"input_reference": b"image"}, MODEL, False)

    def test_transform_video_create_request(self):
        body, files, url = self.config.transform_video_create_request(
            model=MODEL,
            prompt="A quiet ocean at sunrise",
            api_base="https://queue.fal.run",
            video_create_optional_request_params={
                "duration": "5",
                "resolution": "480p",
                "aspect_ratio": "16:9",
                "generate_audio": False,
                "model": MODEL,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == f"https://queue.fal.run/{MODEL}"
        assert files == []
        assert body == {
            "prompt": "A quiet ocean at sunrise",
            "duration": "5",
            "resolution": "480p",
            "aspect_ratio": "16:9",
            "generate_audio": False,
        }
        assert "model" not in body

    def test_transform_video_create_response_encodes_model_and_usage(self):
        response = Mock(spec=httpx.Response)
        response.json.return_value = {"request_id": "abc"}

        video = self.config.transform_video_create_response(
            model=MODEL,
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
            request_data={"duration": "5", "resolution": "480p"},
        )

        decoded = decode_video_id_with_provider(video.id)
        assert decoded["custom_llm_provider"] == "fal_ai"
        assert decoded["model_id"] == MODEL
        assert decoded["video_id"] == "abc"
        assert video.status == "queued"
        assert video.usage == {"duration_seconds": 5.0, "video_resolution": "480p"}

        auto_video = self.config.transform_video_create_response(
            model=MODEL,
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
            request_data={"duration": "auto"},
        )
        assert auto_video.usage == {"video_resolution": "720p"}
        assert auto_video.seconds is None
        assert auto_video.size is None

    def test_status_request_uses_queue_base_path(self):
        response = Mock(spec=httpx.Response)
        response.json.return_value = {"request_id": "abc"}
        video = self.config.transform_video_create_response(
            model=MODEL,
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
            request_data={},
        )

        url, params = self.config.transform_video_status_retrieve_request(
            video_id=video.id,
            api_base="https://queue.fal.run",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == "https://queue.fal.run/bytedance/seedance-2.5/requests/abc/status"
        assert params == {}
        assert _queue_request_base_path("workflows/owner/app/x") == "workflows/owner/app"
        assert _queue_request_base_path("comfy/owner/app/x") == "comfy/owner/app"

    def test_status_request_rejects_unencoded_video_id(self):
        with pytest.raises(ValueError, match="must be created through litellm"):
            self.config.transform_video_status_retrieve_request(
                video_id="abc",
                api_base="https://queue.fal.run",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    @pytest.mark.parametrize(
        ("response_data", "expected_status"),
        [
            ({"request_id": "abc", "status": "IN_QUEUE"}, "queued"),
            ({"request_id": "abc", "status": "IN_PROGRESS"}, "in_progress"),
            ({"request_id": "abc", "status": "COMPLETED"}, "completed"),
        ],
    )
    def test_status_response_mapping(self, response_data, expected_status):
        response = Mock(spec=httpx.Response)
        response.json.return_value = response_data

        video = self.config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
        )

        assert video.status == expected_status
        assert video.created_at == 0

    def test_status_response_id_stays_pollable(self):
        response = Mock(spec=httpx.Response)
        response.json.return_value = {
            "request_id": "abc",
            "status": "IN_PROGRESS",
            "response_url": "https://queue.fal.run/bytedance/seedance-2.5/requests/abc",
        }

        video = self.config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
        )

        status_url, _ = self.config.transform_video_status_retrieve_request(
            video_id=video.id,
            api_base="https://queue.fal.run",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        content_url, _ = self.config.transform_video_content_request(
            video_id=video.id,
            api_base="https://queue.fal.run",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert status_url == "https://queue.fal.run/bytedance/seedance-2.5/requests/abc/status"
        assert content_url == "https://queue.fal.run/bytedance/seedance-2.5/requests/abc"

    def test_status_response_error(self):
        response = Mock(spec=httpx.Response)
        response.json.return_value = {
            "request_id": "abc",
            "status": "COMPLETED",
            "error": "generation failed",
        }

        video = self.config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
        )

        assert video.status == "failed"
        assert video.error == {"code": "fal_error", "message": "generation failed"}

    def test_content_response_downloads_video_url(self, monkeypatch):
        content_response = httpx.Response(
            200,
            content=b"video-bytes",
            request=httpx.Request("GET", "https://cdn.example.com/video.mp4"),
        )

        class FakeHTTPClient:
            def get(self, url):
                assert url == "https://cdn.example.com/video.mp4"
                return content_response

        monkeypatch.setattr(fal_video_module, "_get_httpx_client", lambda: FakeHTTPClient())
        response = Mock(spec=httpx.Response)
        response.json.return_value = {"video": {"url": "https://cdn.example.com/video.mp4"}}

        assert self.config.transform_video_content_response(response, self.logging_obj) == b"video-bytes"

    def test_content_response_rejects_missing_video(self):
        response = Mock(spec=httpx.Response)
        response.json.return_value = {"error": "generation failed"}

        with pytest.raises(ValueError, match="generation failed"):
            self.config.transform_video_content_response(response, self.logging_obj)

    def test_provider_config_and_error_class(self):
        provider_config = ProviderConfigManager.get_provider_video_config(
            model=MODEL,
            provider=LlmProviders.FAL_AI,
        )
        assert isinstance(provider_config, FalAIVideoConfig)
        assert isinstance(self.config.get_error_class("bad key", 401, {}), FalAIVideoError)

    def test_video_cost_uses_tiered_rows(self):
        rows = {
            model: row
            for model, row in litellm.model_cost.items()
            if row.get("litellm_provider") == "fal_ai" and row.get("mode") == "video_generation"
        }
        assert rows
        for model, row in rows.items():
            assert default_video_cost_calculator(model, 5, "fal_ai", video_resolution="480p") == (
                5 * row["output_cost_per_second_480p"]
            )
            assert default_video_cost_calculator(model, 5, "fal_ai", video_resolution="720p") == (
                5 * row["output_cost_per_second"]
            )
