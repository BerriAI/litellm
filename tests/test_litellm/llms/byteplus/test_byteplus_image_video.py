"""
Unit tests for the BytePlus image-generation and video-generation handlers.

Transformation-layer tests (no network) plus env-gated live smoke tests.
"""

import os
import sys

import httpx
import pytest

workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

import litellm
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager


# ----------------------------- image -----------------------------
class TestBytePlusImageGeneration:
    def _cfg(self):
        return ProviderConfigManager.get_provider_image_generation_config(
            model="seedream-4-5-251128", provider=LlmProviders.BYTEPLUS
        )

    def test_config_resolves(self):
        from litellm.llms.byteplus.image_generation.transformation import (
            BytePlusImageGenerationConfig,
        )

        assert isinstance(self._cfg(), BytePlusImageGenerationConfig)

    def test_complete_url(self):
        url = self._cfg().get_complete_url(
            api_base=None,
            api_key="x",
            model="seedream-4-5-251128",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"

    def test_size_defaults_to_valid_min_area(self):
        # BytePlus rejects images under 3,686,400 px; default must be applied.
        mapped = self._cfg().map_openai_params(
            non_default_params={}, optional_params={}, model="m", drop_params=False
        )
        assert mapped["size"] == "2048x2048"

    def test_params_pass_through(self):
        mapped = self._cfg().map_openai_params(
            non_default_params={"size": "1024x1024", "response_format": "url", "n": 2},
            optional_params={},
            model="m",
            drop_params=False,
        )
        assert mapped == {"size": "1024x1024", "response_format": "url", "n": 2}

    def test_validate_environment_requires_key(self, monkeypatch):
        monkeypatch.delenv("BYTEPLUS_API_KEY", raising=False)
        with pytest.raises(ValueError):
            self._cfg().validate_environment(
                headers={},
                model="m",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )
        h = self._cfg().validate_environment(
            headers={}, model="m", messages=[], optional_params={}, litellm_params={},
            api_key="ark-xyz",
        )
        assert h["Authorization"] == "Bearer ark-xyz"

    def test_response_parsing(self):
        raw = httpx.Response(
            200,
            json={
                "model": "seedream-4-5-251128",
                "created": 1780576790,
                "data": [{"url": "https://example.com/img.jpeg"}],
                "usage": {"generated_images": 1},
            },
        )
        out = self._cfg().transform_image_generation_response(
            model="seedream-4-5-251128",
            raw_response=raw,
            model_response=ImageResponse(),
            logging_obj=None,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert out.data[0].url == "https://example.com/img.jpeg"

    def test_cost_calculator(self, monkeypatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")
        from litellm.llms.byteplus.image_generation.cost_calculator import (
            cost_calculator,
        )

        resp = ImageResponse()
        resp.data = [object()]  # one image
        cost = cost_calculator(
            model="byteplus/seedream-4-5-251128", image_response=resp
        )
        assert cost == 0.04


# ----------------------------- video -----------------------------
class TestBytePlusVideoGeneration:
    def _cfg(self):
        return ProviderConfigManager.get_provider_video_config(
            model="seedance-1-5-pro-251215", provider=LlmProviders.BYTEPLUS
        )

    def test_config_resolves(self):
        from litellm.llms.byteplus.videos.transformation import BytePlusVideoConfig

        assert isinstance(self._cfg(), BytePlusVideoConfig)

    def test_size_to_aspect_ratio(self):
        from litellm.types.videos.main import VideoCreateOptionalRequestParams

        mapped = self._cfg().map_openai_params(
            VideoCreateOptionalRequestParams(size="1280x720", seconds=5),
            model="seedance-1-5-pro-251215",
            drop_params=False,
        )
        assert mapped["ratio"] == "16:9"
        assert mapped["duration"] == 5

    def test_create_request_builds_content_array(self):
        from litellm.types.router import GenericLiteLLMParams

        data, files, url = self._cfg().transform_video_create_request(
            model="seedance-1-5-pro-251215",
            prompt="ocean wave",
            api_base="https://ark.ap-southeast.bytepluses.com/api/v3",
            video_create_optional_request_params={
                "ratio": "16:9",
                "duration": 5,
                "generate_audio": True,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url.endswith("/contents/generations/tasks")
        assert data["model"] == "seedance-1-5-pro-251215"
        assert data["content"] == [{"type": "text", "text": "ocean wave"}]
        assert data["ratio"] == "16:9" and data["generate_audio"] is True

    def test_create_request_with_image_reference(self):
        from litellm.types.router import GenericLiteLLMParams

        data, _, _ = self._cfg().transform_video_create_request(
            model="seedance-1-5-pro-251215",
            prompt="make it move",
            api_base="https://ark.ap-southeast.bytepluses.com/api/v3",
            video_create_optional_request_params={
                "input_reference": "https://example.com/a.jpg"
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["content"] == [
            {"type": "text", "text": "make it move"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
        ]

    def test_status_mapping_and_url_extraction(self):
        cfg = self._cfg()
        assert cfg._map_status("queued") == "queued"
        assert cfg._map_status("running") == "in_progress"
        assert cfg._map_status("succeeded") == "completed"
        assert cfg._map_status("failed") == "failed"

        succeeded = {
            "id": "cgt-123",
            "model": "seedance-1-5-pro-251215",
            "status": "succeeded",
            "content": {"video_url": "https://example.com/v.mp4"},
            "created_at": 1780577242,
            "duration": 5,
        }
        out = cfg.transform_video_status_retrieve_response(
            raw_response=httpx.Response(200, json=succeeded),
            logging_obj=None,
            custom_llm_provider="byteplus",
        )
        assert out.status == "completed"
        # The video bytes are fetched via video_content, which extracts the URL.
        assert (
            cfg._extract_video_url_from_response(succeeded)
            == "https://example.com/v.mp4"
        )
        # Still-processing / failed responses surface a clear error.
        with pytest.raises(ValueError):
            cfg._extract_video_url_from_response({"status": "running"})

    @pytest.mark.skipif(
        not os.environ.get("BYTEPLUS_API_KEY"), reason="BYTEPLUS_API_KEY not set"
    )
    def test_video_create_live(self):
        v = litellm.video_generation(
            model="byteplus/seedance-1-5-pro-251215",
            prompt="a calm ocean wave",
            seconds=5,
            size="1280x720",
        )
        assert v.id and v.status in ("queued", "in_progress", "completed")
