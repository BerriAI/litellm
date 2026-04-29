"""
Tests for LTX Video generation transformation.
"""

import asyncio
import os
from unittest.mock import Mock, patch

import httpx
import pytest

import litellm
import litellm.llms.ltx.videos.transformation as ltx_video_transformation
from litellm.cost_calculator import completion_cost
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.ltx.videos.transformation import LTXVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)


class TestLTXVideoTransformation:
    """Test LTXVideoConfig transformation class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = LTXVideoConfig()
        self.mock_logging_obj = Mock()

    def test_get_supported_openai_params(self):
        """Test supported OpenAI parameters list."""
        params = self.config.get_supported_openai_params("ltx-2-3-fast")
        assert "model" in params
        assert "prompt" in params
        assert "input_reference" in params
        assert "seconds" in params
        assert "size" in params
        assert "user" not in params
        assert "extra_headers" in params

    def test_map_openai_params_basic(self):
        """Test parameter mapping from OpenAI format to LTX format."""
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "input_reference": "https://example.com/image.jpg",
                "seconds": "5",
                "size": "1920x1080",
            },
            model="ltx-2-3-fast",
            drop_params=False,
        )

        assert mapped["image_uri"] == "https://example.com/image.jpg"
        assert mapped["duration"] == 5
        assert mapped["resolution"] == "1920x1080"

    def test_map_openai_params_passthrough(self):
        """Test that LTX-specific params are passed through."""
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "fps": 30,
                "generate_audio": False,
                "camera_motion": "dolly_in",
            },
            model="ltx-2-3-fast",
            drop_params=False,
        )

        assert mapped["fps"] == 30
        assert mapped["generate_audio"] is False
        assert mapped["camera_motion"] == "dolly_in"

    def test_map_openai_params_seconds_int(self):
        """Test seconds conversion when provided as int."""
        mapped = self.config.map_openai_params(
            video_create_optional_params={"seconds": 10},
            model="ltx-2-3-fast",
            drop_params=False,
        )
        assert mapped["duration"] == 10

    def test_map_openai_params_user_raises_when_not_dropping(self):
        """Test unsupported OpenAI user param fails loudly for LTX."""
        with pytest.raises(ValueError, match="Parameter user is not supported"):
            self.config.map_openai_params(
                video_create_optional_params={"user": "end-user-123"},
                model="ltx-2-3-fast",
                drop_params=False,
            )

    def test_map_openai_params_user_is_dropped_when_requested(self):
        """Test unsupported user param can be dropped explicitly."""
        mapped = self.config.map_openai_params(
            video_create_optional_params={"user": "end-user-123"},
            model="ltx-2-3-fast",
            drop_params=True,
        )

        assert mapped == {}

    def test_validate_environment(self):
        """Test authentication header setup."""
        headers = self.config.validate_environment(
            headers={},
            model="ltx-2-3-fast",
            api_key="test-api-key",
        )

        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_missing_key(self, monkeypatch):
        """Test that missing API key raises ValueError."""
        monkeypatch.delenv("LTX_API_KEY", raising=False)
        monkeypatch.setattr(litellm, "api_key", None)
        with pytest.raises(ValueError, match="LTX API key is required"):
            self.config.validate_environment(
                headers={},
                model="ltx-2-3-fast",
                api_key=None,
            )

    def test_validate_environment_empty_model_still_requires_key(self, monkeypatch):
        """Test that content retrieval no longer relies on a model='' sentinel."""
        monkeypatch.delenv("LTX_API_KEY", raising=False)
        monkeypatch.setattr(litellm, "api_key", None)
        with pytest.raises(ValueError, match="LTX API key is required"):
            self.config.validate_environment(
                headers={},
                model="",
                api_key=None,
            )

    def test_get_complete_url_default(self):
        """Test default API base URL."""
        url = self.config.get_complete_url(
            model="ltx-2-3-fast",
            api_base=None,
            litellm_params={},
        )
        assert url == "https://api.ltx.video/v1"

    def test_get_complete_url_custom(self):
        """Test custom API base URL."""
        url = self.config.get_complete_url(
            model="ltx-2-3-fast",
            api_base="https://custom.api.com/v1/",
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1"

    def test_transform_video_create_request_text_to_video(self):
        """Test text-to-video request transformation."""
        data, files, url = self.config.transform_video_create_request(
            model="ltx-2-3-fast",
            prompt="A serene mountain landscape at sunset",
            api_base="https://api.ltx.video/v1",
            video_create_optional_request_params={
                "duration": 5,
                "resolution": "1920x1080",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["model"] == "ltx-2-3-fast"
        assert data["prompt"] == "A serene mountain landscape at sunset"
        assert data["duration"] == 5
        assert data["resolution"] == "1920x1080"
        assert "image_uri" not in data
        assert files == []
        assert url == "https://api.ltx.video/v1/text-to-video"

    def test_transform_video_create_request_image_to_video(self):
        """Test image-to-video request transformation when image_uri is present."""
        data, files, url = self.config.transform_video_create_request(
            model="ltx-2-3-pro",
            prompt="Animate this image with gentle motion",
            api_base="https://api.ltx.video/v1",
            video_create_optional_request_params={
                "image_uri": "https://example.com/source.jpg",
                "duration": 3,
                "resolution": "1280x720",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["model"] == "ltx-2-3-pro"
        assert data["prompt"] == "Animate this image with gentle motion"
        assert data["image_uri"] == "https://example.com/source.jpg"
        assert files == []
        assert url == "https://api.ltx.video/v1/image-to-video"

    def test_transform_video_create_request_with_optional_params(self):
        """Test request with LTX-specific optional parameters."""
        data, files, url = self.config.transform_video_create_request(
            model="ltx-2-3-pro",
            prompt="A cinematic pan across a cityscape",
            api_base="https://api.ltx.video/v1",
            video_create_optional_request_params={
                "duration": 8,
                "resolution": "1920x1080",
                "fps": 30,
                "generate_audio": True,
                "camera_motion": "dolly_in",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["fps"] == 30
        assert data["generate_audio"] is True
        assert data["camera_motion"] == "dolly_in"
        assert url == "https://api.ltx.video/v1/text-to-video"

    def test_transform_video_create_response_binary(self, monkeypatch, tmp_path):
        """Test that binary response produces a completed VideoObject."""
        monkeypatch.setattr(ltx_video_transformation, "LTX_VIDEO_STORAGE_DIR", tmp_path)
        mock_response = Mock(spec=httpx.Response)
        mock_response.content = b"\x00\x00\x00\x1cftypisom"  # fake video bytes
        mock_response.status_code = 200
        mock_response.request = httpx.Request("POST", "https://api.ltx.video/v1")

        request_data = {
            "model": "ltx-2-3-fast",
            "prompt": "test",
            "duration": 5,
            "resolution": "1920x1080",
        }

        result = self.config.transform_video_create_response(
            model="ltx-2-3-fast",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="ltx",
            request_data=request_data,
        )

        stored_video_path = tmp_path / f"{extract_original_video_id(result.id)}.mp4"

        assert isinstance(result, VideoObject)
        assert result.status == "completed"
        assert result.created_at is not None
        assert result.created_at > 0
        assert result.completed_at is not None
        assert result.model == "ltx-2-3-fast"
        assert result.size == "1920x1080"
        assert result.seconds == "5"
        assert result.id.startswith("video_")
        assert result.usage is not None
        assert result.usage["duration_seconds"] == 5.0
        assert stored_video_path.read_bytes() == mock_response.content
        assert result._hidden_params["video_content_path"] == str(stored_video_path)

    def test_transform_video_create_response_usage_drives_cost_calculation(
        self, monkeypatch, tmp_path
    ):
        """Test LTX usage.duration_seconds is consumed by completion_cost()."""
        monkeypatch.setattr(ltx_video_transformation, "LTX_VIDEO_STORAGE_DIR", tmp_path)
        mock_response = Mock(spec=httpx.Response)
        mock_response.content = b"\x00\x00\x00\x1cftypisom"
        mock_response.status_code = 200
        mock_response.request = httpx.Request("POST", "https://api.ltx.video/v1")

        result = self.config.transform_video_create_response(
            model="ltx/ltx-2-3-fast",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="ltx",
            request_data={
                "model": "ltx/ltx-2-3-fast",
                "duration": 5,
            },
        )

        with patch(
            "litellm.llms.openai.cost_calculation.video_generation_cost",
            return_value=0.3,
        ) as mock_video_generation_cost:
            cost = completion_cost(
                completion_response=result,
                model="ltx/ltx-2-3-fast",
                custom_llm_provider="ltx",
                call_type="create_video",
            )

        assert cost == 0.3
        assert mock_video_generation_cost.call_args.kwargs["duration_seconds"] == 5.0

    def test_transform_video_create_response_without_request_data(
        self, monkeypatch, tmp_path
    ):
        """Test response transformation without request data."""
        monkeypatch.setattr(ltx_video_transformation, "LTX_VIDEO_STORAGE_DIR", tmp_path)
        mock_response = Mock(spec=httpx.Response)
        mock_response.content = b"\x00\x00\x00"
        mock_response.status_code = 200
        mock_response.request = httpx.Request("POST", "https://api.ltx.video/v1")

        result = self.config.transform_video_create_response(
            model="ltx-2-3-fast",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider=None,
            request_data=None,
        )

        assert isinstance(result, VideoObject)
        assert result.status == "completed"
        assert result.id  # should have a UUID

    def test_transform_video_create_response_cleans_up_expired_files(
        self, monkeypatch, tmp_path
    ):
        """Test stale locally persisted LTX videos are pruned on write."""
        monkeypatch.setattr(ltx_video_transformation, "LTX_VIDEO_STORAGE_DIR", tmp_path)
        monkeypatch.setattr(
            ltx_video_transformation, "LTX_VIDEO_STORAGE_MAX_AGE_SECONDS", 1
        )

        stale_video_path = tmp_path / "stale-video.mp4"
        stale_video_path.parent.mkdir(parents=True, exist_ok=True)
        stale_video_path.write_bytes(b"old-bytes")
        old_timestamp = ltx_video_transformation.time.time() - 10
        os.utime(stale_video_path, (old_timestamp, old_timestamp))

        mock_response = Mock(spec=httpx.Response)
        mock_response.content = b"new-video-bytes"
        mock_response.status_code = 200
        mock_response.request = httpx.Request("POST", "https://api.ltx.video/v1")

        result = self.config.transform_video_create_response(
            model="ltx-2-3-fast",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider=None,
            request_data={"model": "ltx-2-3-fast"},
        )

        assert stale_video_path.exists() is False
        assert (tmp_path / f"{result.id}.mp4").read_bytes() == b"new-video-bytes"

    def test_transform_video_create_response_empty_binary_raises(self):
        """Test that empty create responses fail loudly."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.content = b""
        mock_response.status_code = 200
        mock_response.request = httpx.Request("POST", "https://api.ltx.video/v1")

        with pytest.raises(BaseLLMException, match="empty video response body"):
            self.config.transform_video_create_response(
                model="ltx-2-3-fast",
                raw_response=mock_response,
                logging_obj=self.mock_logging_obj,
                custom_llm_provider="ltx",
                request_data={"model": "ltx-2-3-fast"},
            )

    def test_transform_video_content_request_uses_local_file(
        self, monkeypatch, tmp_path
    ):
        """Test content requests resolve to the locally persisted file."""
        monkeypatch.setattr(ltx_video_transformation, "LTX_VIDEO_STORAGE_DIR", tmp_path)
        original_video_id = "ltx-local-video"
        stored_video_path = tmp_path / f"{original_video_id}.mp4"
        stored_video_path.write_bytes(b"fake-video-binary-data")

        url, data = self.config.transform_video_content_request(
            video_id=encode_video_id_with_provider(
                original_video_id, "ltx", "ltx-2-3-fast"
            ),
            api_base="https://api.ltx.video/v1",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == stored_video_path.resolve().as_uri()
        assert data == {}

    def test_transform_video_content_request_rejects_path_traversal_video_id(
        self, monkeypatch, tmp_path
    ):
        """Test content requests cannot traverse outside LTX video storage."""
        storage_dir = tmp_path / "litellm_ltx_videos"
        monkeypatch.setattr(
            ltx_video_transformation, "LTX_VIDEO_STORAGE_DIR", storage_dir
        )
        sibling_video_path = tmp_path / "other_dir" / "secret"
        sibling_video_path.parent.mkdir(parents=True)
        sibling_video_path.with_suffix(".mp4").write_bytes(b"private-video")

        with pytest.raises(BaseLLMException, match="must not contain path separators"):
            self.config.transform_video_content_request(
                video_id=encode_video_id_with_provider(
                    "../other_dir/secret", "ltx", "ltx-2-3-fast"
                ),
                api_base="https://api.ltx.video/v1",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_video_content_handler_reads_local_file(self, monkeypatch, tmp_path):
        """Test the shared video content handler can serve local LTX artifacts."""
        monkeypatch.setattr(ltx_video_transformation, "LTX_VIDEO_STORAGE_DIR", tmp_path)
        original_video_id = "ltx-local-video"
        expected_bytes = b"fake-video-binary-data"
        stored_video_path = tmp_path / f"{original_video_id}.mp4"
        stored_video_path.write_bytes(expected_bytes)

        result = BaseLLMHTTPHandler().video_content_handler(
            video_id=encode_video_id_with_provider(
                original_video_id, "ltx", "ltx-2-3-fast"
            ),
            video_content_provider_config=self.config,
            custom_llm_provider="ltx",
            litellm_params=GenericLiteLLMParams(),
            logging_obj=self.mock_logging_obj,
            timeout=30,
        )

        assert result == expected_bytes

    def test_async_video_content_handler_reads_local_file(self, monkeypatch, tmp_path):
        """Test the async shared content handler can serve local LTX artifacts."""
        monkeypatch.setattr(ltx_video_transformation, "LTX_VIDEO_STORAGE_DIR", tmp_path)
        original_video_id = "ltx-local-video"
        expected_bytes = b"fake-video-binary-data"
        stored_video_path = tmp_path / f"{original_video_id}.mp4"
        stored_video_path.write_bytes(expected_bytes)

        result = asyncio.run(
            BaseLLMHTTPHandler().async_video_content_handler(
                video_id=encode_video_id_with_provider(
                    original_video_id, "ltx", "ltx-2-3-fast"
                ),
                video_content_provider_config=self.config,
                custom_llm_provider="ltx",
                litellm_params=GenericLiteLLMParams(),
                logging_obj=self.mock_logging_obj,
                timeout=30,
                client=Mock(spec=AsyncHTTPHandler),
            )
        )

        assert result == expected_bytes

    def test_unsupported_operations(self, monkeypatch, tmp_path):
        """Test that unsupported operations raise NotImplementedError."""
        monkeypatch.setattr(ltx_video_transformation, "LTX_VIDEO_STORAGE_DIR", tmp_path)
        stored_video_path = tmp_path / "ltx-local-video.mp4"
        stored_video_path.write_bytes(b"fake-video-binary-data")

        with pytest.raises(NotImplementedError, match="content variants"):
            self.config.transform_video_content_request(
                video_id=encode_video_id_with_provider(
                    "ltx-local-video", "ltx", "ltx-2-3-fast"
                ),
                api_base="",
                litellm_params=GenericLiteLLMParams(),
                headers={},
                variant="thumbnail",
            )

        with pytest.raises(BaseLLMException, match="No locally stored LTX video"):
            self.config.transform_video_content_request(
                video_id="missing-video",
                api_base="",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

        with pytest.raises(
            BaseLLMException, match="different instance or after a process restart"
        ):
            self.config.transform_video_content_request(
                video_id="missing-video",
                api_base="",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

        with pytest.raises(NotImplementedError):
            self.config.transform_video_status_retrieve_request(
                video_id="test",
                api_base="",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

        with pytest.raises(NotImplementedError):
            self.config.transform_video_remix_request(
                video_id="test",
                prompt="test",
                api_base="",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

        with pytest.raises(NotImplementedError):
            self.config.transform_video_delete_request(
                video_id="test",
                api_base="",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

        with pytest.raises(NotImplementedError):
            self.config.transform_video_list_request(
                api_base="", litellm_params=GenericLiteLLMParams(), headers={}
            )

    def test_full_text_to_video_workflow(self, monkeypatch, tmp_path):
        """Test complete text-to-video workflow."""
        monkeypatch.setattr(ltx_video_transformation, "LTX_VIDEO_STORAGE_DIR", tmp_path)
        config = LTXVideoConfig()
        mock_logging_obj = Mock()

        # Step 1: Map params
        mapped = config.map_openai_params(
            video_create_optional_params={
                "seconds": "5",
                "size": "1920x1080",
            },
            model="ltx-2-3-fast",
            drop_params=False,
        )

        assert mapped["duration"] == 5
        assert mapped["resolution"] == "1920x1080"

        # Step 2: Create request
        data, files, url = config.transform_video_create_request(
            model="ltx-2-3-fast",
            prompt="A serene mountain landscape",
            api_base="https://api.ltx.video/v1",
            video_create_optional_request_params=mapped,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://api.ltx.video/v1/text-to-video"
        assert data["prompt"] == "A serene mountain landscape"
        assert data["model"] == "ltx-2-3-fast"
        assert data["duration"] == 5

        # Step 3: Parse response (binary)
        mock_response = Mock(spec=httpx.Response)
        mock_response.content = b"fake-video-binary-data"
        mock_response.status_code = 200
        mock_response.request = httpx.Request("POST", "https://api.ltx.video/v1")

        video_obj = config.transform_video_create_response(
            model="ltx-2-3-fast",
            raw_response=mock_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="ltx",
            request_data=data,
        )

        assert video_obj.status == "completed"
        assert video_obj.model == "ltx-2-3-fast"
        assert video_obj.seconds == "5"

    def test_full_image_to_video_workflow(self, monkeypatch, tmp_path):
        """Test complete image-to-video workflow."""
        monkeypatch.setattr(ltx_video_transformation, "LTX_VIDEO_STORAGE_DIR", tmp_path)
        config = LTXVideoConfig()
        mock_logging_obj = Mock()

        # Step 1: Map params
        mapped = config.map_openai_params(
            video_create_optional_params={
                "input_reference": "https://example.com/image.jpg",
                "seconds": "3",
                "size": "1280x720",
            },
            model="ltx-2-3-pro",
            drop_params=False,
        )

        assert mapped["image_uri"] == "https://example.com/image.jpg"

        # Step 2: Create request
        data, files, url = config.transform_video_create_request(
            model="ltx-2-3-pro",
            prompt="Animate the scene with flowing water",
            api_base="https://api.ltx.video/v1",
            video_create_optional_request_params=mapped,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://api.ltx.video/v1/image-to-video"
        assert data["image_uri"] == "https://example.com/image.jpg"

        # Step 3: Parse response
        mock_response = Mock(spec=httpx.Response)
        mock_response.content = b"fake-video-binary-data"
        mock_response.status_code = 200
        mock_response.request = httpx.Request("POST", "https://api.ltx.video/v1")

        video_obj = config.transform_video_create_response(
            model="ltx-2-3-pro",
            raw_response=mock_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="ltx",
            request_data=data,
        )

        assert video_obj.status == "completed"
        assert video_obj.size == "1280x720"

    def test_ltx_models_are_registered_globally(self):
        """Test LTX models are exposed through LiteLLM's global model registries."""
        assert hasattr(litellm, "ltx_models")
        assert "ltx" in litellm.models_by_provider
        assert litellm.models_by_provider["ltx"] is litellm.ltx_models


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
