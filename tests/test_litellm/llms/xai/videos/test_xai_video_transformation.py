"""
Tests for xAI (Grok Imagine) video generation transformation.
"""

from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.xai.videos.transformation import XAIVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider


class TestXAIVideoTransformation:
    """Test XAIVideoConfig transformation class."""

    def setup_method(self):
        self.config = XAIVideoConfig()
        self.mock_logging_obj = Mock()

    def test_get_supported_openai_params(self):
        params = self.config.get_supported_openai_params("grok-imagine-video")
        assert "model" in params
        assert "prompt" in params
        assert "input_reference" in params
        assert "seconds" in params
        assert "size" in params

    def test_get_complete_url_default(self):
        url = self.config.get_complete_url("grok-imagine-video", None, {})
        assert url == "https://api.x.ai/v1"

    def test_get_complete_url_custom(self):
        url = self.config.get_complete_url(
            "grok-imagine-video", "https://custom.api.com/v1/", {}
        )
        assert url == "https://custom.api.com/v1"

    def test_validate_environment(self):
        headers = self.config.validate_environment(
            {}, "grok-imagine-video", api_key="test-key-123"
        )
        assert headers["Authorization"] == "Bearer test-key-123"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_missing_key(self):
        with pytest.raises(ValueError, match="xAI API key is required"):
            self.config.validate_environment({}, "grok-imagine-video", api_key=None)

    def test_map_openai_params_seconds(self):
        mapped = self.config.map_openai_params(
            {"seconds": "10"}, "grok-imagine-video", False
        )
        assert mapped["duration"] == 10

    def test_map_openai_params_size_to_aspect_ratio(self):
        mapped = self.config.map_openai_params(
            {"size": "1280x720"}, "grok-imagine-video", False
        )
        assert mapped["aspect_ratio"] == "16:9"

    def test_map_openai_params_portrait_size(self):
        mapped = self.config.map_openai_params(
            {"size": "720x1280"}, "grok-imagine-video", False
        )
        assert mapped["aspect_ratio"] == "9:16"

    def test_map_openai_params_input_reference(self):
        mapped = self.config.map_openai_params(
            {"input_reference": "https://example.com/image.png"},
            "grok-imagine-video",
            False,
        )
        assert mapped["image"] == {"url": "https://example.com/image.png"}

    def test_map_openai_params_passthrough(self):
        """Provider-specific params not in supported list are passed through."""
        mapped = self.config.map_openai_params(
            {"reference_images": [{"url": "https://example.com/ref.png"}]},
            "grok-imagine-video",
            False,
        )
        assert mapped["reference_images"] == [{"url": "https://example.com/ref.png"}]

    def test_transform_video_create_request(self):
        data, files, url = self.config.transform_video_create_request(
            model="grok-imagine-video",
            prompt="A rocket launching from Mars",
            api_base="https://api.x.ai/v1",
            video_create_optional_request_params={
                "duration": 10,
                "aspect_ratio": "16:9",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["model"] == "grok-imagine-video"
        assert data["prompt"] == "A rocket launching from Mars"
        assert data["duration"] == 10
        assert data["aspect_ratio"] == "16:9"
        assert files == []
        assert url == "https://api.x.ai/v1/videos/generations"

    def test_transform_video_create_response(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {"request_id": "req-abc-123"}

        result = self.config.transform_video_create_response(
            model="grok-imagine-video",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="xai",
            request_data={"duration": 10, "aspect_ratio": "16:9"},
        )

        assert isinstance(result, VideoObject)
        assert result.status == "queued"
        assert result.model == "grok-imagine-video"
        assert result.seconds == "10"
        assert result.size == "1280x720"
        # ID should be encoded with provider info
        assert result.id.startswith("video_")

    def test_transform_video_status_retrieve_request(self):
        video_id = encode_video_id_with_provider(
            "req-abc-123", "xai", "grok-imagine-video"
        )
        url, params = self.config.transform_video_status_retrieve_request(
            video_id=video_id,
            api_base="https://api.x.ai/v1",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://api.x.ai/v1/videos/req-abc-123"
        assert params == {}

    def test_transform_video_status_retrieve_response_done(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "done",
            "progress": 100,
            "model": "grok-imagine-video",
            "video": {
                "url": "https://vidgen.x.ai/video.mp4",
                "duration": 8,
                "respect_moderation": True,
            },
            "usage": {"cost_in_usd_ticks": 400000000},
        }

        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="xai",
        )

        assert isinstance(result, VideoObject)
        assert result.status == "completed"
        assert result.model == "grok-imagine-video"
        assert result.seconds == "8"
        assert result.progress == 100
        assert result.usage == {"cost_in_usd_ticks": 400000000}

    def test_transform_video_status_retrieve_response_pending(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "pending",
            "progress": 42,
        }

        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
        )

        assert result.status == "in_progress"
        assert result.progress == 42

    def test_transform_video_status_retrieve_response_failed(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "failed",
            "progress": 0,
            "error": {
                "code": "internal_error",
                "message": "Generation failed due to internal error",
            },
        }

        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
        )

        assert result.status == "failed"
        assert result.error["code"] == "internal_error"

    def test_extract_video_url_pending(self):
        with pytest.raises(ValueError, match="still processing"):
            self.config._extract_video_url({"status": "pending", "video": {}})

    def test_extract_video_url_failed(self):
        with pytest.raises(ValueError, match="generation failed"):
            self.config._extract_video_url(
                {
                    "status": "failed",
                    "error": {"message": "Content policy violation"},
                }
            )

    def test_transform_video_content_request(self):
        video_id = encode_video_id_with_provider(
            "req-abc-123", "xai", "grok-imagine-video"
        )
        url, params = self.config.transform_video_content_request(
            video_id=video_id,
            api_base="https://api.x.ai/v1",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://api.x.ai/v1/videos/req-abc-123"

    def test_transform_video_edit_request(self):
        url, data = self.config.transform_video_edit_request(
            video_id="source-video-url",
            prompt="Make it brighter",
            api_base="https://api.x.ai/v1",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://api.x.ai/v1/videos/edits"
        assert data["prompt"] == "Make it brighter"
        assert data["video"]["url"] == "source-video-url"

    def test_transform_video_extension_request(self):
        url, data = self.config.transform_video_extension_request(
            video_id="source-video-url",
            prompt="Continue the scene",
            api_base="https://api.x.ai/v1",
            litellm_params=GenericLiteLLMParams(),
            headers={},
            seconds="6",
        )

        assert url == "https://api.x.ai/v1/videos/extensions"
        assert data["prompt"] == "Continue the scene"
        assert data["duration"] == 6

    def test_unsupported_operations_raise(self):
        with pytest.raises(NotImplementedError):
            self.config.transform_video_remix_request()
        with pytest.raises(NotImplementedError):
            self.config.transform_video_list_request()
        with pytest.raises(NotImplementedError):
            self.config.transform_video_delete_request()

    def test_full_workflow(self):
        """Test complete workflow: create -> status (pending) -> status (done)."""
        # Step 1: Create
        data, files, url = self.config.transform_video_create_request(
            model="grok-imagine-video",
            prompt="A sunset over the ocean",
            api_base="https://api.x.ai/v1",
            video_create_optional_request_params={"duration": 8},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://api.x.ai/v1/videos/generations"

        # Step 2: Parse create response
        mock_create = Mock(spec=httpx.Response)
        mock_create.json.return_value = {"request_id": "req-workflow-test"}

        video_obj = self.config.transform_video_create_response(
            model="grok-imagine-video",
            raw_response=mock_create,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="xai",
            request_data=data,
        )

        assert video_obj.status == "queued"
        assert video_obj.id.startswith("video_")

        # Step 3: Poll - still pending
        mock_pending = Mock(spec=httpx.Response)
        mock_pending.json.return_value = {"status": "pending", "progress": 50}

        pending_obj = self.config.transform_video_status_retrieve_response(
            raw_response=mock_pending,
            logging_obj=self.mock_logging_obj,
        )

        assert pending_obj.status == "in_progress"
        assert pending_obj.progress == 50

        # Step 4: Poll - done
        mock_done = Mock(spec=httpx.Response)
        mock_done.json.return_value = {
            "status": "done",
            "progress": 100,
            "model": "grok-imagine-video",
            "video": {"url": "https://vidgen.x.ai/video.mp4", "duration": 8},
        }

        done_obj = self.config.transform_video_status_retrieve_response(
            raw_response=mock_done,
            logging_obj=self.mock_logging_obj,
        )

        assert done_obj.status == "completed"
        assert done_obj.seconds == "8"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
