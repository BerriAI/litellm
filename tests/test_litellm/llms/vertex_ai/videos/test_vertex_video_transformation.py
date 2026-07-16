"""
Tests for Vertex AI (Veo) video generation transformation.
"""

import base64
import json
from pathlib import Path
from typing import Mapping, cast
from unittest.mock import Mock, patch

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai.cost_calculation import video_generation_cost
from litellm.llms.vertex_ai.videos.transformation import (
    VertexAIVideoConfig,
    _convert_image_to_vertex_format,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject

VEO_31_LITE_VERTEX_MODEL = "vertex_ai/veo-3.1-lite-generate-001"
ROOT_MODEL_COST_PATH = (
    Path(__file__).parents[5] / "model_prices_and_context_window.json"
)
BACKUP_MODEL_COST_PATH = (
    Path(__file__).parents[5]
    / "litellm"
    / "model_prices_and_context_window_backup.json"
)
ModelCostMap = Mapping[str, Mapping[str, object]]


def _load_model_cost_map(path: Path) -> ModelCostMap:
    return cast(ModelCostMap, json.loads(path.read_text()))


class TestVertexAIVideoConfig:
    """Test VertexAIVideoConfig transformation class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = VertexAIVideoConfig()
        self.mock_logging_obj = Mock()

    def test_get_supported_openai_params(self):
        """Test that correct params are supported."""
        params = self.config.get_supported_openai_params("veo-002")

        assert "model" in params
        assert "prompt" in params
        assert "input_reference" in params
        assert "seconds" in params
        assert "size" in params

    @patch.object(VertexAIVideoConfig, "get_access_token")
    def test_validate_environment(self, mock_get_access_token):
        """Test environment validation for Vertex AI."""
        # Mock the authentication
        mock_get_access_token.return_value = ("mock-access-token", "test-project")

        headers = {}
        litellm_params = {"vertex_project": "test-project"}

        result = self.config.validate_environment(
            headers=headers,
            model="veo-002",
            api_key=None,
            litellm_params=litellm_params,
        )

        # Should add Authorization header
        assert "Authorization" in result
        assert result["Authorization"] == "Bearer mock-access-token"
        assert "Content-Type" in result

    def test_get_complete_url(self):
        """Test URL construction for Vertex AI video generation."""
        litellm_params = {
            "vertex_project": "test-project",
            "vertex_location": "us-central1",
        }

        url = self.config.get_complete_url(
            model="vertex_ai/veo-002", api_base=None, litellm_params=litellm_params
        )

        expected = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/veo-002"
        assert url == expected
        # Should NOT include endpoint - that's added by transform methods
        assert not url.endswith(":predictLongRunning")

    def test_get_complete_url_with_custom_api_base(self):
        """Test URL construction with custom API base."""
        litellm_params = {
            "vertex_project": "test-project",
            "vertex_location": "us-west1",
        }

        url = self.config.get_complete_url(
            model="veo-002",
            api_base="https://custom-endpoint.example.com",
            litellm_params=litellm_params,
        )

        assert url.startswith("https://custom-endpoint.example.com")
        assert "test-project" in url
        assert "us-west1" in url
        assert "veo-002" in url
        # Should NOT include endpoint
        assert not url.endswith(":predictLongRunning")

    def test_get_complete_url_missing_project(self):
        """Test that missing vertex_project raises error."""
        litellm_params = {}

        # Note: The method might not raise if vertex_project can be fetched from env
        # This test verifies the behavior when completely missing
        try:
            url = self.config.get_complete_url(
                model="veo-002", api_base=None, litellm_params=litellm_params
            )
            # If no error is raised, vertex_project was obtained from environment
            # In that case, just verify a URL was returned
            assert url is not None
        except ValueError as e:
            # Expected behavior when vertex_project is truly missing
            assert "vertex_project is required" in str(e)

    def test_get_complete_url_default_location(self):
        """Test URL construction with default location."""
        litellm_params = {"vertex_project": "test-project"}

        url = self.config.get_complete_url(
            model="veo-002", api_base=None, litellm_params=litellm_params
        )

        # Should default to us-central1
        assert "us-central1" in url
        # Should NOT include endpoint
        assert not url.endswith(":predictLongRunning")

    def test_veo_31_lite_model_cost_entries_match_pricing(self):
        for path in (ROOT_MODEL_COST_PATH, BACKUP_MODEL_COST_PATH):
            model_cost = _load_model_cost_map(path)
            info = model_cost.get(VEO_31_LITE_VERTEX_MODEL)

            assert info is not None, f"{VEO_31_LITE_VERTEX_MODEL} missing from {path}"
            assert info["litellm_provider"] == "vertex_ai-video-models"
            assert info["mode"] == "video_generation"
            assert info["max_input_tokens"] == 1024
            assert info["output_cost_per_second"] == 0.05
            assert info["output_cost_per_second_1080p"] == 0.08
            assert info["supported_modalities"] == ["text", "image"]

    def test_veo_31_lite_provider_routing_from_local_model_map(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        model_cost = _load_model_cost_map(BACKUP_MODEL_COST_PATH)
        vertex_video_models = {
            model_name.removeprefix("vertex_ai/")
            for model_name, info in model_cost.items()
            if info.get("litellm_provider") == "vertex_ai-video-models"
        }
        monkeypatch.setattr(litellm, "vertex_ai_video_models", vertex_video_models)

        model, custom_llm_provider, _, _ = get_llm_provider(
            model="veo-3.1-lite-generate-001"
        )

        assert model == "veo-3.1-lite-generate-001"
        assert custom_llm_provider == "vertex_ai"

    def test_veo_31_lite_cost_uses_resolution_tiers(self):
        model_cost = _load_model_cost_map(BACKUP_MODEL_COST_PATH)
        model_info = model_cost[VEO_31_LITE_VERTEX_MODEL]

        assert video_generation_cost(
            model=VEO_31_LITE_VERTEX_MODEL,
            duration_seconds=10.0,
            custom_llm_provider="vertex_ai",
            model_info=dict(model_info),
            video_resolution="720p",
        ) == pytest.approx(0.5)
        assert video_generation_cost(
            model=VEO_31_LITE_VERTEX_MODEL,
            duration_seconds=10.0,
            custom_llm_provider="vertex_ai",
            model_info=dict(model_info),
            video_resolution="1080p",
        ) == pytest.approx(0.8)

    def test_transform_video_create_request(self):
        """Test transformation of video creation request."""
        prompt = "A cat playing with a ball of yarn"
        api_base = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/veo-002"

        data, files, url = self.config.transform_video_create_request(
            model="veo-002",
            prompt=prompt,
            api_base=api_base,
            video_create_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        # Check Vertex AI format
        assert "instances" in data
        assert len(data["instances"]) == 1
        assert data["instances"][0]["prompt"] == prompt

        # Parameters should not be present when empty
        assert "parameters" not in data or data["parameters"] == {}

        # Check URL has :predictLongRunning appended
        assert url.endswith(":predictLongRunning")
        assert api_base in url

        # Check no files are uploaded
        assert files == []

    def test_transform_video_create_request_with_parameters(self):
        """Test video creation request with aspect ratio and duration."""
        prompt = "A dog running in a park"
        api_base = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/veo-002"

        data, files, url = self.config.transform_video_create_request(
            model="veo-002",
            prompt=prompt,
            api_base=api_base,
            video_create_optional_request_params={
                "aspectRatio": "16:9",
                "durationSeconds": 8,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["instances"][0]["prompt"] == prompt
        assert data["parameters"]["aspectRatio"] == "16:9"
        assert data["parameters"]["durationSeconds"] == 8
        assert url.endswith(":predictLongRunning")

    def test_transform_video_create_request_with_image(self):
        """Test video creation request with image input."""
        prompt = "Extend this image with animation"
        api_base = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/veo-002"

        # Create a mock image file
        mock_image = Mock()
        mock_image.read.return_value = b"fake_image_data"
        mock_image.seek = Mock()

        with patch(
            "litellm.llms.vertex_ai.videos.transformation.ImageEditRequestUtils.get_image_content_type",
            return_value="image/jpeg",
        ):
            data, files, url = self.config.transform_video_create_request(
                model="veo-002",
                prompt=prompt,
                api_base=api_base,
                video_create_optional_request_params={"image": mock_image},
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

        # Check image was converted to base64
        assert "image" in data["instances"][0]
        assert "bytesBase64Encoded" in data["instances"][0]["image"]
        assert "mimeType" in data["instances"][0]["image"]
        assert data["instances"][0]["image"]["mimeType"] == "image/jpeg"
        assert url.endswith(":predictLongRunning")

    def test_map_openai_params(self):
        """Test parameter mapping from OpenAI to Vertex AI format."""
        openai_params = {"seconds": "8", "size": "1280x720"}

        mapped = self.config.map_openai_params(
            video_create_optional_params=openai_params,
            model="veo-002",
            drop_params=False,
        )

        assert mapped["durationSeconds"] == 8
        assert mapped["aspectRatio"] == "16:9"
        assert "resolution" not in mapped

    @pytest.mark.parametrize(
        ("size", "expected_resolution"),
        (("1280x720", "720p"), ("1920x1080", "1080p")),
    )
    def test_map_openai_size_to_resolution_for_veo_3(
        self, size: str, expected_resolution: str
    ):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"size": size},
            model=VEO_31_LITE_VERTEX_MODEL,
            drop_params=False,
        )

        assert mapped["aspectRatio"] == "16:9"
        assert mapped["resolution"] == expected_resolution

    def test_map_openai_size_does_not_infer_resolution_for_veo_2(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"size": "1920x1080"},
            model="vertex_ai/veo-2.0-generate-001",
            drop_params=False,
        )

        assert mapped["aspectRatio"] == "16:9"
        assert "resolution" not in mapped

    def test_map_openai_size_does_not_override_provider_resolution(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "size": "1920x1080",
                "parameters": {"resolution": "720p"},
            },
            model=VEO_31_LITE_VERTEX_MODEL,
            drop_params=False,
        )

        assert mapped["aspectRatio"] == "16:9"
        assert "resolution" not in mapped
        assert mapped["parameters"] == {"resolution": "720p"}

    def test_map_openai_size_does_not_override_direct_resolution(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "size": "1920x1080",
                "resolution": "720p",
            },
            model=VEO_31_LITE_VERTEX_MODEL,
            drop_params=False,
        )

        assert mapped["aspectRatio"] == "16:9"
        assert mapped["resolution"] == "720p"

    def test_map_openai_params_default_duration(self):
        """Test that durationSeconds is omitted when not provided."""
        openai_params = {"size": "1280x720"}

        mapped = self.config.map_openai_params(
            video_create_optional_params=openai_params,
            model="veo-002",
            drop_params=False,
        )

        assert mapped["aspectRatio"] == "16:9"
        assert "durationSeconds" not in mapped

    def test_map_openai_params_size_conversions(self):
        """Test size to aspect ratio conversions."""
        test_cases = [
            ("1280x720", "16:9"),
            ("1920x1080", "16:9"),
            ("720x1280", "9:16"),
            ("1080x1920", "9:16"),
            ("unknown", "16:9"),  # Default
        ]

        for size, expected_ratio in test_cases:
            mapped = self.config.map_openai_params(
                video_create_optional_params={"size": size},
                model="veo-002",
                drop_params=False,
            )
            assert mapped["aspectRatio"] == expected_ratio

    def test_transform_video_create_response(self):
        """Test transformation of video creation response."""
        # Mock response with operation name
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "name": "projects/test-project/locations/us-central1/publishers/google/models/veo-002/operations/12345",
            "metadata": {"createTime": "2024-01-15T10:30:00.000Z"},
        }

        video_obj = self.config.transform_video_create_response(
            model="vertex_ai/veo-002",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="vertex_ai",
        )

        assert isinstance(video_obj, VideoObject)
        assert video_obj.status == "processing"
        assert video_obj.object == "video"
        # Video ID is encoded with provider info, so just check it's not empty
        assert video_obj.id
        assert len(video_obj.id) > 0

    def test_transform_video_create_response_missing_operation_name(self):
        """Test that missing operation name raises error."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {}

        with pytest.raises(ValueError, match="No operation name in Veo response"):
            self.config.transform_video_create_response(
                model="veo-002",
                raw_response=mock_response,
                logging_obj=self.mock_logging_obj,
            )

    def test_transform_video_status_retrieve_request(self):
        """Test transformation of video status retrieve request."""
        operation_name = "projects/test-project/locations/us-central1/publishers/google/models/veo-002/operations/12345"

        # Provide an api_base that would be returned from get_complete_url
        api_base = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/veo-002"

        url, params = self.config.transform_video_status_retrieve_request(
            video_id=operation_name,
            api_base=api_base,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        # Check URL contains fetchPredictOperation endpoint
        assert "fetchPredictOperation" in url
        assert "test-project" in url
        assert "us-central1" in url
        assert "veo-002" in url

        # Check params contain operation name
        assert params["operationName"] == operation_name

    def test_transform_video_status_retrieve_request_invalid_format(self):
        """Test that invalid operation name format raises error."""
        invalid_operation_name = "invalid/operation/name"

        with pytest.raises(ValueError, match="Invalid operation name format"):
            self.config.transform_video_status_retrieve_request(
                video_id=invalid_operation_name,
                api_base=None,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_transform_video_status_retrieve_response_processing(self):
        """Test transformation of status response while processing."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "name": "projects/test-project/locations/us-central1/publishers/google/models/veo-002/operations/12345",
            "done": False,
            "metadata": {"createTime": "2024-01-15T10:30:00.000Z"},
        }

        video_obj = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="vertex_ai",
        )

        assert isinstance(video_obj, VideoObject)
        assert video_obj.status == "processing"

    def test_transform_video_status_retrieve_response_completed(self):
        """Test transformation of status response when completed."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "name": "projects/test-project/locations/us-central1/publishers/google/models/veo-002/operations/12345",
            "done": True,
            "metadata": {"createTime": "2024-01-15T10:30:00.000Z"},
            "response": {
                "@type": "type.googleapis.com/cloud.ai.large_models.vision.GenerateVideoResponse",
                "raiMediaFilteredCount": 0,
                "videos": [
                    {
                        "bytesBase64Encoded": base64.b64encode(
                            b"fake_video_data"
                        ).decode(),
                        "mimeType": "video/mp4",
                    }
                ],
            },
        }

        video_obj = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="vertex_ai",
        )

        assert isinstance(video_obj, VideoObject)
        assert video_obj.status == "completed"

    def test_transform_video_status_retrieve_response_error(self):
        """Test transformation of status response when an error is returned."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "name": "projects/test-project/locations/us-central1/publishers/google/models/veo-002/operations/12345",
            "done": True,
            "metadata": {"createTime": "2024-01-15T10:30:00.000Z"},
            "error": {
                "code": 3,
                "message": "Unsupported output video duration 3 seconds, supported durations are [8,5,6,7] for feature text_to_video.",
            },
        }

        video_obj = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="vertex_ai",
        )

        assert isinstance(video_obj, VideoObject)
        assert video_obj.status == "failed"
        assert video_obj.error == mock_response.json.return_value["error"]

    def test_transform_video_content_request(self):
        """Test transformation of video content request."""
        operation_name = "projects/test-project/locations/us-central1/publishers/google/models/veo-002/operations/12345"
        api_base = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/veo-002"

        url, params = self.config.transform_video_content_request(
            video_id=operation_name,
            api_base=api_base,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        # Should use same fetchPredictOperation endpoint
        assert "fetchPredictOperation" in url
        assert params["operationName"] == operation_name

    def test_transform_video_content_response(self):
        """Test transformation of video content response."""
        fake_video_bytes = b"fake_video_data_12345"
        encoded_video = base64.b64encode(fake_video_bytes).decode()

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "name": "projects/test-project/locations/us-central1/publishers/google/models/veo-002/operations/12345",
            "done": True,
            "response": {
                "@type": "type.googleapis.com/cloud.ai.large_models.vision.GenerateVideoResponse",
                "videos": [
                    {"bytesBase64Encoded": encoded_video, "mimeType": "video/mp4"}
                ],
            },
        }

        video_bytes = self.config.transform_video_content_response(
            raw_response=mock_response, logging_obj=self.mock_logging_obj
        )

        assert isinstance(video_bytes, bytes)
        assert video_bytes == fake_video_bytes

    def test_transform_video_content_response_not_complete(self):
        """Test that incomplete video raises error."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "name": "projects/test-project/locations/us-central1/publishers/google/models/veo-002/operations/12345",
            "done": False,
        }

        with pytest.raises(ValueError, match="Video generation is not complete yet"):
            self.config.transform_video_content_response(
                raw_response=mock_response, logging_obj=self.mock_logging_obj
            )

    def test_transform_video_content_response_missing_video_data(self):
        """Test that missing video data raises error."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "name": "projects/test-project/locations/us-central1/publishers/google/models/veo-002/operations/12345",
            "done": True,
            "response": {"videos": []},
        }

        with pytest.raises(ValueError, match="No video data found"):
            self.config.transform_video_content_response(
                raw_response=mock_response, logging_obj=self.mock_logging_obj
            )

    def test_get_video_edit_prefetch_params(self):
        """Test that prefetch params returns the fetchPredictOperation URL and body."""
        operation_name = "projects/test-project/locations/us-central1/publishers/google/models/veo-3.1-generate-001/operations/op-123"
        api_base = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models"

        fetch_url, fetch_body = self.config.get_video_edit_prefetch_params(
            video_id=operation_name,
            api_base=api_base,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "fetchPredictOperation" in fetch_url
        assert "veo-3.1-generate-001" in fetch_url
        assert fetch_body == {"operationName": operation_name}

    def test_transform_video_edit_request_with_bytes(self):
        """Test video edit request builds predictLongRunning body from pre-fetched bytes."""
        operation_name = "projects/test-project/locations/us-central1/publishers/google/models/veo-3.1-generate-001/operations/op-123"
        api_base = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models"
        fake_bytes = base64.b64encode(b"fake_video").decode()

        prefetched = {
            "done": True,
            "response": {
                "videos": [{"bytesBase64Encoded": fake_bytes, "mimeType": "video/mp4"}]
            },
        }

        url, data = self.config.transform_video_edit_request(
            prompt="Make it brighter",
            video_id=operation_name,
            api_base=api_base,
            litellm_params=GenericLiteLLMParams(),
            headers={"Authorization": "Bearer token"},
            prefetched_source_data=prefetched,
        )

        assert url.endswith(":predictLongRunning")
        assert "veo-3.1-generate-001" in url
        instance = data["instances"][0]
        assert instance["prompt"] == "Make it brighter"
        assert instance["video"]["bytesBase64Encoded"] == fake_bytes
        assert instance["video"]["mimeType"] == "video/mp4"

    def test_transform_video_edit_request_with_gcs_uri(self):
        """Test that gcsUri is used when present in source video."""
        operation_name = "projects/test-project/locations/us-central1/publishers/google/models/veo-3.1-generate-001/operations/op-456"
        api_base = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models"

        prefetched = {
            "done": True,
            "response": {
                "videos": [{"gcsUri": "gs://bucket/video.mp4", "mimeType": "video/mp4"}]
            },
        }

        _, data = self.config.transform_video_edit_request(
            prompt="Make it darker",
            video_id=operation_name,
            api_base=api_base,
            litellm_params=GenericLiteLLMParams(),
            headers={},
            prefetched_source_data=prefetched,
        )

        assert data["instances"][0]["video"] == {"gcsUri": "gs://bucket/video.mp4"}

    def test_transform_video_edit_request_source_not_done_raises(self):
        """Test that editing an in-progress video raises a clear error."""
        operation_name = "projects/test-project/locations/us-central1/publishers/google/models/veo-3.1-generate-001/operations/op-789"
        api_base = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models"

        with pytest.raises(ValueError, match="not complete yet"):
            self.config.transform_video_edit_request(
                prompt="Make it brighter",
                video_id=operation_name,
                api_base=api_base,
                litellm_params=GenericLiteLLMParams(),
                headers={},
                prefetched_source_data={"done": False},
            )

    def test_transform_video_edit_response(self):
        """Test that edit response returns a processing VideoObject with encoded ID."""
        operation_name = "projects/test-project/locations/us-central1/publishers/google/models/veo-3.1-generate-001/operations/new-op-123"
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {"name": operation_name}

        video_obj = self.config.transform_video_edit_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="vertex_ai",
        )

        assert isinstance(video_obj, VideoObject)
        assert video_obj.status == "processing"
        assert video_obj.id
        assert video_obj.model == "veo-3.1-generate-001"

    def test_transform_video_edit_response_includes_usage_for_cost(self):
        """Edit responses include duration/resolution usage for spend accounting."""
        operation_name = "projects/test-project/locations/us-central1/publishers/google/models/veo-3.1-generate-001/operations/new-op-123"
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {"name": operation_name}
        request_data = {
            "instances": [{"prompt": "Make it brighter", "video": {}}],
            "parameters": {"durationSeconds": 8, "resolution": "1080p"},
        }

        video_obj = self.config.transform_video_edit_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="vertex_ai",
            request_data=request_data,
        )

        assert video_obj.usage is not None
        assert video_obj.usage["duration_seconds"] == 8.0
        assert video_obj.usage["video_resolution"] == "1080p"

    def test_transform_video_remix_request_not_supported(self):
        """Test that video remix raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Video remix is not supported"):
            self.config.transform_video_remix_request(
                video_id="test-video-id",
                prompt="new prompt",
                api_base="https://example.com",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_transform_video_list_request_not_supported(self):
        """Test that video list raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Video list is not supported"):
            self.config.transform_video_list_request(
                api_base="https://example.com",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_transform_video_delete_request_not_supported(self):
        """Test that video delete raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Video delete is not supported"):
            self.config.transform_video_delete_request(
                video_id="test-video-id",
                api_base="https://example.com",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_get_error_class(self):
        """Test error class generation."""
        error = self.config.get_error_class(
            error_message="Test error", status_code=500, headers={}
        )

        # Should return VertexAIError
        from litellm.llms.vertex_ai.common_utils import VertexAIError

        assert isinstance(error, VertexAIError)
        assert error.status_code == 500
        assert "Test error" in str(error)


class TestConvertImageToVertexFormat:
    """Test the _convert_image_to_vertex_format helper function."""

    def test_convert_image_to_vertex_format(self):
        """Test image conversion to Vertex AI format."""
        fake_image_data = b"fake_jpeg_image_data"
        mock_image = Mock()
        mock_image.read.return_value = fake_image_data
        mock_image.seek = Mock()

        with patch(
            "litellm.llms.vertex_ai.videos.transformation.ImageEditRequestUtils.get_image_content_type",
            return_value="image/jpeg",
        ):
            result = _convert_image_to_vertex_format(mock_image)

        assert "bytesBase64Encoded" in result
        assert "mimeType" in result
        assert result["mimeType"] == "image/jpeg"

        # Verify base64 encoding
        decoded = base64.b64decode(result["bytesBase64Encoded"])
        assert decoded == fake_image_data

    def test_convert_image_to_vertex_format_with_seek(self):
        """Test image conversion with seek support."""
        fake_image_data = b"fake_png_image_data"
        mock_image = Mock()
        mock_image.read.return_value = fake_image_data
        mock_image.seek = Mock()

        with patch(
            "litellm.llms.vertex_ai.videos.transformation.ImageEditRequestUtils.get_image_content_type",
            return_value="image/png",
        ):
            result = _convert_image_to_vertex_format(mock_image)

        # Verify seek was called
        mock_image.seek.assert_called_once_with(0)

        assert result["mimeType"] == "image/png"
        decoded = base64.b64decode(result["bytesBase64Encoded"])
        assert decoded == fake_image_data


class TestImageAndParametersPassthrough:
    """
    Tests that image (gcsUri / bare gs:// / file-like) and a pre-built
    parameters dict are correctly forwarded through map_openai_params and
    transform_video_create_request.
    """

    def setup_method(self):
        self.config = VertexAIVideoConfig()
        self.api_base = (
            "https://us-central1-aiplatform.googleapis.com/v1/projects/"
            "test-project/locations/us-central1/publishers/google/models/veo-002"
        )

    # ------------------------------------------------------------------ #
    # map_openai_params                                                    #
    # ------------------------------------------------------------------ #

    def test_map_openai_params_passes_image_dict(self):
        """image dict (gcsUri format) is forwarded as-is."""
        image = {"gcsUri": "gs://my-bucket/boardwalk.jpg"}
        mapped = self.config.map_openai_params(
            video_create_optional_params={"image": image},
            model="veo-002",
            drop_params=False,
        )
        assert mapped["image"] == image

    def test_map_openai_params_passes_parameters_dict(self):
        """A pre-built parameters dict is forwarded as-is."""
        params = {"sampleCount": 1, "videoLengthSeconds": 5, "aspectRatio": "16:9"}
        mapped = self.config.map_openai_params(
            video_create_optional_params={"parameters": params},
            model="veo-002",
            drop_params=False,
        )
        assert mapped["parameters"] == params

    def test_map_openai_params_input_reference_takes_priority_over_image(self):
        """input_reference wins over a directly passed image key."""
        mock_file = Mock()
        image_dict = {"gcsUri": "gs://my-bucket/other.jpg"}
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "input_reference": mock_file,
                "image": image_dict,
            },
            model="veo-002",
            drop_params=False,
        )
        assert mapped["image"] is mock_file

    # ------------------------------------------------------------------ #
    # transform_video_create_request – image forms                        #
    # ------------------------------------------------------------------ #

    def test_transform_request_image_gcs_uri_dict(self):
        """image passed as {"gcsUri": "gs://..."} is placed in instances as-is."""
        image = {"gcsUri": "gs://my-bucket/boardwalk.jpg"}
        data, _, url = self.config.transform_video_create_request(
            model="veo-002",
            prompt="Cinematic drone shot",
            api_base=self.api_base,
            video_create_optional_request_params={"image": image},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["instances"][0]["image"] == image
        assert url.endswith(":predictLongRunning")

    def test_transform_request_image_bare_gs_uri_string(self):
        """A bare gs:// string is wrapped in {"gcsUri": ...} without downloading."""
        gs_uri = "gs://my-bucket/boardwalk.jpg"
        data, _, _ = self.config.transform_video_create_request(
            model="veo-002",
            prompt="Cinematic drone shot",
            api_base=self.api_base,
            video_create_optional_request_params={"image": gs_uri},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["instances"][0]["image"] == {"gcsUri": gs_uri}

    def test_transform_request_image_bytes_base64_dict(self):
        """image already in bytesBase64Encoded format is passed through unchanged."""
        image = {"bytesBase64Encoded": "abc123", "mimeType": "image/jpeg"}
        data, _, _ = self.config.transform_video_create_request(
            model="veo-002",
            prompt="Cinematic drone shot",
            api_base=self.api_base,
            video_create_optional_request_params={"image": image},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["instances"][0]["image"] == image

    # ------------------------------------------------------------------ #
    # transform_video_create_request – parameters dict                    #
    # ------------------------------------------------------------------ #

    def test_transform_request_parameters_dict_not_double_nested(self):
        """A pre-built parameters dict becomes request_data["parameters"] directly."""
        params = {"sampleCount": 1, "videoLengthSeconds": 5, "aspectRatio": "16:9"}
        data, _, _ = self.config.transform_video_create_request(
            model="veo-002",
            prompt="Cinematic drone shot",
            api_base=self.api_base,
            video_create_optional_request_params={"parameters": params},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["parameters"] == params
        # Must NOT be double-nested
        assert "parameters" not in data["parameters"]

    # ------------------------------------------------------------------ #
    # Full user scenario                                                   #
    # ------------------------------------------------------------------ #

    def test_transform_request_full_user_scenario(self):
        """
        Reproduces the exact user request:
            image: {"gcsUri": "gs://your-bucket-name/path/to/boardwalk.jpg"}
            parameters: {"sampleCount": 1, "videoLengthSeconds": 5,
                         "aspectRatio": "16:9", "storageUri": "gs://test/outputs/"}
        """
        image = {"gcsUri": "gs://your-bucket-name/path/to/boardwalk.jpg"}
        parameters = {
            "sampleCount": 1,
            "videoLengthSeconds": 5,
            "aspectRatio": "16:9",
            "storageUri": "gs://test/outputs/",
        }

        # Simulate the full pipeline: map_openai_params → transform_video_create_request
        mapped = self.config.map_openai_params(
            video_create_optional_params={"image": image, "parameters": parameters},
            model="veo-3.1-generate-preview",
            drop_params=False,
        )

        data, _, url = self.config.transform_video_create_request(
            model="veo-3.1-generate-preview",
            prompt="Cinematic drone shot moving forward along the beach boardwalk",
            api_base=self.api_base,
            video_create_optional_request_params=mapped,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        # instances contains prompt + image
        assert len(data["instances"]) == 1
        instance = data["instances"][0]
        assert (
            instance["prompt"]
            == "Cinematic drone shot moving forward along the beach boardwalk"
        )
        assert instance["image"] == image

        # parameters block is correct and not double-nested
        assert data["parameters"] == parameters
        assert "parameters" not in data["parameters"]

        assert url.endswith(":predictLongRunning")
