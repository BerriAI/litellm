from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from litellm.llms.bytedance.videos.transformation import ByteDanceVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider


class TestByteDanceVideoTransformation:
    def setup_method(self):
        self.config = ByteDanceVideoConfig()
        self.mock_logging_obj = Mock()

    def test_validate_environment_sets_bearer_auth(self):
        headers: dict = {}
        result = self.config.validate_environment(
            headers=headers,
            model="dreamina-seedance-2-0-260128",
            api_key="ark-test-key-123",
        )

        assert result["Authorization"] == "Bearer ark-test-key-123"
        assert result["Content-Type"] == "application/json"

    def test_validate_environment_missing_key_raises(self):
        with pytest.raises(ValueError, match="ByteDance API key is required"):
            self.config.validate_environment(
                headers={},
                model="dreamina-seedance-2-0-260128",
                api_key=None,
            )

    def test_map_openai_params_size_to_ratio(self):
        result = self.config.map_openai_params(
            video_create_optional_params={"size": "1920x1080"},
            model="dreamina-seedance-2-0-260128",
            drop_params=False,
        )

        assert result["ratio"] == "1920:1080"

    def test_map_openai_params_colon_size_passthrough(self):
        result = self.config.map_openai_params(
            video_create_optional_params={"size": "16:9"},
            model="dreamina-seedance-2-0-260128",
            drop_params=False,
        )

        assert result["ratio"] == "16:9"

    def test_map_openai_params_seconds_to_duration(self):
        result = self.config.map_openai_params(
            video_create_optional_params={"seconds": "8"},
            model="dreamina-seedance-2-0-260128",
            drop_params=False,
        )

        assert result["duration"] == 8

    def test_map_openai_params_seconds_int(self):
        result = self.config.map_openai_params(
            video_create_optional_params={"seconds": 10},
            model="dreamina-seedance-2-0-260128",
            drop_params=False,
        )

        assert result["duration"] == 10

    def test_transform_video_create_request_text_only(self):
        data, files, url = self.config.transform_video_create_request(
            model="dreamina-seedance-2-0-260128",
            prompt="A cat walking on a sunny beach",
            api_base="https://ark.ap-southeast.bytepluses.com",
            video_create_optional_request_params={
                "ratio": "16:9",
                "duration": 5,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["model"] == "dreamina-seedance-2-0-260128"
        assert len(data["content"]) == 1
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"] == "A cat walking on a sunny beach"
        assert data["ratio"] == "16:9"
        assert data["duration"] == 5
        assert files == []
        assert url == "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"

    def test_transform_video_create_request_with_image_reference(self):
        data, files, url = self.config.transform_video_create_request(
            model="dreamina-seedance-2-0-260128",
            prompt="The person slowly turns their head",
            api_base="https://ark.ap-southeast.bytepluses.com",
            video_create_optional_request_params={
                "input_reference": "https://example.com/person.jpg",
                "ratio": "9:16",
                "duration": 8,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert len(data["content"]) == 2
        assert data["content"][0]["type"] == "text"
        assert data["content"][1]["type"] == "image_url"
        assert data["content"][1]["image_url"]["url"] == "https://example.com/person.jpg"
        assert data["content"][1]["role"] == "first_frame"
        assert "input_reference" not in data

    def test_transform_video_create_request_passes_extra_params(self):
        data, _, _ = self.config.transform_video_create_request(
            model="dreamina-seedance-2-0-260128",
            prompt="Test",
            api_base="https://ark.ap-southeast.bytepluses.com",
            video_create_optional_request_params={
                "resolution": "1080p",
                "generate_audio": True,
                "watermark": False,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["resolution"] == "1080p"
        assert data["generate_audio"] is True
        assert data["watermark"] is False

    def test_transform_video_create_response(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "task-abc-123",
        }

        result = self.config.transform_video_create_response(
            model="dreamina-seedance-2-0-260128",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="bytedance",
            request_data={
                "model": "dreamina-seedance-2-0-260128",
                "ratio": "16:9",
                "duration": 5,
                "resolution": "1080p",
            },
        )

        assert isinstance(result, VideoObject)
        assert result.status == "queued"
        assert result.id.startswith("video_")
        assert result.model == "dreamina-seedance-2-0-260128"
        assert result.size == "16x9"
        assert result.seconds == "5"
        assert result.usage["video_resolution"] == "1080p"

    def test_transform_video_status_retrieve_request_url(self):
        video_id = encode_video_id_with_provider("task-abc-123", "bytedance", "dreamina-seedance-2-0-260128")

        url, params = self.config.transform_video_status_retrieve_request(
            video_id=video_id,
            api_base="https://ark.ap-southeast.bytepluses.com",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks/task-abc-123"
        assert params == {}

    def test_transform_video_status_retrieve_response_succeeded(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "task-abc-123",
            "status": "succeeded",
            "model": "dreamina-seedance-2-0-260128",
            "content": {
                "video_url": "https://cdn.example.com/video.mp4",
                "last_frame_url": "https://cdn.example.com/frame.jpg",
            },
            "usage": {
                "completion_tokens": 12345,
                "total_tokens": 12345,
            },
            "duration": 5,
            "ratio": "16:9",
            "resolution": "720p",
            "created_at": 1719000000,
            "updated_at": 1719000060,
        }

        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="bytedance",
        )

        assert isinstance(result, VideoObject)
        assert result.status == "completed"
        assert result.seconds == "5"
        assert result.size == "16x9"
        assert result.model == "dreamina-seedance-2-0-260128"
        assert result.completed_at == 1719000060
        assert result.id.startswith("video_")
        assert result.usage["video_resolution"] == "720p"

    def test_transform_video_status_retrieve_response_running(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "task-abc-123",
            "status": "running",
            "created_at": 1719000000,
        }

        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
        )

        assert result.status == "in_progress"

    def test_transform_video_status_retrieve_response_failed(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "task-abc-123",
            "status": "failed",
            "error": {
                "code": "CONTENT_VIOLATION",
                "message": "Content policy violated",
            },
            "created_at": 1719000000,
            "updated_at": 1719000010,
        }

        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
        )

        assert result.status == "failed"
        assert result.error is not None
        assert result.error["code"] == "CONTENT_VIOLATION"
        assert result.completed_at == 1719000010

    def test_status_mapping_all_values(self):
        expected = {
            "queued": "queued",
            "running": "in_progress",
            "succeeded": "completed",
            "failed": "failed",
            "cancelled": "failed",
            "expired": "failed",
        }
        for seedance_status, openai_status in expected.items():
            assert self.config._map_seedance_status(seedance_status) == openai_status

    def test_status_mapping_unknown_defaults_to_queued(self):
        assert self.config._map_seedance_status("unknown_status") == "queued"

    def test_extract_video_url_succeeded(self):
        response_data = {
            "status": "succeeded",
            "content": {
                "video_url": "https://cdn.example.com/video.mp4",
            },
        }
        url = self.config._extract_video_url_from_response(response_data)
        assert url == "https://cdn.example.com/video.mp4"

    def test_extract_video_url_still_processing_raises(self):
        response_data = {"status": "running", "content": None}
        with pytest.raises(ValueError, match="still processing"):
            self.config._extract_video_url_from_response(response_data)

    def test_extract_video_url_failed_raises(self):
        response_data = {
            "status": "failed",
            "error": {"message": "Generation failed"},
        }
        with pytest.raises(ValueError, match="Generation failed"):
            self.config._extract_video_url_from_response(response_data)

    def test_extract_video_url_no_content_raises(self):
        response_data = {"status": "succeeded", "content": {}}
        with pytest.raises(ValueError, match="not found"):
            self.config._extract_video_url_from_response(response_data)

    def test_transform_video_delete_request(self):
        video_id = encode_video_id_with_provider("task-abc-123", "bytedance", "dreamina-seedance-2-0-260128")

        url, data = self.config.transform_video_delete_request(
            video_id=video_id,
            api_base="https://ark.ap-southeast.bytepluses.com",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks/task-abc-123"
        assert data == {}

    def test_transform_video_status_encodes_video_id_path_segment(self):
        url, _ = self.config.transform_video_status_retrieve_request(
            video_id="../../tasks/other?x=1#frag",
            api_base="https://ark.ap-southeast.bytepluses.com",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "../../" not in url
        assert "%2F" in url or "%2f" in url

    def test_full_video_workflow(self):
        data, files, url = self.config.transform_video_create_request(
            model="dreamina-seedance-2-0-260128",
            prompt="A peaceful sunset over the ocean",
            api_base="https://ark.ap-southeast.bytepluses.com",
            video_create_optional_request_params={
                "ratio": "16:9",
                "duration": 5,
                "resolution": "720p",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["model"] == "dreamina-seedance-2-0-260128"
        assert url.endswith("/contents/generations/tasks")

        mock_create_response = Mock(spec=httpx.Response)
        mock_create_response.json.return_value = {"id": "task-workflow-123"}

        video_obj = self.config.transform_video_create_response(
            model="dreamina-seedance-2-0-260128",
            raw_response=mock_create_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="bytedance",
            request_data=data,
        )

        assert video_obj.status == "queued"
        assert video_obj.id.startswith("video_")

        mock_status_response = Mock(spec=httpx.Response)
        mock_status_response.json.return_value = {
            "id": "task-workflow-123",
            "status": "succeeded",
            "model": "dreamina-seedance-2-0-260128",
            "content": {
                "video_url": "https://cdn.example.com/output.mp4",
            },
            "duration": 5,
            "ratio": "16:9",
            "usage": {"completion_tokens": 5000, "total_tokens": 5000},
            "created_at": 1719000000,
            "updated_at": 1719000060,
        }

        status_obj = self.config.transform_video_status_retrieve_response(
            raw_response=mock_status_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="bytedance",
        )

        assert status_obj.status == "completed"
        assert status_obj.seconds == "5"
        assert status_obj.completed_at == 1719000060

        url = self.config._extract_video_url_from_response(mock_status_response.json())
        assert url == "https://cdn.example.com/output.mp4"

    def test_get_complete_url_default(self):
        url = self.config.get_complete_url(
            model="dreamina-seedance-2-0-260128",
            api_base=None,
            litellm_params={},
        )
        assert url == "https://ark.ap-southeast.bytepluses.com"

    def test_get_complete_url_custom(self):
        url = self.config.get_complete_url(
            model="dreamina-seedance-2-0-260128",
            api_base="https://custom.api.com/",
            litellm_params={},
        )
        assert url == "https://custom.api.com"

    def test_get_supported_openai_params_includes_new_params(self):
        supported = self.config.get_supported_openai_params("dreamina-seedance-2-0-260128")
        for param in (
            "last_frame",
            "reference_images",
            "resolution",
            "seed",
            "generate_audio",
            "watermark",
            "return_last_frame",
        ):
            assert param in supported, f"{param} missing from supported params"

    def test_map_openai_params_resolution(self):
        result = self.config.map_openai_params(
            video_create_optional_params={"resolution": "1080p"},
            model="dreamina-seedance-2-0-260128",
            drop_params=False,
        )
        assert result["resolution"] == "1080p"

    def test_map_openai_params_seedance_specific_params(self):
        result = self.config.map_openai_params(
            video_create_optional_params={
                "seed": 42,
                "generate_audio": False,
                "watermark": False,
                "return_last_frame": True,
            },
            model="dreamina-seedance-2-0-260128",
            drop_params=False,
        )
        assert result["seed"] == 42
        assert result["generate_audio"] is False
        assert result["watermark"] is False
        assert result["return_last_frame"] is True

    def test_transform_video_create_request_with_last_frame(self):
        data, _, _ = self.config.transform_video_create_request(
            model="dreamina-seedance-2-0-260128",
            prompt="Smooth transition from start to end",
            api_base="https://ark.ap-southeast.bytepluses.com",
            video_create_optional_request_params={
                "input_reference": "https://example.com/first.jpg",
                "last_frame": "https://example.com/last.jpg",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert len(data["content"]) == 3
        assert data["content"][0]["type"] == "text"
        assert data["content"][1]["role"] == "first_frame"
        assert data["content"][1]["image_url"]["url"] == "https://example.com/first.jpg"
        assert data["content"][2]["role"] == "last_frame"
        assert data["content"][2]["image_url"]["url"] == "https://example.com/last.jpg"
        assert "last_frame" not in data

    def test_transform_video_create_request_with_reference_images(self):
        ref_urls = [
            "https://example.com/ref1.jpg",
            "https://example.com/ref2.jpg",
            "https://example.com/ref3.jpg",
        ]
        data, _, _ = self.config.transform_video_create_request(
            model="dreamina-seedance-2-0-260128",
            prompt="Match the style of these references",
            api_base="https://ark.ap-southeast.bytepluses.com",
            video_create_optional_request_params={
                "reference_images": ref_urls,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        ref_items = [c for c in data["content"] if c.get("role") == "reference_image"]
        assert len(ref_items) == 3
        assert ref_items[0]["image_url"]["url"] == "https://example.com/ref1.jpg"
        assert ref_items[1]["image_url"]["url"] == "https://example.com/ref2.jpg"
        assert ref_items[2]["image_url"]["url"] == "https://example.com/ref3.jpg"
        assert "reference_images" not in data

    def test_transform_video_create_request_rejects_frame_and_reference_mixed(self):
        with pytest.raises(ValueError, match="does not allow first_frame/last_frame and reference_images"):
            self.config.transform_video_create_request(
                model="dreamina-seedance-2-0-260128",
                prompt="Should fail",
                api_base="https://ark.ap-southeast.bytepluses.com",
                video_create_optional_request_params={
                    "input_reference": "https://example.com/first.jpg",
                    "reference_images": ["https://example.com/ref1.jpg"],
                },
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_transform_video_create_request_frame_mode_with_config(self):
        data, _, _ = self.config.transform_video_create_request(
            model="dreamina-seedance-2-0-260128",
            prompt="Frame mode generation",
            api_base="https://ark.ap-southeast.bytepluses.com",
            video_create_optional_request_params={
                "input_reference": "https://example.com/first.jpg",
                "last_frame": "https://example.com/last.jpg",
                "ratio": "16:9",
                "duration": 5,
                "resolution": "1080p",
                "seed": 42,
                "generate_audio": False,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["content"][0]["type"] == "text"
        assert data["content"][1]["role"] == "first_frame"
        assert data["content"][2]["role"] == "last_frame"
        assert len(data["content"]) == 3

        assert data["ratio"] == "16:9"
        assert data["duration"] == 5
        assert data["resolution"] == "1080p"
        assert data["seed"] == 42
        assert data["generate_audio"] is False

        for key in ("input_reference", "last_frame", "reference_images"):
            assert key not in data

    def test_transform_video_create_request_resolution_in_request_data(self):
        data, _, _ = self.config.transform_video_create_request(
            model="dreamina-seedance-2-0-260128",
            prompt="Test resolution",
            api_base="https://ark.ap-southeast.bytepluses.com",
            video_create_optional_request_params={"resolution": "720p"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["resolution"] == "720p"

    def test_transform_video_create_response_default_duration_when_no_seconds(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {"id": "task-no-dur-123"}

        result = self.config.transform_video_create_response(
            model="dreamina-seedance-2-0-260128",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="bytedance",
            request_data={"model": "dreamina-seedance-2-0-260128", "ratio": "16:9"},
        )

        assert result.usage["duration_seconds"] == 4.0

    def test_transform_video_create_response_uses_request_duration(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {"id": "task-dur-123"}

        result = self.config.transform_video_create_response(
            model="dreamina-seedance-2-0-260128",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="bytedance",
            request_data={"model": "dreamina-seedance-2-0-260128", "duration": 8},
        )

        assert result.usage["duration_seconds"] == 8.0

    def test_transform_video_status_retrieve_response_default_duration_when_no_duration(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "task-abc-123",
            "status": "succeeded",
            "created_at": 1719000000,
            "updated_at": 1719000060,
        }

        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
        )

        assert result.usage["duration_seconds"] == 4.0

    def test_transform_video_content_response_downloads_video(self):
        mock_poll_response = Mock(spec=httpx.Response)
        mock_poll_response.json.return_value = {
            "status": "succeeded",
            "content": {"video_url": "https://cdn.example.com/video.mp4"},
        }

        mock_video_response = Mock()
        mock_video_response.content = b"fake-video-bytes"
        mock_video_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_video_response

        with patch(
            "litellm.llms.bytedance.videos.transformation.get_httpx_client",
            return_value=mock_client,
        ):
            result = self.config.transform_video_content_response(
                raw_response=mock_poll_response,
                logging_obj=self.mock_logging_obj,
            )

        assert result == b"fake-video-bytes"
        mock_client.get.assert_called_once_with("https://cdn.example.com/video.mp4")

    @pytest.mark.asyncio
    async def test_async_transform_video_content_response_downloads_video(self):
        mock_poll_response = Mock(spec=httpx.Response)
        mock_poll_response.json.return_value = {
            "status": "succeeded",
            "content": {"video_url": "https://cdn.example.com/video.mp4"},
        }

        mock_video_response = Mock()
        mock_video_response.content = b"fake-async-video-bytes"
        mock_video_response.raise_for_status = Mock()

        mock_async_client = AsyncMock()
        mock_async_client.get.return_value = mock_video_response

        with patch(
            "litellm.llms.bytedance.videos.transformation.get_async_httpx_client",
            return_value=mock_async_client,
        ):
            result = await self.config.async_transform_video_content_response(
                raw_response=mock_poll_response,
                logging_obj=self.mock_logging_obj,
            )

        assert result == b"fake-async-video-bytes"
        mock_async_client.get.assert_called_once_with("https://cdn.example.com/video.mp4")

    def test_transform_video_delete_response(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "task-abc-123",
            "created_at": 1719000000,
        }

        result = self.config.transform_video_delete_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
        )

        assert isinstance(result, VideoObject)
        assert result.id == "task-abc-123"
        assert result.status == "failed"

    def test_negative_duration_clamped_to_zero_in_create_response(self):
        """seconds=-1 (auto duration) must not produce negative cost."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {"id": "task-neg-dur"}

        result = self.config.transform_video_create_response(
            model="dreamina-seedance-2-0-260128",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="bytedance",
            request_data={"model": "dreamina-seedance-2-0-260128", "duration": -1},
        )

        assert result.usage["duration_seconds"] == 0.0

    def test_negative_duration_clamped_to_zero_in_status_response(self):
        """Negative duration from provider must not produce negative cost."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "task-neg-dur",
            "status": "succeeded",
            "duration": -1,
            "created_at": 1719000000,
            "updated_at": 1719000060,
        }

        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
        )

        assert result.usage["duration_seconds"] == 0.0

    def test_unsupported_operations_raise(self):
        with pytest.raises(NotImplementedError):
            self.config.transform_video_remix_request("vid", "prompt", "base", GenericLiteLLMParams(), {})
        with pytest.raises(NotImplementedError):
            self.config.transform_video_list_request("base", GenericLiteLLMParams(), {})
