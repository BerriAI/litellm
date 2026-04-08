"""
Tests for Volcengine video generation transformation.
"""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from litellm.llms.volcengine.videos.transformation import VolcEngineVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)
from litellm.utils import ProviderConfigManager


class TestVolcEngineVideoTransformation:
    """Test Volcengine video request / response transformations."""

    def setup_method(self):
        self.config = VolcEngineVideoConfig()
        self.mock_logging_obj = Mock()

    def test_map_openai_params_maps_duration_and_ratio(self):
        mapped_params = self.config.map_openai_params(
            video_create_optional_params={
                "seconds": "11",
                "size": "1280x720",
                "generate_audio": True,
                "watermark": False,
            },
            model="volcengine/ep-test-123",
            drop_params=False,
        )

        assert mapped_params["duration"] == 11
        assert mapped_params["ratio"] == "16:9"
        assert mapped_params["generate_audio"] is True
        assert mapped_params["watermark"] is False

    def test_transform_video_create_request_builds_content_payload(self):
        data, files, url = self.config.transform_video_create_request(
            model="volcengine/ep-test-123",
            prompt="Create a polished fruit tea ad",
            api_base="https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks",
            video_create_optional_request_params={
                "ratio": "16:9",
                "duration": 11,
                "generate_audio": True,
                "input_reference": "https://example.com/reference.png",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["model"] == "ep-test-123"
        assert data["ratio"] == "16:9"
        assert data["duration"] == 11
        assert data["generate_audio"] is True
        assert data["content"][0] == {
            "type": "text",
            "text": "Create a polished fruit tea ad",
        }
        assert data["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/reference.png"},
            "role": "reference_image",
        }
        assert files == []
        assert (
            url == "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
        )

    def test_transform_video_create_response_encodes_id_and_usage(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {"id": "cgt-20260402175225-7g6f9"}

        request_data = {
            "model": "ep-test-123",
            "duration": 11,
            "ratio": "16:9",
            "content": [{"type": "text", "text": "Create a polished fruit tea ad"}],
        }

        result = self.config.transform_video_create_response(
            model="volcengine/ep-test-123",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="volcengine",
            request_data=request_data,
        )

        decoded = decode_video_id_with_provider(result.id)
        assert decoded["custom_llm_provider"] == "volcengine"
        assert decoded["video_id"] == "cgt-20260402175225-7g6f9"
        assert decoded["model_id"] == "volcengine/ep-test-123"
        assert result.status == "queued"
        assert result.model == "ep-test-123"
        assert result.seconds == "11"
        assert result.size == "16:9"
        assert result.usage == {"duration_seconds": 11.0}

    def test_transform_video_status_retrieve_response_maps_task(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "cgt-20260402175225-7g6f9",
            "model": "ep-test-123",
            "status": "succeeded",
            "created_at": 1712697600,
            "updated_at": 1712697660,
            "duration": 11,
            "ratio": "16:9",
            "content": {
                "video_url": "https://example.com/generated.mp4",
                "audio_url": "https://example.com/generated.mp3",
            },
        }

        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="volcengine",
        )

        decoded = decode_video_id_with_provider(result.id)
        assert decoded["custom_llm_provider"] == "volcengine"
        assert decoded["video_id"] == "cgt-20260402175225-7g6f9"
        assert result.status == "completed"
        assert result.progress == 100
        assert result.created_at == 1712697600
        assert result.completed_at == 1712697660
        assert result.model == "ep-test-123"
        assert result._hidden_params["video_url"] == "https://example.com/generated.mp4"
        assert result._hidden_params["audio_url"] == "https://example.com/generated.mp3"

    def test_transform_video_content_request_decodes_video_id(self):
        encoded_video_id = encode_video_id_with_provider(
            "cgt-20260402175225-7g6f9",
            "volcengine",
            "ep-test-123",
        )

        url, params = self.config.transform_video_content_request(
            video_id=encoded_video_id,
            api_base="https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert (
            url
            == "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/cgt-20260402175225-7g6f9"
        )
        assert params == {}

    def test_transform_video_content_response_downloads_bytes(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "cgt-20260402175225-7g6f9",
            "status": "succeeded",
            "content": {
                "video_url": "https://example.com/generated.mp4?response-content-disposition=attachment%3Bfilename*%3DUTF-8''%E8%A7%86%E9%A2%91.mp4"
            },
        }

        mock_video_response = Mock(spec=httpx.Response)
        mock_video_response.content = b"video-bytes"

        with patch(
            "litellm.llms.volcengine.videos.transformation._get_httpx_client"
        ) as mock_get_httpx_client:
            mock_client = Mock()
            mock_client.client.get.return_value = mock_video_response
            mock_get_httpx_client.return_value = mock_client

            result = self.config.transform_video_content_response(
                raw_response=mock_response,
                logging_obj=self.mock_logging_obj,
            )

        mock_client.client.get.assert_called_once_with(
            "https://example.com/generated.mp4?response-content-disposition=attachment%3Bfilename*%3DUTF-8''%E8%A7%86%E9%A2%91.mp4"
        )
        mock_video_response.raise_for_status.assert_called_once()
        assert result == b"video-bytes"

    @pytest.mark.asyncio
    async def test_async_transform_video_content_response_downloads_bytes(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "cgt-20260402175225-7g6f9",
            "status": "succeeded",
            "content": {
                "video_url": "https://example.com/generated.mp4?response-content-disposition=attachment%3Bfilename*%3DUTF-8''%E8%A7%86%E9%A2%91.mp4"
            },
        }

        mock_video_response = Mock(spec=httpx.Response)
        mock_video_response.content = b"async-video-bytes"

        with patch(
            "litellm.llms.volcengine.videos.transformation.get_async_httpx_client"
        ) as mock_get_async_httpx_client:
            mock_client = Mock()
            mock_client.client.get = AsyncMock(return_value=mock_video_response)
            mock_get_async_httpx_client.return_value = mock_client

            result = await self.config.async_transform_video_content_response(
                raw_response=mock_response,
                logging_obj=self.mock_logging_obj,
            )

        mock_client.client.get.assert_awaited_once_with(
            "https://example.com/generated.mp4?response-content-disposition=attachment%3Bfilename*%3DUTF-8''%E8%A7%86%E9%A2%91.mp4"
        )
        mock_video_response.raise_for_status.assert_called_once()
        assert result == b"async-video-bytes"

    def test_transform_video_list_request_and_response(self):
        encoded_after = encode_video_id_with_provider(
            "cgt-20260402175225-7g6f9",
            "volcengine",
            "ep-test-123",
        )

        url, params = self.config.transform_video_list_request(
            api_base="https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks",
            litellm_params=GenericLiteLLMParams(),
            headers={},
            after=encoded_after,
            limit=20,
            extra_query={"filter.status": "succeeded"},
        )

        assert (
            url == "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
        )
        assert params["page_size"] == "20"
        assert params["filter.task_ids"] == "cgt-20260402175225-7g6f9"
        assert params["filter.status"] == "succeeded"

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "items": [
                {
                    "id": "cgt-20260402175225-7g6f9",
                    "model": "ep-test-123",
                    "status": "succeeded",
                    "created_at": 1712697600,
                    "updated_at": 1712697660,
                    "content": {"video_url": "https://example.com/generated-1.mp4"},
                },
                {
                    "id": "cgt-20260402175225-8h7g0",
                    "model": "ep-test-123",
                    "status": "queued",
                    "created_at": 1712697700,
                },
            ],
            "page_num": 1,
            "page_size": 20,
            "total": 21,
        }

        result = self.config.transform_video_list_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="volcengine",
        )

        first_decoded = decode_video_id_with_provider(result["first_id"])
        last_decoded = decode_video_id_with_provider(result["last_id"])
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        assert first_decoded["video_id"] == "cgt-20260402175225-7g6f9"
        assert last_decoded["video_id"] == "cgt-20260402175225-8h7g0"
        assert result["has_more"] is True
        assert result["data"][0]["status"] == "completed"
        assert result["data"][1]["status"] == "queued"

    def test_transform_video_delete_response_uses_request_path_for_empty_body(self):
        request = httpx.Request(
            "DELETE",
            "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/cgt-20260402175225-7g6f9",
        )
        response = httpx.Response(
            200,
            request=request,
            json={},
        )

        result = self.config.transform_video_delete_response(
            raw_response=response,
            logging_obj=self.mock_logging_obj,
        )

        assert result.id == "cgt-20260402175225-7g6f9"
        assert result.status == "deleted"

    def test_provider_config_manager_returns_volcengine_video_config(self):
        config = ProviderConfigManager.get_provider_video_config(
            model=None,
            provider=LlmProviders.VOLCENGINE,
        )

        assert isinstance(config, VolcEngineVideoConfig)
