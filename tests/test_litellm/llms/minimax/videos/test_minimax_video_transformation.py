"""Tests for MiniMax v1 video generation transformations."""

from unittest.mock import Mock, patch

import httpx

import litellm
from litellm.llms.minimax.videos.transformation import MinimaxVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider
from litellm.utils import ProviderConfigManager


class TestMinimaxVideoTransformation:
    def setup_method(self):
        self.config = MinimaxVideoConfig()
        self.logging_obj = Mock()

    def test_provider_config_is_registered(self):
        config = ProviderConfigManager.get_provider_video_config(
            model="MiniMax-Hailuo-2.3",
            provider=litellm.LlmProviders.MINIMAX,
        )
        assert isinstance(config, MinimaxVideoConfig)

    def test_transform_create_request_maps_text_and_image_parameters(self):
        params = self.config.map_openai_params(
            {
                "input_reference": "https://example.com/frame.png",
                "seconds": "6",
                "size": "768P",
                "extra_body": {"prompt_optimizer": True},
            },
            model="MiniMax-Hailuo-2.3",
            drop_params=False,
        )
        data, files, url = self.config.transform_video_create_request(
            model="MiniMax-Hailuo-2.3",
            prompt="A city at sunrise",
            api_base="https://api.minimax.io/v1",
            video_create_optional_request_params=params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data == {
            "model": "MiniMax-Hailuo-2.3",
            "prompt": "A city at sunrise",
            "first_frame_image": "https://example.com/frame.png",
            "duration": 6,
            "resolution": "768P",
            "prompt_optimizer": True,
        }
        assert files == []
        assert url == "https://api.minimax.io/v1/video_generation"

    def test_create_response_wraps_task_id_and_maps_status(self):
        response = httpx.Response(
            200,
            json={"task_id": "task-123", "base_resp": {"status_code": 0}},
        )
        result = self.config.transform_video_create_response(
            model="MiniMax-Hailuo-2.3",
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="minimax",
            request_data={"duration": 6, "resolution": "768P"},
        )

        assert isinstance(result, VideoObject)
        decoded = decode_video_id_with_provider(result.id)
        assert decoded["custom_llm_provider"] == "minimax"
        assert decoded["model_id"] == "MiniMax-Hailuo-2.3"
        assert decoded["video_id"] == "task-123"
        assert result.status == "queued"
        assert result.seconds == "6"
        assert result.size == "768P"

    def test_status_request_and_response(self):
        encoded_id = encode_video_id_with_provider("task-123", "minimax", "MiniMax-Hailuo-2.3")
        url, data = self.config.transform_video_status_retrieve_request(
            video_id=encoded_id,
            api_base="https://api.minimaxi.com/v1",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == "https://api.minimaxi.com/v1/query/video_generation?task_id=task-123"
        assert data == {}

        response = httpx.Response(
            200,
            json={
                "task_id": "task-123",
                "status": "Success",
                "file_id": "file-123",
                "base_resp": {"status_code": 0},
            },
        )
        result = self.config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="minimax",
        )
        decoded = decode_video_id_with_provider(result.id)
        assert decoded["video_id"] == "task-123"
        assert decoded["custom_llm_provider"] == "minimax"
        assert result.status == "completed"

    def test_content_response_retrieves_file_and_downloads_video(self):
        query_request = httpx.Request(
            "GET",
            "https://api.minimax.io/v1/query/video_generation?task_id=task-123",
            headers={"Authorization": "Bearer test-key"},
        )
        query_response = httpx.Response(
            200,
            json={"task_id": "task-123", "status": "Success", "file_id": "file-123"},
            request=query_request,
        )
        file_response = httpx.Response(
            200,
            json={"file": {"download_url": "https://cdn.example.com/video.mp4"}},
            request=httpx.Request("GET", "https://api.minimax.io/v1/files/retrieve"),
        )
        video_response = httpx.Response(
            200,
            content=b"video-bytes",
            headers={"content-type": "video/mp4"},
            request=httpx.Request("GET", "https://cdn.example.com/video.mp4"),
        )
        client = Mock()
        client.get.side_effect = [file_response, video_response]

        with patch(
            "litellm.llms.minimax.videos.transformation._get_httpx_client",
            return_value=client,
        ):
            result = self.config.transform_video_content_response(query_response, self.logging_obj)

        assert result == b"video-bytes"
        assert client.get.call_args_list[0].args[0] == ("https://api.minimax.io/v1/files/retrieve?file_id=file-123")
        assert client.get.call_args_list[0].kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert client.get.call_args_list[1].args[0] == "https://cdn.example.com/video.mp4"
