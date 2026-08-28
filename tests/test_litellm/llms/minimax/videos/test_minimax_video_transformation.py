"""Tests for MiniMax v1 video generation transformations."""

from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

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

    def test_supported_params_and_environment(self):
        assert self.config.get_supported_openai_params("MiniMax-Hailuo-2.3") == [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
            "extra_body",
            "prompt_optimizer",
            "fast_pretreatment",
            "duration",
            "resolution",
            "callback_url",
        ]

        headers = self.config.validate_environment(
            headers={"X-Test": "value"},
            model="MiniMax-Hailuo-2.3",
            litellm_params=GenericLiteLLMParams(api_key="params-key"),
        )
        assert headers == {
            "X-Test": "value",
            "Authorization": "Bearer params-key",
            "Content-Type": "application/json",
        }

        explicit_headers = self.config.validate_environment(
            headers={},
            model="MiniMax-Hailuo-2.3",
            api_key="explicit-key",
            litellm_params=GenericLiteLLMParams(api_key="params-key"),
        )
        assert explicit_headers["Authorization"] == "Bearer explicit-key"

        with (
            patch.object(litellm, "api_key", None),
            patch(
                "litellm.llms.minimax.videos.transformation.get_secret_str",
                return_value=None,
            ),
            pytest.raises(ValueError, match="MiniMax API key is required"),
        ):
            self.config.validate_environment(
                headers={},
                model="MiniMax-Hailuo-2.3",
                litellm_params=GenericLiteLLMParams(),
            )

    @pytest.mark.parametrize(
        ("api_base", "expected"),
        [
            ("https://api.minimax.io/v1/video_generation", "https://api.minimax.io/v1"),
            ("https://api.minimaxi.com/v1/query/video_generation", "https://api.minimaxi.com/v1"),
            ("https://api.minimax.io/custom/", "https://api.minimax.io/custom/v1"),
        ],
    )
    def test_get_complete_url(self, api_base, expected):
        assert (
            self.config.get_complete_url(
                model="MiniMax-Hailuo-2.3",
                api_base=api_base,
                litellm_params={},
            )
            == expected
        )

    def test_get_complete_url_uses_configured_default(self):
        with patch(
            "litellm.llms.minimax.videos.transformation.get_secret_str",
            return_value="https://api.minimaxi.com/v1/files/retrieve",
        ):
            assert (
                self.config.get_complete_url(
                    model="MiniMax-Hailuo-2.3",
                    api_base=None,
                    litellm_params={},
                )
                == "https://api.minimaxi.com/v1"
            )

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

    def test_file_inputs_are_encoded_as_data_urls(self, tmp_path):
        image_bytes = b"\x89PNG\r\n\x1a\nimage"

        byte_params = self.config.map_openai_params(
            {"input_reference": image_bytes},
            model="MiniMax-Hailuo-2.3",
            drop_params=False,
        )
        assert byte_params["first_frame_image"].startswith("data:image/png;base64,")

        image_file = BytesIO(image_bytes)
        image_file.seek(2)
        file_params = self.config.map_openai_params(
            {"input_reference": image_file},
            model="MiniMax-Hailuo-2.3",
            drop_params=False,
        )
        assert file_params["first_frame_image"] == byte_params["first_frame_image"]
        assert image_file.tell() == 2

        tuple_params = self.config.map_openai_params(
            {"input_reference": ("frame.webp", image_bytes, "image/webp")},
            model="MiniMax-Hailuo-2.3",
            drop_params=False,
        )
        assert tuple_params["first_frame_image"].startswith("data:image/webp;base64,")

        image_path = tmp_path / "frame.png"
        image_path.write_bytes(image_bytes)
        path_params = self.config.map_openai_params(
            {"input_reference": image_path},
            model="MiniMax-Hailuo-2.3",
            drop_params=False,
        )
        assert path_params["first_frame_image"] == byte_params["first_frame_image"]

    def test_invalid_file_inputs_are_rejected(self):
        with pytest.raises(ValueError, match="tuple must include file content"):
            self.config.map_openai_params(
                {"input_reference": ("frame.png",)},
                model="MiniMax-Hailuo-2.3",
                drop_params=False,
            )

        with pytest.raises(TypeError, match="URL, path, bytes, or file object"):
            self.config.map_openai_params(
                {"input_reference": object()},
                model="MiniMax-Hailuo-2.3",
                drop_params=False,
            )

        text_file = Mock()
        text_file.read.return_value = "not-bytes"
        with pytest.raises(TypeError, match="file content must be bytes"):
            self.config.map_openai_params(
                {"input_reference": text_file},
                model="MiniMax-Hailuo-2.3",
                drop_params=False,
            )

    def test_map_params_skips_empty_values_and_preserves_provider_fields(self):
        params = self.config.map_openai_params(
            {
                "model": "ignored",
                "prompt": "ignored",
                "user": "ignored",
                "extra_headers": {"X-Test": "ignored"},
                "seconds": "not-a-number",
                "prompt_optimizer": False,
                "callback_url": None,
                "extra_body": {
                    "fast_pretreatment": True,
                    "prompt_optimizer": None,
                },
            },
            model="MiniMax-Hailuo-2.3",
            drop_params=False,
        )
        assert params == {
            "duration": "not-a-number",
            "prompt_optimizer": False,
            "fast_pretreatment": True,
        }

        assert (
            self.config.map_openai_params(
                {"extra_body": "ignored"},
                model="MiniMax-Hailuo-2.3",
                drop_params=False,
            )
            == {}
        )

    def test_transform_create_request_removes_sdk_only_fields(self):
        data, files, url = self.config.transform_video_create_request(
            model="MiniMax-Hailuo-2.3",
            prompt="A city at sunrise",
            api_base="https://api.minimax.io/v1/",
            video_create_optional_request_params={
                "duration": 6,
                "extra_headers": {"X-Test": "value"},
                "extra_body": {"ignored": True},
                "user": "user-123",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data == {
            "model": "MiniMax-Hailuo-2.3",
            "prompt": "A city at sunrise",
            "duration": 6,
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
        assert result.usage == {"duration_seconds": 6.0}

    def test_create_response_validates_provider_payload(self):
        with pytest.raises(ValueError, match="did not return a task_id"):
            self.config.transform_video_create_response(
                model="MiniMax-Hailuo-2.3",
                raw_response=httpx.Response(200, json={"base_resp": {"status_code": 0}}),
                logging_obj=self.logging_obj,
            )

        with pytest.raises(Exception, match="quota exceeded"):
            self.config.transform_video_create_response(
                model="MiniMax-Hailuo-2.3",
                raw_response=httpx.Response(
                    200,
                    json={
                        "base_resp": {
                            "status_code": 1001,
                            "status_msg": "quota exceeded",
                        }
                    },
                ),
                logging_obj=self.logging_obj,
            )

        with pytest.raises(Exception, match="upstream error"):
            self.config.transform_video_create_response(
                model="MiniMax-Hailuo-2.3",
                raw_response=httpx.Response(500, text="upstream error"),
                logging_obj=self.logging_obj,
            )

        with pytest.raises(ValueError, match="invalid JSON response"):
            self.config.transform_video_create_response(
                model="MiniMax-Hailuo-2.3",
                raw_response=httpx.Response(200, text="not-json"),
                logging_obj=self.logging_obj,
            )

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

    def test_content_request_and_failed_status_response(self):
        encoded_id = encode_video_id_with_provider("task/id", "minimax", "MiniMax-Hailuo-2.3")
        content_url, content_data = self.config.transform_video_content_request(
            video_id=encoded_id,
            api_base="https://api.minimax.io/v1/",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert content_url == "https://api.minimax.io/v1/query/video_generation?task_id=task%2Fid"
        assert content_data == {}

        result = self.config.transform_video_status_retrieve_response(
            raw_response=httpx.Response(
                200,
                json={
                    "task_id": "task-123",
                    "model": "MiniMax-Hailuo-2.3",
                    "status": "Failed",
                    "duration": 10,
                    "resolution": "1080P",
                    "base_resp": {"status_code": "0"},
                },
            ),
            logging_obj=self.logging_obj,
        )
        assert result.id == "task-123"
        assert result.status == "failed"
        assert result.error == {
            "code": "generation_failed",
            "message": "Failed",
        }
        assert result.seconds == "10"
        assert result.size == "1080P"

        with pytest.raises(ValueError, match="did not return a task_id"):
            self.config.transform_video_status_retrieve_response(
                raw_response=httpx.Response(200, json={"base_resp": {"status_code": 0}}),
                logging_obj=self.logging_obj,
            )

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
        client.get.return_value = file_response

        with (
            patch(
                "litellm.llms.minimax.videos.transformation._get_httpx_client",
                return_value=client,
            ),
            patch(
                "litellm.llms.minimax.videos.transformation.safe_get",
                return_value=video_response,
            ) as safe_get_mock,
        ):
            result = self.config.transform_video_content_response(query_response, self.logging_obj)

        assert result == b"video-bytes"
        assert client.get.call_args_list[0].args[0] == ("https://api.minimax.io/v1/files/retrieve?file_id=file-123")
        assert client.get.call_args_list[0].kwargs["headers"]["Authorization"] == "Bearer test-key"
        safe_get_mock.assert_called_once_with(client, "https://cdn.example.com/video.mp4")

    def test_content_response_handles_binary_and_incomplete_results(self):
        query_response = httpx.Response(
            200,
            json={"task_id": "task-123", "status": "Success", "file_id": "file-123"},
            request=httpx.Request(
                "GET",
                "https://api.minimax.io/v1/query/video_generation?task_id=task-123",
            ),
        )
        binary_response = httpx.Response(
            200,
            content=b"video-bytes",
            headers={"content-type": "application/octet-stream"},
        )
        client = Mock()
        client.get.return_value = binary_response

        with patch(
            "litellm.llms.minimax.videos.transformation._get_httpx_client",
            return_value=client,
        ):
            assert self.config.transform_video_content_response(query_response, self.logging_obj) == b"video-bytes"

        with pytest.raises(ValueError, match="not ready for download"):
            self.config.transform_video_content_response(
                httpx.Response(200, json={"status": "Processing"}),
                self.logging_obj,
            )

        client.get.return_value = httpx.Response(200, json={"base_resp": {"status_code": 0}})
        with (
            patch(
                "litellm.llms.minimax.videos.transformation._get_httpx_client",
                return_value=client,
            ),
            pytest.raises(ValueError, match="did not return a video download URL"),
        ):
            self.config.transform_video_content_response(query_response, self.logging_obj)

    @pytest.mark.asyncio
    async def test_async_content_response_downloads_video(self):
        query_response = httpx.Response(
            200,
            json={"task_id": "task-123", "status": "Success", "file_id": "file-123"},
            request=httpx.Request(
                "GET",
                "https://api.minimax.io/v1/query/video_generation?task_id=task-123",
                headers={"Authorization": "Bearer test-key"},
            ),
        )
        file_response = httpx.Response(
            200,
            json={"download_url": "https://cdn.example.com/video.mp4"},
        )
        video_response = httpx.Response(200, content=b"async-video")
        client = Mock()
        client.get = AsyncMock(return_value=file_response)

        with (
            patch(
                "litellm.llms.minimax.videos.transformation.get_async_httpx_client",
                return_value=client,
            ),
            patch(
                "litellm.llms.minimax.videos.transformation.async_safe_get",
                new=AsyncMock(return_value=video_response),
            ) as safe_get_mock,
        ):
            result = await self.config.async_transform_video_content_response(
                query_response,
                self.logging_obj,
            )

        assert result == b"async-video"
        safe_get_mock.assert_awaited_once_with(client, "https://cdn.example.com/video.mp4")

    @pytest.mark.asyncio
    async def test_async_content_response_handles_binary_and_incomplete_results(self):
        query_response = httpx.Response(
            200,
            json={"file_id": "file-123"},
            request=httpx.Request("GET", "https://custom.example.com/query/video_generation"),
        )
        client = Mock()
        client.get = AsyncMock(
            return_value=httpx.Response(
                200,
                content=b"video-bytes",
                headers={"content-type": "video/mp4"},
            )
        )
        with patch(
            "litellm.llms.minimax.videos.transformation.get_async_httpx_client",
            return_value=client,
        ):
            assert (
                await self.config.async_transform_video_content_response(
                    query_response,
                    self.logging_obj,
                )
                == b"video-bytes"
            )

        with pytest.raises(ValueError, match="not ready for download"):
            await self.config.async_transform_video_content_response(
                httpx.Response(200, json={"status": "Queued"}),
                self.logging_obj,
            )

        client.get = AsyncMock(return_value=httpx.Response(200, json={"base_resp": {"status_code": 0}}))
        with (
            patch(
                "litellm.llms.minimax.videos.transformation.get_async_httpx_client",
                return_value=client,
            ),
            pytest.raises(ValueError, match="did not return a video download URL"),
        ):
            await self.config.async_transform_video_content_response(
                query_response,
                self.logging_obj,
            )

    def test_unsupported_video_operations(self):
        request_args = {
            "api_base": "https://api.minimax.io/v1",
            "litellm_params": GenericLiteLLMParams(),
            "headers": {},
        }
        with pytest.raises(NotImplementedError, match="remix"):
            self.config.transform_video_remix_request(
                video_id="task-123",
                prompt="new prompt",
                **request_args,
            )
        with pytest.raises(NotImplementedError, match="remix"):
            self.config.transform_video_remix_response(
                raw_response=httpx.Response(200),
                logging_obj=self.logging_obj,
            )
        with pytest.raises(NotImplementedError, match="listing"):
            self.config.transform_video_list_request(**request_args)
        with pytest.raises(NotImplementedError, match="listing"):
            self.config.transform_video_list_response(
                raw_response=httpx.Response(200),
                logging_obj=self.logging_obj,
            )
        with pytest.raises(NotImplementedError, match="deletion"):
            self.config.transform_video_delete_request(video_id="task-123", **request_args)
        with pytest.raises(NotImplementedError, match="deletion"):
            self.config.transform_video_delete_response(
                raw_response=httpx.Response(200),
                logging_obj=self.logging_obj,
            )

    @pytest.mark.parametrize(
        ("provider_status", "openai_status"),
        [
            ("Succeeded", "completed"),
            ("Canceled", "failed"),
            ("Preparing", "queued"),
            ("Processing", "in_progress"),
            (None, "in_progress"),
        ],
    )
    def test_status_mapping(self, provider_status, openai_status):
        assert self.config._map_status(provider_status) == openai_status

    def test_response_helpers(self):
        assert self.config._request_headers(httpx.Response(200)) == {}
        assert self.config._api_base_from_response(httpx.Response(200)) == "https://api.minimax.io/v1"
        custom_response = httpx.Response(
            200,
            request=httpx.Request("GET", "https://custom.example.com/query/video_generation"),
        )
        assert self.config._api_base_from_response(custom_response) == "https://custom.example.com/v1"
        assert self.config._get_download_url({"url": "https://cdn.example.com/top.mp4"}) == (
            "https://cdn.example.com/top.mp4"
        )
        assert self.config._get_download_url({"file": {"url": "https://cdn.example.com/file.mp4"}}) == (
            "https://cdn.example.com/file.mp4"
        )
        assert self.config._get_download_url({}) is None

        empty_video = VideoObject(id="task-123", object="video", status="queued")
        assert self.config._usage_from_video(empty_video) == {}
        empty_video.seconds = "invalid"
        assert self.config._usage_from_video(empty_video) == {}

        video_data = {}
        self.config._add_request_metadata(video_data, None)
        assert video_data == {}
