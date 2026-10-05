import sys
from typing import Final
from unittest.mock import AsyncMock, Mock

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
from litellm.llms.openai.cost_calculation import video_generation_cost
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.utils import decode_video_id_with_provider
from litellm.utils import ProviderConfigManager

MODEL = "bytedance/seedance-2.5/text-to-video"
H3_TEXT_MODEL = "minimax/h3/text-to-video"
H3_REFERENCE_MODEL = "minimax/h3/reference-to-video"


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

    def test_map_openai_params_supports_h3_profiles(self):
        url = "https://example.com/image.png"

        assert self.config.map_openai_params({"size": "2k"}, H3_TEXT_MODEL, False) == {"resolution": "2K"}
        assert self.config.map_openai_params({"size": "1024x768"}, H3_TEXT_MODEL, False) == {
            "resolution": "768P",
            "aspect_ratio": "4:3",
        }
        mapped = self.config.map_openai_params(
            {"seconds": 6, "input_reference": url},
            H3_REFERENCE_MODEL,
            False,
        )
        assert mapped["duration"] == 6
        assert isinstance(mapped["duration"], int)
        assert mapped["reference_image_urls"] == [url]
        assert "image_url" not in mapped

    def test_map_openai_params_h3_omits_auto_duration(self):
        assert self.config.map_openai_params({"seconds": "auto"}, H3_TEXT_MODEL, False) == {}
        assert self.config.map_openai_params({"seconds": "auto"}, MODEL, False) == {"duration": "auto"}

    def test_map_openai_params_h3_size_beyond_tiers_uses_top_resolution(self):
        side = str(sys.maxsize + 1)
        assert self.config.map_openai_params({"size": f"{side}x{side}"}, H3_TEXT_MODEL, False) == {
            "resolution": "4K",
            "aspect_ratio": "1:1",
        }

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

    def test_get_complete_url_respects_api_base_override(self):
        url = self.config.get_complete_url(
            model=MODEL,
            api_base="https://proxy.internal/",
            litellm_params={},
        )

        assert url == "https://proxy.internal"

    def test_validate_environment_requires_fal_ai_api_key(self, monkeypatch):
        monkeypatch.setattr(fal_video_module, "get_secret_str", lambda _: None)

        with pytest.raises(ValueError, match="FAL_AI_API_KEY is not set"):
            self.config.validate_environment(
                headers={},
                model=MODEL,
                api_key=None,
                litellm_params=GenericLiteLLMParams(),
            )

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

    def test_transform_video_create_response_uses_h3_default_resolution(self):
        response = Mock(spec=httpx.Response)
        response.json.return_value = {"request_id": "abc"}

        video = self.config.transform_video_create_response(
            model=H3_TEXT_MODEL,
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
            request_data={"duration": 5},
        )

        assert video.usage == {"duration_seconds": 5.0, "video_resolution": "2K"}

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
        status_url = "https://queue.fal.run/bytedance/seedance-2.5/requests/abc/status"
        response = httpx.Response(200, json=response_data, request=httpx.Request("GET", status_url))
        config = self.config
        if expected_status == "completed":
            result_response: Final = httpx.Response(
                200,
                json={"video": {"url": "https://cdn.example.com/video.mp4"}},
                request=httpx.Request("GET", status_url.removesuffix("/status")),
            )
            client: Final = Mock()
            client.get.return_value = result_response
            config = FalAIVideoConfig(sync_client_factory=lambda: client)

        video = config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
        )

        assert video.status == expected_status
        assert video.created_at == 0
        decoded = decode_video_id_with_provider(video.id)
        assert decoded["model_id"] == "bytedance/seedance-2.5"
        assert decoded["video_id"] == "abc"

        poll_url, _ = self.config.transform_video_status_retrieve_request(
            video_id=video.id,
            api_base="https://queue.fal.run",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert poll_url == status_url

    def test_status_response_error(self):
        response_data = {
            "request_id": "abc",
            "status": "COMPLETED",
            "error": "generation failed",
        }
        status_url = "https://queue.fal.run/bytedance/seedance-2.5/requests/abc/status"
        response = httpx.Response(
            200,
            json=response_data,
            request=httpx.Request("GET", status_url),
        )
        result_response: Final = httpx.Response(
            200,
            json={"video": {"url": "https://cdn.example.com/video.mp4"}},
            request=httpx.Request("GET", status_url.removesuffix("/status")),
        )
        client: Final = Mock()
        client.get.return_value = result_response
        config = FalAIVideoConfig(sync_client_factory=lambda: client)

        video = config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
        )

        assert video.status == "failed"
        assert video.error == {"code": "fal_error", "message": "generation failed"}

    def test_status_completed_result_error_surfaces_fal_message(self):
        status_url = "https://queue.fal.run/minimax/h3/requests/abc/status"
        auth_headers: Final = {"Authorization": "Key synthetic-fal-key", "Content-Type": "application/json"}
        response: Final = httpx.Response(
            200,
            json={"request_id": "abc", "status": "COMPLETED"},
            request=httpx.Request("GET", status_url, headers=auth_headers),
        )
        result_url: Final = status_url.removesuffix("/status")
        result_response: Final = httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "loc": ["body", "input.reference_image_urls"],
                        "msg": "Failed to download the file. Please check if the URL is accessible and try again.",
                    }
                ]
            },
            request=httpx.Request("GET", result_url, headers=auth_headers),
        )
        client: Final = Mock()
        client.get.return_value = result_response
        config = FalAIVideoConfig(sync_client_factory=lambda: client)

        video = config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
        )

        assert video.status == "failed"
        assert "input.reference_image_urls: Failed to download the file" in video.error["message"]
        client.get.assert_called_once_with(url=result_url, headers=auth_headers)

    @pytest.mark.parametrize("status_code", [429, 503])
    def test_status_completed_transient_result_error_keeps_completed(self, status_code):
        status_url = "https://queue.fal.run/minimax/h3/requests/abc/status"
        response: Final = httpx.Response(
            200,
            json={"request_id": "abc", "status": "COMPLETED"},
            request=httpx.Request("GET", status_url),
        )
        result_response: Final = httpx.Response(
            status_code,
            json={"detail": "temporary fal failure"},
            request=httpx.Request("GET", status_url.removesuffix("/status")),
        )
        client: Final = Mock()
        client.get.return_value = result_response
        config = FalAIVideoConfig(sync_client_factory=lambda: client)

        video = config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
        )

        assert video.status == "completed"
        assert video.error is None

    @pytest.mark.asyncio
    async def test_async_status_completed_result_error_surfaces_fal_message(self):
        status_url = "https://queue.fal.run/minimax/h3/requests/abc/status"
        auth_headers: Final = {"Authorization": "Key synthetic-fal-key", "Content-Type": "application/json"}
        response: Final = httpx.Response(
            200,
            json={"request_id": "abc", "status": "COMPLETED"},
            request=httpx.Request("GET", status_url, headers=auth_headers),
        )
        result_url: Final = status_url.removesuffix("/status")
        result_response: Final = httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "loc": ["body", "input.reference_image_urls"],
                        "msg": "Failed to download the file. Please check if the URL is accessible and try again.",
                    }
                ]
            },
            request=httpx.Request("GET", result_url, headers=auth_headers),
        )
        client: Final = Mock()
        client.get = AsyncMock(return_value=result_response)
        config = FalAIVideoConfig(async_client_factory=lambda: client)

        video = await config.async_transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
        )

        assert video.status == "failed"
        assert "input.reference_image_urls: Failed to download the file" in video.error["message"]
        client.get.assert_awaited_once_with(url=result_url, headers=auth_headers)

    def test_status_completed_result_probe_reuses_status_client_and_extra_headers(self):
        status_url = "https://queue.fal.run/minimax/h3/requests/abc/status"
        status_headers: Final = {
            "Authorization": "Key synthetic-fal-key",
            "Content-Type": "application/json",
            "X-Routing": "canary-7",
        }
        response: Final = httpx.Response(
            200,
            json={"request_id": "abc", "status": "COMPLETED"},
            request=httpx.Request("GET", status_url, headers=status_headers),
        )
        result_url: Final = status_url.removesuffix("/status")
        status_client: Final = Mock()
        status_client.get.return_value = httpx.Response(
            403, text="missing X-Routing", request=httpx.Request("GET", result_url)
        )
        factory_client: Final = Mock()
        config = FalAIVideoConfig(sync_client_factory=lambda: factory_client)

        video = config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
            client=status_client,
        )

        assert video.status == "failed"
        assert video.error == {"code": "fal_error", "message": "missing X-Routing"}
        factory_client.get.assert_not_called()
        status_client.get.assert_called_once()
        assert status_client.get.call_args.kwargs["url"] == result_url
        assert status_client.get.call_args.kwargs["headers"].items() >= status_headers.items()

    def test_status_completed_result_probe_transport_error_keeps_completed(self):
        status_url = "https://queue.fal.run/minimax/h3/requests/abc/status"
        response: Final = httpx.Response(
            200,
            json={"request_id": "abc", "status": "COMPLETED"},
            request=httpx.Request("GET", status_url),
        )
        client: Final = Mock()
        client.get.side_effect = httpx.ReadError("connection reset by fal.ai")

        video = FalAIVideoConfig().transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
            client=client,
        )

        assert video.status == "completed"
        assert video.error is None

    @pytest.mark.asyncio
    async def test_async_status_completed_result_probe_reuses_status_client_and_extra_headers(self):
        status_url = "https://queue.fal.run/minimax/h3/requests/abc/status"
        status_headers: Final = {
            "Authorization": "Key synthetic-fal-key",
            "Content-Type": "application/json",
            "X-Routing": "canary-7",
        }
        response: Final = httpx.Response(
            200,
            json={"request_id": "abc", "status": "COMPLETED"},
            request=httpx.Request("GET", status_url, headers=status_headers),
        )
        result_url: Final = status_url.removesuffix("/status")
        status_client: Final = Mock()
        status_client.get = AsyncMock(
            return_value=httpx.Response(403, text="missing X-Routing", request=httpx.Request("GET", result_url))
        )
        factory_client: Final = Mock()
        factory_client.get = AsyncMock()
        config = FalAIVideoConfig(async_client_factory=lambda: factory_client)

        video = await config.async_transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
            client=status_client,
        )

        assert video.status == "failed"
        assert video.error == {"code": "fal_error", "message": "missing X-Routing"}
        factory_client.get.assert_not_awaited()
        status_client.get.assert_awaited_once()
        assert status_client.get.await_args.kwargs["url"] == result_url
        assert status_client.get.await_args.kwargs["headers"].items() >= status_headers.items()

    @pytest.mark.asyncio
    async def test_async_status_completed_result_probe_transport_error_keeps_completed(self):
        status_url = "https://queue.fal.run/minimax/h3/requests/abc/status"
        response: Final = httpx.Response(
            200,
            json={"request_id": "abc", "status": "COMPLETED"},
            request=httpx.Request("GET", status_url),
        )
        client: Final = Mock()
        client.get = AsyncMock(side_effect=httpx.ConnectError("tls handshake failed"))

        video = await FalAIVideoConfig().async_transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
            client=client,
        )

        assert video.status == "completed"
        assert video.error is None

    def test_status_in_progress_does_not_fetch_result(self):
        status_url = "https://queue.fal.run/minimax/h3/requests/abc/status"
        response = httpx.Response(
            200,
            json={"request_id": "abc", "status": "IN_PROGRESS"},
            request=httpx.Request("GET", status_url),
        )

        client: Final = Mock()
        config = FalAIVideoConfig(sync_client_factory=lambda: client)

        video = config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
        )

        assert video.status == "in_progress"
        client.get.assert_not_called()

    def test_status_response_uses_namespaced_request_url(self):
        response: Final = httpx.Response(
            200,
            json={"status": "IN_PROGRESS"},
            request=httpx.Request(
                "GET",
                "https://example.com/proxy/workflows/owner/app/requests/xyz/status",
            ),
        )

        video = self.config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
        )

        decoded = decode_video_id_with_provider(video.id)
        assert decoded["model_id"] == "workflows/owner/app"
        assert decoded["video_id"] == "xyz"
        assert video.model == "workflows/owner/app"

    def test_content_response_downloads_video_url(self):
        content_response = httpx.Response(
            200,
            content=b"video-bytes",
            request=httpx.Request("GET", "https://cdn.example.com/video.mp4"),
        )

        class FakeHTTPClient:
            def get(self, url):
                assert url == "https://cdn.example.com/video.mp4"
                return content_response

        config = FalAIVideoConfig(sync_client_factory=FakeHTTPClient)
        response = Mock(spec=httpx.Response)
        response.json.return_value = {"video": {"url": "https://cdn.example.com/video.mp4"}}

        assert config.transform_video_content_response(response, self.logging_obj) == b"video-bytes"

    def test_content_response_rejects_missing_video(self):
        response = Mock(spec=httpx.Response)
        response.json.return_value = {"error": "generation failed"}

        with pytest.raises(ValueError, match="generation failed"):
            self.config.transform_video_content_response(response, self.logging_obj)

    def test_content_response_surfaces_list_detail_error(self):
        response: Final = httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "loc": ["body", "input.reference_image_urls"],
                        "msg": "Failed to download the file. Please check if the URL is accessible and try again.",
                    }
                ]
            },
            request=httpx.Request("GET", "https://queue.fal.run/minimax/h3/requests/abc"),
        )

        with pytest.raises(FalAIVideoError) as error:
            self.config.transform_video_content_response(response, self.logging_obj)

        assert error.value.status_code == 422
        assert "input.reference_image_urls: Failed to download the file" in error.value.message
        assert "Failed to download the file" in error.value.response.text

    def test_content_response_surfaces_string_detail_error(self):
        response: Final = httpx.Response(
            400,
            json={"detail": "Request is still in progress"},
            request=httpx.Request("GET", "https://queue.fal.run/minimax/h3/requests/abc"),
        )

        with pytest.raises(FalAIVideoError) as error:
            self.config.transform_video_content_response(response, self.logging_obj)

        assert error.value.status_code == 400
        assert error.value.message == "Request is still in progress"
        assert "Request is still in progress" in error.value.response.text

    @pytest.mark.asyncio
    async def test_async_content_response_surfaces_list_detail_error(self):
        response: Final = httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "loc": ["body", "input.reference_image_urls"],
                        "msg": "Failed to download the file. Please check if the URL is accessible and try again.",
                    }
                ]
            },
            request=httpx.Request("GET", "https://queue.fal.run/minimax/h3/requests/abc"),
        )

        with pytest.raises(FalAIVideoError) as error:
            await self.config.async_transform_video_content_response(response, self.logging_obj)

        assert error.value.status_code == 422
        assert "input.reference_image_urls: Failed to download the file" in error.value.message
        assert "Failed to download the file" in error.value.response.text

    @pytest.mark.asyncio
    async def test_async_content_response_surfaces_string_detail_error(self):
        response = httpx.Response(
            400,
            json={"detail": "Request is still in progress"},
            request=httpx.Request("GET", "https://queue.fal.run/minimax/h3/requests/abc"),
        )

        with pytest.raises(FalAIVideoError) as error:
            await self.config.async_transform_video_content_response(response, self.logging_obj)

        assert error.value.status_code == 400
        assert error.value.message == "Request is still in progress"
        assert "Request is still in progress" in error.value.response.text

    def test_extract_video_url_surfaces_list_detail_error(self):
        response: Final = Mock(spec=httpx.Response)
        response.json.return_value = {
            "detail": [{"loc": ["body", "input.reference_image_urls"], "msg": "Failed to download the file"}]
        }

        with pytest.raises(ValueError, match=r"input\.reference_image_urls: Failed to download the file"):
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
            for key, value in row.items():
                if key.startswith("output_cost_per_second_") and value is not None:
                    tier = key.removeprefix("output_cost_per_second_")
                    assert default_video_cost_calculator(model, 5, "fal_ai", video_resolution=tier) == 5 * value
            assert default_video_cost_calculator(model, 5, "fal_ai", video_resolution="9999p") == (
                5 * row["output_cost_per_second"]
            )

    def test_h3_video_cost_uses_model_info_tiers(self, local_model_cost_map):
        row = litellm.model_cost[f"fal_ai/{H3_TEXT_MODEL}"]
        model_info = litellm.get_model_info(model=H3_TEXT_MODEL, custom_llm_provider="fal_ai")

        assert (
            video_generation_cost(
                model=H3_TEXT_MODEL,
                duration_seconds=5,
                custom_llm_provider="fal_ai",
                model_info=model_info,
                video_resolution="2K",
            )
            == 5 * row["output_cost_per_second_2k"]
        )
        assert (
            video_generation_cost(
                model=H3_TEXT_MODEL,
                duration_seconds=5,
                custom_llm_provider="fal_ai",
                model_info=model_info,
                video_resolution="768p",
            )
            == 5 * row["output_cost_per_second_768p"]
        )
