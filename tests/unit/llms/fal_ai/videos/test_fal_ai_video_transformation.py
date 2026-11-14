import json
import sys
from typing import Final
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import respx
from litellm.exceptions import UnsupportedParamsError

import litellm
import litellm.llms.fal_ai.videos.transformation as fal_video_module
from litellm.cost_calculator import default_video_cost_calculator
from litellm.llms.fal_ai.cost_calculator import fal_ai_passthrough_cost
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
H3_MAX_TEXT_MODEL = "minimax/h3-max/text-to-video"
H3_MAX_TURBO_IMAGE_MODEL = "minimax/h3-max-turbo/image-to-video"
H3_MAX_RECAST_MODEL = "minimax/h3-max/recast"


@pytest.mark.asyncio
@pytest.mark.parametrize("pricing_source", ["litellm_params", "model_info"])
@pytest.mark.parametrize(
    ("pricing", "expected_cost"),
    [
        ({"output_cost_per_second": 0.017}, 0.085),
        ({"output_cost_per_video_clip": {"5s": {"480p": 0.19}}}, 0.19),
    ],
)
async def test_router_video_uses_configured_deployment_price(
    monkeypatch: pytest.MonkeyPatch, pricing_source: str, pricing: dict[str, object], expected_cost: float
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost})
    router: Final = litellm.Router(
        num_retries=0,
        model_list=[
            {
                "model_name": "configured-video",
                "litellm_params": {
                    "model": "fal_ai/fal-ai/longcat-video/distilled/text-to-video/480p",
                    "api_key": "synthetic-fal-key",
                    **(pricing if pricing_source == "litellm_params" else {}),
                },
                "model_info": pricing if pricing_source == "model_info" else {},
            }
        ],
    )
    with respx.mock:
        upstream: Final = respx.post("https://queue.fal.run/fal-ai/longcat-video/distilled/text-to-video/480p").mock(
            return_value=httpx.Response(200, json={"request_id": "configured-priced-clip"})
        )
        video: Final = await router.avideo_generation(model="configured-video", prompt="clip", seconds="5", size="480p")
    assert video._hidden_params["response_cost"] == pytest.approx(expected_cost)
    assert upstream.call_count == 1


@pytest.mark.parametrize("drop_params", [False, True])
@pytest.mark.parametrize(
    "model",
    [
        "fal-ai/pixverse/v5/text-to-video",
        "pixverse/v6/text-to-video",
        "fal_ai/pixverse/v6/text-to-video",
        "fal_ai/fal-ai/pixverse/v5/text-to-video",
    ],
)
@pytest.mark.parametrize(
    ("size", "resolution"),
    [
        ("360p", "360p"),
        ("540p", "540p"),
        ("720p", "720p"),
        ("1080p", "1080p"),
        ("640x360", "360p"),
        ("960x540", "540p"),
        ("720x1280", "720p"),
    ],
)
def test_pixverse_resolution_maps_to_request_and_billing(
    size: str, resolution: str, drop_params: bool, model: str
) -> None:
    config: Final = FalAIVideoConfig()
    params: Final = config.map_openai_params({"seconds": "5", "size": size}, model, drop_params)
    assert params["resolution"] == resolution
    response: Final = config.transform_video_create_response(
        model=model,
        raw_response=httpx.Response(
            200, json={"request_id": "pixverse-clip"}, request=httpx.Request("POST", "https://queue.fal.run")
        ),
        logging_obj=Mock(),
        request_data=params,
    )
    assert response.usage is not None
    assert response.usage["video_resolution"] == resolution


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["fal-ai/pixverse/v5/text-to-video", "pixverse/v6/text-to-video"])
@pytest.mark.parametrize(
    ("seconds", "native_duration", "expected_duration", "expected_cost"),
    [(None, None, "5", 0.19), ("8", None, "8", 0.31), (None, "8", "8", 0.31)],
)
async def test_pixverse_duration_reaches_request_usage_and_clip_billing(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    seconds: str | None,
    native_duration: str | None,
    expected_duration: str,
    expected_cost: float,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost})
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "pixverse-clip",
                "litellm_params": {"model": f"fal_ai/{model}", "api_key": "synthetic-fal-key"},
                "model_info": {
                    "output_cost_per_video_clip": {
                        "4s": {"540p": 0.04},
                        "5s": {"540p": 0.19, "720p": 0.61},
                        "8s": {"540p": 0.31},
                    }
                },
            }
        ]
    )
    with respx.mock:
        upstream: Final = respx.post(f"https://queue.fal.run/{model}").respond(
            200, json={"request_id": "pixverse-duration"}
        )
        video: Final = await router.avideo_generation(
            model="pixverse-clip",
            prompt="clip",
            size="540p",
            seconds=seconds,
            extra_body={"duration": native_duration} if native_duration is not None else None,
        )
    assert upstream.call_count == 1
    assert json.loads(upstream.calls.last.request.content)["duration"] == expected_duration
    assert video.seconds == expected_duration
    assert video.usage == {"duration_seconds": float(expected_duration), "video_resolution": "540p"}
    assert video._hidden_params["response_cost"] == pytest.approx(expected_cost)


@pytest.mark.parametrize("drop_params", [False, True])
def test_native_video_parameters_keep_precedence(drop_params: bool) -> None:
    params: Final = {"aspect_ratio": "1:1", "seed": 0, "duration": 6}
    assert FalAIVideoConfig().map_openai_params({"seconds": "5", "size": "1280x720", **params}, MODEL, drop_params) == {
        "resolution": "720p",
        **params,
    }


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

    def test_map_openai_params_supports_h3_max_profile(self) -> None:
        url: Final = "https://example.com/image.png"

        assert self.config.map_openai_params({"size": "768p"}, H3_MAX_TEXT_MODEL, False) == {"resolution": "768P"}
        assert self.config.map_openai_params({"size": "1920x1080"}, H3_MAX_TEXT_MODEL, False) == {
            "resolution": "1080P",
            "aspect_ratio": "16:9",
        }
        mapped: Final = self.config.map_openai_params(
            {"seconds": 5.5, "input_reference": url, "size": "1280x768"},
            H3_MAX_TURBO_IMAGE_MODEL,
            False,
        )
        assert mapped["resolution"] == "768P"
        assert mapped["image_url"] == url
        assert "reference_image_urls" not in mapped
        assert mapped["duration"] == "5"

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("bytedance/seedance-2.5/reference-to-video", {"image_urls": ["https://example.com/image.png"]}),
            ("bytedance/seedance-2.0/fast/reference-to-video", {"image_urls": ["https://example.com/image.png"]}),
            ("bytedance/seedance-2.5/image-to-video", {"image_url": "https://example.com/image.png"}),
        ],
    )
    def test_map_openai_params_seedance_input_reference_follows_task(self, model: str, expected: dict[str, object]):
        mapped = self.config.map_openai_params({"input_reference": "https://example.com/image.png"}, model, False)

        assert mapped == expected

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

    def test_transform_video_create_response_luma_ray_defaults_to_540p(self):
        response = Mock(spec=httpx.Response)
        response.json.return_value = {"request_id": "abc"}

        for model in (
            "luma/agent/ray/v3.2/text-to-video",
            "luma/agent/ray/v3.2/image-to-video",
            "luma/agent/ray/v3.2/reframe",
            "luma/agent/ray/v3.2/video-to-video",
        ):
            video = self.config.transform_video_create_response(
                model=model,
                raw_response=response,
                logging_obj=self.logging_obj,
                custom_llm_provider="fal_ai",
                request_data={"duration": "5"},
            )
            assert video.usage == {"duration_seconds": 5.0, "video_resolution": "540p"}, model
            row = litellm.model_cost[f"fal_ai/{model}"]
            assert default_video_cost_calculator(model, 5, "fal_ai", video_resolution="540p") == (
                5 * row["output_cost_per_second"]
            ), model

    def test_transform_video_create_response_h3_max_leaves_default_resolution_to_cost_map(
        self, local_model_cost_map: None
    ) -> None:
        response: Final = Mock(spec=httpx.Response)
        response.json.return_value = {"request_id": "abc"}

        video: Final = self.config.transform_video_create_response(
            model=H3_MAX_RECAST_MODEL,
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="fal_ai",
            request_data={"duration": 5},
        )

        assert video.usage == {"duration_seconds": 5.0}
        row: Final = litellm.model_cost[f"fal_ai/{H3_MAX_RECAST_MODEL}"]
        assert default_video_cost_calculator(
            model=H3_MAX_RECAST_MODEL,
            duration_seconds=5.0,
            custom_llm_provider="fal_ai",
            model_info=None,
            video_resolution=None,
        ) == pytest.approx(row["output_cost_per_second"] * 5.0)

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

    @pytest.mark.parametrize(("duration", "resolution"), [(0, None), (5, "720p"), (17, "9999p")])
    def test_video_cost_uses_tiered_rows(self, duration: float, resolution: str | None) -> None:
        rows = {
            model: row
            for model, row in litellm.model_cost.items()
            if row.get("litellm_provider") == "fal_ai"
            and row.get("mode") == "video_generation"
            and not row.get("output_cost_per_video_clip")
        }
        assert rows
        for model, row in rows.items():
            if (flat_cost := row.get("output_cost_per_video")) is not None:
                assert (
                    default_video_cost_calculator(model, duration, "fal_ai", video_resolution=resolution) == flat_cost
                )
                continue
            for key, value in row.items():
                if key.startswith("output_cost_per_second_") and value is not None:
                    tier = key.removeprefix("output_cost_per_second_")
                    assert default_video_cost_calculator(model, 5, "fal_ai", video_resolution=tier) == 5 * value
            assert default_video_cost_calculator(model, 5, "fal_ai", video_resolution="9999p") == (
                5 * row["output_cost_per_second"]
            )

    def test_video_cost_uses_catalog_clip_cells(self) -> None:
        models: Final = tuple(
            model
            for model, row in litellm.model_cost.items()
            if row.get("litellm_provider") == "fal_ai"
            and row.get("mode") == "video_generation"
            and row.get("output_cost_per_video_clip")
        )
        pixverse_model: Final = "fal_ai/fal-ai/pixverse/v5/text-to-video"
        assert pixverse_model in models
        pixverse_prices: Final = litellm.get_model_info(model=pixverse_model, custom_llm_provider="fal_ai").get(
            "output_cost_per_video_clip"
        )
        assert pixverse_prices
        pixverse_tiers: Final = next(iter(pixverse_prices.values()))
        assert next(iter(pixverse_tiers.values())) > 0
        for model in models:
            info: Final = litellm.get_model_info(model=model, custom_llm_provider="fal_ai")
            prices: Final = info.get("output_cost_per_video_clip")
            assert prices is not None
            for duration, tiers in prices.items():
                seconds: Final = float(duration.removesuffix("s"))
                for resolution, price in tiers.items():
                    assert default_video_cost_calculator(model, seconds, "fal_ai", video_resolution=resolution) == price
                    assert (
                        video_generation_cost(model, seconds, "fal_ai", model_info=info, video_resolution=resolution)
                        == price
                    )

    @pytest.mark.parametrize(
        "model",
        [
            "fal-ai/pixverse/v3.5/effects",
            "fal-ai/pixverse/v3.5/transition",
            "fal-ai/pixverse/v4/effects",
            "fal-ai/pixverse/v4.5/effects",
            "fal-ai/pixverse/v4.5/transition",
            "fal-ai/pixverse/v5/effects",
            "fal-ai/pixverse/v5/transition",
            "fal-ai/pixverse/extend",
        ],
    )
    def test_pixverse_five_second_clip_prices(self, model: str) -> None:
        info: Final = litellm.get_model_info(model=model, custom_llm_provider="fal_ai")
        prices: Final = info.get("output_cost_per_video_clip")
        assert prices is not None
        for resolution, price in prices["5s"].items():
            assert price > 0
            assert default_video_cost_calculator(model, 5, "fal_ai", video_resolution=resolution) == pytest.approx(price)
            assert video_generation_cost(
                model, 5, "fal_ai", model_info=info, video_resolution=resolution
            ) == pytest.approx(price)
            assert fal_ai_passthrough_cost(
                model, {"duration": "5", "resolution": resolution}
            ) == pytest.approx(price)

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


@pytest.mark.parametrize("field", ["frame_images", "input_references"])
def test_first_class_video_fields_are_rejected_or_dropped(field: str) -> None:
    config: Final = FalAIVideoConfig()
    with pytest.raises(UnsupportedParamsError, match=field):
        config.map_openai_params({field: 1}, "video-model", False)
    assert config.map_openai_params({field: 1}, "video-model", True) == config.map_openai_params(
        {}, "video-model", True
    )
