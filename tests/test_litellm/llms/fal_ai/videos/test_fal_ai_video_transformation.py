"""
Tests for fal.ai video generation transformation.
"""

import base64
import io
import json
from typing import Final

import httpx
import pytest

import litellm
from litellm.cost_calculator import default_video_cost_calculator
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.fal_ai.videos.transformation import FAL_QUEUE_API_BASE, FalAIVideoConfig, FalAIVideoError
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider
from litellm.utils import ProviderConfigManager

PNG_BYTES: Final = b"\x89PNG\r\n\x1a\n" + b"fakepixels"
JPEG_BYTES: Final = b"\xff\xd8\xff\xe0" + b"fakepixels"
MP4_BYTES: Final = b"\x00\x00\x00\x18ftypisom" + b"fakeframes"

STATUS_URL: Final = "https://queue.fal.run/minimax/h3/requests/req-abc-123/status"
RESULT_URL: Final = "https://queue.fal.run/minimax/h3/requests/req-abc-123"


def make_response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("GET", STATUS_URL),
    )


def encoded_id(model: str = "minimax/h3/image-to-video", request_id: str = "req-abc-123") -> str:
    return encode_video_id_with_provider(request_id, "fal_ai", model)


def decode_data_url(data_url: str) -> tuple[str, bytes]:
    head, _, b64 = data_url.partition(";base64,")
    return head.removeprefix("data:"), base64.b64decode(b64)


class _RecordingClient(HTTPHandler):
    """Stand-in for the shared httpx client: replays one response and records the call."""

    def __init__(self, response: httpx.Response | None = None, exc: Exception | None = None):
        super().__init__(client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500))))
        self._response = response
        self._exc = exc
        self.url: str | None = None
        self.headers: dict | None = None

    def get(self, url, params=None, headers=None, follow_redirects=None, timeout=None):
        if self._exc is not None:
            raise self._exc
        self.url = url
        self.headers = headers
        assert self._response is not None
        return self._response


@pytest.fixture
def config() -> FalAIVideoConfig:
    return FalAIVideoConfig()


class TestProviderRegistration:
    def test_fal_ai_resolves_to_video_config(self):
        cfg = ProviderConfigManager.get_provider_video_config(
            model="fal-ai/ltx-video", provider=litellm.LlmProviders.FAL_AI
        )
        assert isinstance(cfg, FalAIVideoConfig)

    def test_get_llm_provider_splits_fal_ai_prefix(self):
        model, provider, _, _ = litellm.get_llm_provider(model="fal_ai/minimax/h3/text-to-video")
        assert provider == "fal_ai"
        assert model == "minimax/h3/text-to-video"


class TestMapOpenAIParams:
    def test_input_reference_maps_to_image_url(self, config):
        mapped = config.map_openai_params(
            {"input_reference": "https://example.com/cat.png"},
            model="minimax/h3/image-to-video",
            drop_params=False,
        )
        assert mapped == {"image_url": "https://example.com/cat.png"}

    @pytest.mark.parametrize(
        "model",
        [
            "alibaba/happy-horse/video-edit",
            "xai/grok-imagine-video/edit-video",
            "blackforestlabs/flux-3/extend-video",
            "some-vendor/some-model/video-to-video",
        ],
    )
    def test_input_reference_maps_to_video_url_for_video_endpoints(self, config, model):
        mapped = config.map_openai_params(
            {"input_reference": "https://example.com/clip.mp4"},
            model=model,
            drop_params=False,
        )
        assert mapped == {"video_url": "https://example.com/clip.mp4"}

    def test_seconds_maps_to_int_duration(self, config):
        mapped = config.map_openai_params({"seconds": "8"}, model="minimax/h3/text-to-video", drop_params=False)
        assert mapped == {"duration": 8}

    def test_seconds_maps_to_string_duration_for_seedance(self, config):
        mapped = config.map_openai_params(
            {"seconds": "8"}, model="bytedance/seedance-2.0/text-to-video", drop_params=False
        )
        assert mapped == {"duration": "8"}

    @pytest.mark.parametrize("model", ["alibaba/happy-horse/video-edit", "fal-ai/ltx-video"])
    def test_seconds_dropped_for_endpoints_without_duration(self, config, model):
        mapped = config.map_openai_params({"seconds": "8"}, model=model, drop_params=False)
        assert "duration" not in mapped

    def test_unparseable_seconds_falls_back_to_family_default(self, config):
        mapped = config.map_openai_params(
            {"seconds": "auto"}, model="xai/grok-imagine-video/text-to-video", drop_params=False
        )
        assert mapped == {"duration": 6}

    def test_size_maps_to_short_side_resolution(self, config):
        mapped = config.map_openai_params(
            {"size": "1280x720"}, model="xai/grok-imagine-video/text-to-video", drop_params=False
        )
        assert mapped == {"resolution": "720p"}

    @pytest.mark.parametrize(
        "size,expected",
        [("1366x768", "768P"), ("2560x1440", "2K"), ("3840x2160", "4K")],
    )
    def test_size_maps_to_uppercase_tiers_for_minimax(self, config, size, expected):
        mapped = config.map_openai_params({"size": size}, model="minimax/h3/text-to-video", drop_params=False)
        assert mapped == {"resolution": expected}

    def test_unknown_params_pass_through_verbatim(self, config):
        mapped = config.map_openai_params(
            {"enable_prompt_expansion": False, "aspect_ratio": "9:16", "seed": 42},
            model="minimax/h3/text-to-video",
            drop_params=False,
        )
        assert mapped == {"enable_prompt_expansion": False, "aspect_ratio": "9:16", "seed": 42}

    def test_openai_only_params_are_not_forwarded(self, config):
        mapped = config.map_openai_params(
            {"user": "someone", "extra_headers": {"x": "y"}, "model": "minimax/h3/text-to-video"},
            model="minimax/h3/text-to-video",
            drop_params=False,
        )
        assert mapped == {}


class TestBinaryInputReference:
    """The proxy hands input_reference over as a file upload. fal's body is JSON, so it must become a data URL."""

    def test_bytesio_with_name_becomes_data_url(self, config):
        buf = io.BytesIO(PNG_BYTES)
        buf.name = "input_image.png"
        mapped = config.map_openai_params(
            {"input_reference": buf}, model="alibaba/happy-horse/image-to-video", drop_params=False
        )
        assert decode_data_url(mapped["image_url"]) == ("image/png", PNG_BYTES)

    def test_bytesio_is_read_from_the_start(self, config):
        buf = io.BytesIO(PNG_BYTES)
        buf.name = "input_image.png"
        buf.read()
        mapped = config.map_openai_params(
            {"input_reference": buf}, model="minimax/h3/image-to-video", drop_params=False
        )
        assert decode_data_url(mapped["image_url"])[1] == PNG_BYTES

    def test_raw_bytes_content_type_sniffed_from_magic(self, config):
        mapped = config.map_openai_params(
            {"input_reference": JPEG_BYTES}, model="minimax/h3/image-to-video", drop_params=False
        )
        assert decode_data_url(mapped["image_url"]) == ("image/jpeg", JPEG_BYTES)

    def test_tuple_content_type_wins(self, config):
        mapped = config.map_openai_params(
            {"input_reference": ("frame.bin", io.BytesIO(PNG_BYTES), "image/webp")},
            model="minimax/h3/image-to-video",
            drop_params=False,
        )
        assert decode_data_url(mapped["image_url"])[0] == "image/webp"

    def test_video_reference_becomes_video_data_url(self, config):
        buf = io.BytesIO(MP4_BYTES)
        buf.name = "clip.mp4"
        mapped = config.map_openai_params(
            {"input_reference": buf}, model="alibaba/happy-horse/video-edit", drop_params=False
        )
        assert decode_data_url(mapped["video_url"]) == ("video/mp4", MP4_BYTES)

    def test_unnamed_video_bytes_default_to_mp4(self, config):
        mapped = config.map_openai_params(
            {"input_reference": MP4_BYTES}, model="xai/grok-imagine-video/edit-video", drop_params=False
        )
        assert decode_data_url(mapped["video_url"])[0] == "video/mp4"

    def test_mapped_payload_is_json_serializable(self, config):
        buf = io.BytesIO(PNG_BYTES)
        buf.name = "input_image.png"
        mapped = config.map_openai_params(
            {"input_reference": buf, "seconds": "5"},
            model="alibaba/happy-horse/image-to-video",
            drop_params=False,
        )
        assert json.loads(json.dumps(mapped)) == mapped


class TestCreateRequest:
    def test_submit_url_is_the_full_model_path(self, config):
        data, files, url = config.transform_video_create_request(
            model="minimax/h3/image-to-video",
            prompt="a cat",
            api_base=FAL_QUEUE_API_BASE,
            video_create_optional_request_params={"image_url": "https://example.com/cat.png"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == "https://queue.fal.run/minimax/h3/image-to-video"
        assert data == {"prompt": "a cat", "image_url": "https://example.com/cat.png", "duration": 5}
        assert not files

    def test_submit_strips_provider_prefix(self, config):
        _, _, url = config.transform_video_create_request(
            model="fal_ai/fal-ai/ltx-video",
            prompt="a cat",
            api_base=FAL_QUEUE_API_BASE,
            video_create_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == "https://queue.fal.run/fal-ai/ltx-video"

    @pytest.mark.parametrize(
        "model,expected_duration",
        [
            ("minimax/h3/text-to-video", 5),
            ("bytedance/seedance-2.0/text-to-video", "5"),
            ("xai/grok-imagine-video/text-to-video", 6),
        ],
    )
    def test_default_duration_pinned_when_caller_gives_none(self, config, model, expected_duration):
        data, _, _ = config.transform_video_create_request(
            model=model,
            prompt="a cat",
            api_base=FAL_QUEUE_API_BASE,
            video_create_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["duration"] == expected_duration

    @pytest.mark.parametrize("model", ["fal-ai/ltx-video", "alibaba/happy-horse/video-edit"])
    def test_no_duration_for_endpoints_without_one(self, config, model):
        data, _, _ = config.transform_video_create_request(
            model=model,
            prompt="a cat",
            api_base=FAL_QUEUE_API_BASE,
            video_create_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert "duration" not in data

    def test_caller_duration_is_kept(self, config):
        data, _, _ = config.transform_video_create_request(
            model="minimax/h3/text-to-video",
            prompt="a cat",
            api_base=FAL_QUEUE_API_BASE,
            video_create_optional_request_params={"duration": 10},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["duration"] == 10


class TestCreateResponse:
    def test_id_carries_provider_model_path_and_request_id(self, config):
        video = config.transform_video_create_response(
            model="minimax/h3/image-to-video",
            raw_response=make_response({"request_id": "req-abc-123", "status": "IN_QUEUE"}),
            logging_obj=None,
            custom_llm_provider="fal_ai",
            request_data={"prompt": "a cat", "duration": 6},
        )
        assert video.id.startswith("video_")
        assert decode_video_id_with_provider(video.id) == {
            "custom_llm_provider": "fal_ai",
            "model_id": "minimax/h3/image-to-video",
            "video_id": "req-abc-123",
        }

    def test_status_and_billed_seconds_come_from_the_request(self, config):
        video = config.transform_video_create_response(
            model="minimax/h3/text-to-video",
            raw_response=make_response({"request_id": "req-1", "status": "IN_QUEUE"}),
            logging_obj=None,
            custom_llm_provider="fal_ai",
            request_data={"prompt": "x", "duration": 12, "resolution": "768P"},
        )
        assert video.status == "queued"
        assert video.model == "minimax/h3/text-to-video"
        assert video.seconds == "12"
        assert video.size == "768p"
        assert video.usage == {"duration_seconds": 12.0, "video_resolution": "768p"}

    def test_missing_duration_bills_the_family_default(self, config):
        video = config.transform_video_create_response(
            model="xai/grok-imagine-video/text-to-video",
            raw_response=make_response({"request_id": "req-2"}),
            logging_obj=None,
            custom_llm_provider="fal_ai",
            request_data={"prompt": "x"},
        )
        assert video.usage == {"duration_seconds": 6.0}

    def test_auto_duration_bills_the_family_default(self, config):
        video = config.transform_video_create_response(
            model="bytedance/seedance-2.0/text-to-video",
            raw_response=make_response({"request_id": "req-3"}),
            logging_obj=None,
            custom_llm_provider="fal_ai",
            request_data={"prompt": "x", "duration": "auto"},
        )
        assert video.usage == {"duration_seconds": 5.0}

    def test_missing_request_id_raises(self, config):
        with pytest.raises(ValueError, match="request_id"):
            config.transform_video_create_response(
                model="fal-ai/ltx-video",
                raw_response=make_response({"detail": "boom"}),
                logging_obj=None,
                custom_llm_provider="fal_ai",
                request_data={"prompt": "x"},
            )


class TestStatus:
    def test_status_url_uses_root_app_id(self, config):
        url, data = config.transform_video_status_retrieve_request(
            video_id=encoded_id(),
            api_base=FAL_QUEUE_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == STATUS_URL
        assert data == {}

    @pytest.mark.parametrize(
        "model,expected_url",
        [
            (
                "xai/grok-imagine-video/v1.5/image-to-video",
                "https://queue.fal.run/xai/grok-imagine-video/requests/r1/status",
            ),
            ("fal-ai/ltx-video", "https://queue.fal.run/fal-ai/ltx-video/requests/r1/status"),
        ],
    )
    def test_status_url_keeps_only_two_path_segments(self, config, model, expected_url):
        url, _ = config.transform_video_status_retrieve_request(
            video_id=encoded_id(model=model, request_id="r1"),
            api_base=FAL_QUEUE_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == expected_url

    def test_request_id_is_path_encoded(self, config):
        url, _ = config.transform_video_status_retrieve_request(
            video_id=encoded_id(request_id="../../other?x=1"),
            api_base=FAL_QUEUE_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == "https://queue.fal.run/minimax/h3/requests/..%2F..%2Fother%3Fx%3D1/status"

    def test_video_id_without_model_path_raises(self, config):
        with pytest.raises(ValueError, match="model path"):
            config.transform_video_status_retrieve_request(
                video_id="raw-fal-request-id",
                api_base=FAL_QUEUE_API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    @pytest.mark.parametrize(
        "fal_status,expected",
        [
            ("IN_QUEUE", "queued"),
            ("IN_PROGRESS", "in_progress"),
            ("FAILED", "failed"),
            ("SOMETHING_NEW", "failed"),
            ("", "failed"),
        ],
    )
    def test_status_mapping(self, config, fal_status, expected):
        config.transform_video_status_retrieve_request(
            video_id=encoded_id(), api_base=FAL_QUEUE_API_BASE, litellm_params=GenericLiteLLMParams(), headers={}
        )
        video = config.transform_video_status_retrieve_response(
            raw_response=make_response({"status": fal_status, "request_id": "req-abc-123"}),
            logging_obj=None,
            custom_llm_provider="fal_ai",
        )
        assert video.status == expected
        assert (video.error is not None) == (expected == "failed")

    def test_status_response_id_stays_routable(self, config):
        config.transform_video_status_retrieve_request(
            video_id=encoded_id(), api_base=FAL_QUEUE_API_BASE, litellm_params=GenericLiteLLMParams(), headers={}
        )
        video = config.transform_video_status_retrieve_response(
            raw_response=make_response({"status": "IN_PROGRESS", "request_id": "req-abc-123"}),
            logging_obj=None,
            custom_llm_provider="fal_ai",
        )
        assert decode_video_id_with_provider(video.id) == {
            "custom_llm_provider": "fal_ai",
            "model_id": "minimax/h3/image-to-video",
            "video_id": "req-abc-123",
        }


class TestCompletedResultPeek:
    """fal reports COMPLETED for failed jobs too, so a completed poll peeks at the result."""

    def poll(self, config: FalAIVideoConfig, payload: dict):
        config.transform_video_status_retrieve_request(
            video_id=encoded_id(),
            api_base=FAL_QUEUE_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={"Authorization": "Key k"},
        )
        return config.transform_video_status_retrieve_response(
            raw_response=make_response({"request_id": "req-abc-123", "response_url": RESULT_URL, **payload}),
            logging_obj=None,
            custom_llm_provider="fal_ai",
        )

    def test_failed_result_downgrades_status_to_failed(self):
        client = _RecordingClient(
            response=make_response(
                {"detail": [{"msg": "Image dimensions are too small.", "type": "image_too_small"}]}, status_code=422
            )
        )
        video = self.poll(FalAIVideoConfig(result_client=client), {"status": "COMPLETED"})
        assert video.status == "failed"
        assert "image_too_small" in video.error["message"]
        assert client.url == RESULT_URL
        assert client.headers == {"Authorization": "Key k"}

    def test_result_with_error_body_downgrades_status_to_failed(self):
        client = _RecordingClient(response=make_response({"detail": "invalid image_url"}))
        video = self.poll(FalAIVideoConfig(result_client=client), {"status": "COMPLETED"})
        assert video.status == "failed"
        assert video.error == {"code": "COMPLETED", "message": "fal.ai video generation failed: invalid image_url"}

    def test_result_with_video_stays_completed(self):
        client = _RecordingClient(response=make_response({"video": {"url": "https://fal.media/out.mp4"}}))
        video = self.poll(FalAIVideoConfig(result_client=client), {"status": "COMPLETED"})
        assert video.status == "completed"
        assert video.error is None
        assert video.completed_at is not None

    def test_peek_network_error_keeps_completed(self):
        client = _RecordingClient(exc=httpx.ConnectError("boom"))
        video = self.poll(FalAIVideoConfig(result_client=client), {"status": "COMPLETED"})
        assert video.status == "completed"

    def test_still_processing_result_keeps_completed(self):
        client = _RecordingClient(response=make_response({"status": "IN_PROGRESS"}))
        video = self.poll(FalAIVideoConfig(result_client=client), {"status": "COMPLETED"})
        assert video.status == "completed"

    def test_non_completed_statuses_never_peek(self):
        client = _RecordingClient(exc=AssertionError("peek must not run for non-completed statuses"))
        video = self.poll(FalAIVideoConfig(result_client=client), {"status": "IN_QUEUE"})
        assert video.status == "queued"


class TestContent:
    def test_content_url_uses_root_app_id(self, config):
        url, params = config.transform_video_content_request(
            video_id=encoded_id(),
            api_base=FAL_QUEUE_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == RESULT_URL
        assert params == {}

    @pytest.mark.parametrize(
        "payload,expected",
        [
            ({"video": {"url": "https://cdn.fal/v.mp4"}}, "https://cdn.fal/v.mp4"),
            ({"video": "https://cdn.fal/v1.mp4"}, "https://cdn.fal/v1.mp4"),
            ({"videos": [{"url": "https://cdn.fal/v2.mp4"}]}, "https://cdn.fal/v2.mp4"),
        ],
    )
    def test_video_url_shapes(self, config, payload, expected):
        assert config._ready_video_url(make_response(payload)) == expected

    def test_still_processing_result_raises(self, config):
        with pytest.raises(ValueError, match="still processing"):
            config._ready_video_url(make_response({"status": "IN_PROGRESS"}))

    def test_error_result_raises(self, config):
        with pytest.raises(ValueError, match="failed: invalid image_url"):
            config._ready_video_url(make_response({"detail": "invalid image_url"}))


class TestValidateEnvironment:
    def test_explicit_api_key_beats_deployment_credential(self, config):
        headers = config.validate_environment(
            headers={},
            model="fal-ai/ltx-video",
            api_key="explicit-key",
            litellm_params=GenericLiteLLMParams(api_key="credential-key"),
        )
        assert headers["Authorization"] == "Key explicit-key"

    def test_deployment_credential_beats_env_var(self, config, monkeypatch):
        monkeypatch.setenv("FAL_AI_API_KEY", "env-key")
        headers = config.validate_environment(
            headers={},
            model="fal-ai/ltx-video",
            litellm_params=GenericLiteLLMParams(api_key="credential-key"),
        )
        assert headers["Authorization"] == "Key credential-key"

    @pytest.mark.parametrize("env_var", ["FAL_AI_API_KEY", "FAL_KEY"])
    def test_env_var_fallbacks(self, config, monkeypatch, env_var):
        monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.setenv(env_var, "env-key")
        monkeypatch.setattr(litellm, "api_key", None)
        headers = config.validate_environment(headers={"X-Trace": "1"}, model="fal-ai/ltx-video")
        assert headers == {"X-Trace": "1", "Authorization": "Key env-key", "Content-Type": "application/json"}

    def test_missing_key_raises(self, config, monkeypatch):
        monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.setattr(litellm, "api_key", None)
        with pytest.raises(ValueError, match="fal.ai API key"):
            config.validate_environment(headers={}, model="fal-ai/ltx-video")

    def test_get_complete_url_defaults_to_the_queue(self, config):
        assert config.get_complete_url(model="x/y", api_base=None, litellm_params={}) == FAL_QUEUE_API_BASE
        assert config.get_complete_url(model="x/y", api_base="https://proxy.example/", litellm_params={}) == (
            "https://proxy.example"
        )

    def test_get_error_class_returns_fal_error(self, config):
        error = config.get_error_class(error_message="bad key", status_code=401, headers={})
        assert isinstance(error, FalAIVideoError)
        assert isinstance(error, BaseLLMException)
        assert error.status_code == 401


class TestSdkRoundTrip:
    """litellm.video_generation and litellm.video_status through an injected HTTP client."""

    def make_client(self, handler) -> HTTPHandler:
        return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler)))

    def test_video_generation_submits_to_the_queue_and_encodes_the_id(self):
        seen: list = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"request_id": "req-e2e-1", "status": "IN_QUEUE"})

        video = litellm.video_generation(
            model="fal_ai/minimax/h3/image-to-video",
            prompt="a cat surfing",
            seconds="6",
            api_key="test-key",
            extra_body={"image_url": "https://example.com/cat.png", "resolution": "768P"},
            client=self.make_client(handler),
        )

        (sent,) = seen
        assert str(sent.url) == "https://queue.fal.run/minimax/h3/image-to-video"
        assert sent.headers["authorization"] == "Key test-key"
        assert json.loads(sent.content) == {
            "prompt": "a cat surfing",
            "image_url": "https://example.com/cat.png",
            "duration": 6,
            "resolution": "768P",
        }
        assert decode_video_id_with_provider(video.id) == {
            "custom_llm_provider": "fal_ai",
            "model_id": "minimax/h3/image-to-video",
            "video_id": "req-e2e-1",
        }
        assert video.usage == {"duration_seconds": 6.0, "video_resolution": "768p"}

    def test_video_status_routes_by_the_encoded_id(self):
        seen: list = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"status": "IN_PROGRESS", "request_id": "req-e2e-2"})

        video = litellm.video_status(
            video_id=encode_video_id_with_provider("req-e2e-2", "fal_ai", "minimax/h3/image-to-video"),
            api_key="test-key",
            client=self.make_client(handler),
        )

        (sent,) = seen
        assert str(sent.url) == "https://queue.fal.run/minimax/h3/requests/req-e2e-2/status"
        assert sent.headers["authorization"] == "Key test-key"
        assert video.status == "in_progress"
        assert decode_video_id_with_provider(video.id)["model_id"] == "minimax/h3/image-to-video"


class TestPricing:
    @pytest.fixture(autouse=True)
    def _use_local_model_cost_map(self, monkeypatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    @pytest.mark.parametrize(
        "model,resolution,expected",
        [
            ("fal_ai/minimax/h3/text-to-video", None, 5 * 0.13),
            ("fal_ai/minimax/h3/text-to-video", "768p", 5 * 0.06),
            ("fal_ai/alibaba/happy-horse/image-to-video", "720p", 5 * 0.14),
            ("fal_ai/xai/grok-imagine-video/text-to-video", "480p", 5 * 0.05),
            ("fal_ai/fal-ai/ltx-video", None, 0.02),
        ],
    )
    def test_video_cost_uses_fal_price_map_entries(self, model, resolution, expected):
        cost = default_video_cost_calculator(
            model=model,
            duration_seconds=5.0,
            custom_llm_provider="fal_ai",
            video_resolution=resolution,
        )
        assert cost == pytest.approx(expected)
