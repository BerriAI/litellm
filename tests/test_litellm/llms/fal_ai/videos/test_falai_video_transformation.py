from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.fal_ai.videos.transformation import FalAIVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)

SORA_2_MODEL = "fal_ai/fal-ai/sora-2/text-to-video"
KLING_MODEL = "fal_ai/fal-ai/kling-video/v2.5-turbo/pro/text-to-video"
KLING_MODEL_ID = "fal-ai/kling-video/v2.5-turbo/pro/text-to-video"
KLING_QUEUE_NAMESPACE = "fal-ai/kling-video"
FAL_API_BASE = "https://queue.fal.run"


def _fal_status_response(payload, request_id="abc-123", status_code=200):
    request = httpx.Request(
        "GET", f"{FAL_API_BASE}/{KLING_MODEL_ID}/requests/{request_id}/status"
    )
    return httpx.Response(status_code, json=payload, request=request)


class TestFalAIVideoTransformation:
    def setup_method(self):
        self.config = FalAIVideoConfig()
        self.mock_logging_obj = Mock()

    def test_validate_environment_uses_fal_ai_api_key(self, monkeypatch):
        monkeypatch.setenv("FAL_AI_API_KEY", "test-key-123")
        headers = self.config.validate_environment(
            headers={},
            model=SORA_2_MODEL,
        )
        assert headers["Authorization"] == "Key test-key-123"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_falls_back_to_fal_key(self, monkeypatch):
        monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
        monkeypatch.setenv("FAL_KEY", "fallback-key")
        headers = self.config.validate_environment(headers={}, model=SORA_2_MODEL)
        assert headers["Authorization"] == "Key fallback-key"

    def test_validate_environment_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
        monkeypatch.delenv("FAL_KEY", raising=False)
        with pytest.raises(ValueError, match="fal.ai API key is required"):
            self.config.validate_environment(headers={}, model=SORA_2_MODEL)

    def test_get_complete_url_uses_default_base(self, monkeypatch):
        monkeypatch.delenv("FAL_AI_API_BASE", raising=False)
        url = self.config.get_complete_url(
            model=SORA_2_MODEL, api_base=None, litellm_params={}
        )
        assert url == FAL_API_BASE

    def test_get_complete_url_strips_trailing_slash(self):
        url = self.config.get_complete_url(
            model=SORA_2_MODEL,
            api_base="https://custom.example.com/",
            litellm_params={},
        )
        assert url == "https://custom.example.com"

    def test_map_openai_params_converts_seconds_and_size(self):
        params = self.config.map_openai_params(
            video_create_optional_params={"seconds": 5, "size": "1280x720"},
            model=KLING_MODEL,
            drop_params=False,
        )
        assert params["duration"] == "5"
        assert params["aspect_ratio"] == "16:9"

    def test_map_openai_params_falls_back_to_colon_replacement(self):
        params = self.config.map_openai_params(
            video_create_optional_params={"size": "640x480"},
            model=KLING_MODEL,
            drop_params=False,
        )
        assert params["aspect_ratio"] == "640:480"

    def test_map_openai_params_unpacks_extra_body(self):
        params = self.config.map_openai_params(
            video_create_optional_params={
                "extra_body": {"negative_prompt": "blurry", "cfg_scale": 0.5}
            },
            model=KLING_MODEL,
            drop_params=False,
        )
        assert params["negative_prompt"] == "blurry"
        assert params["cfg_scale"] == 0.5
        assert "extra_body" not in params

    def test_transform_video_create_request_builds_queue_url(self):
        data, files, url = self.config.transform_video_create_request(
            model=KLING_MODEL,
            prompt="A demo video",
            api_base=FAL_API_BASE,
            video_create_optional_request_params={
                "duration": "5",
                "aspect_ratio": "16:9",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == f"{FAL_API_BASE}/{KLING_MODEL_ID}"
        assert data["prompt"] == "A demo video"
        assert data["duration"] == "5"
        assert data["aspect_ratio"] == "16:9"
        assert "model" not in data
        assert files == []

    def test_transform_video_create_response_encodes_model_into_video_id(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "request_id": "abc-123",
            "status": "IN_QUEUE",
        }

        video_obj = self.config.transform_video_create_response(
            model=KLING_MODEL,
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="fal_ai",
            request_data={"duration": "5", "aspect_ratio": "16:9"},
        )

        assert isinstance(video_obj, VideoObject)
        assert video_obj.status == "queued"
        assert video_obj.id.startswith("video_")

        decoded = decode_video_id_with_provider(video_obj.id)
        assert decoded.get("video_id") == "abc-123"
        assert decoded.get("custom_llm_provider") == "fal_ai"
        assert decoded.get("model_id") == KLING_MODEL_ID

        assert video_obj.seconds == "5"
        assert video_obj.size == "16x9"

    def test_transform_video_status_retrieve_request_builds_status_url(self):
        encoded_id = encode_video_id_with_provider("abc-123", "fal_ai", KLING_MODEL_ID)
        url, params = self.config.transform_video_status_retrieve_request(
            video_id=encoded_id,
            api_base=FAL_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == f"{FAL_API_BASE}/{KLING_QUEUE_NAMESPACE}/requests/abc-123/status"
        assert params == {}

    def test_transform_video_status_retrieve_request_reconstructs_from_model_id(self):
        encoded_id = encode_video_id_with_provider("abc-123", "fal_ai", KLING_MODEL_ID)
        url, params = self.config.transform_video_status_retrieve_request(
            video_id=encoded_id,
            api_base="https://attacker.example.com",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert (
            url
            == f"https://attacker.example.com/{KLING_QUEUE_NAMESPACE}/requests/abc-123/status"
        )
        assert params == {}

    def test_status_and_content_urls_use_owner_app_namespace(self):
        seedance_id = "fal-ai/bytedance/seedance/v2/pro/text-to-video"
        encoded_id = encode_video_id_with_provider("abc-123", "fal_ai", seedance_id)

        status_url, _ = self.config.transform_video_status_retrieve_request(
            video_id=encoded_id,
            api_base=FAL_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        content_url, _ = self.config.transform_video_content_request(
            video_id=encoded_id,
            api_base=FAL_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert status_url == f"{FAL_API_BASE}/fal-ai/bytedance/requests/abc-123/status"
        assert content_url == f"{FAL_API_BASE}/fal-ai/bytedance/requests/abc-123"

    def test_queue_namespace_keeps_two_segment_model_ids(self):
        encoded_id = encode_video_id_with_provider("abc-123", "fal_ai", "fal-ai/sora-2")
        url, _ = self.config.transform_video_status_retrieve_request(
            video_id=encoded_id,
            api_base=FAL_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{FAL_API_BASE}/fal-ai/sora-2/requests/abc-123/status"

    def test_transform_video_status_request_url_path_segment_is_encoded(self):
        encoded_id = encode_video_id_with_provider(
            "../../../etc/passwd", "fal_ai", KLING_MODEL_ID
        )
        url, _ = self.config.transform_video_status_retrieve_request(
            video_id=encoded_id,
            api_base=FAL_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert "/requests/..%2F..%2F..%2Fetc%2Fpasswd/status" in url

    def test_transform_video_status_response_maps_in_progress(self):
        mock_response = _fal_status_response(
            {
                "request_id": "abc-123",
                "status": "IN_PROGRESS",
                "queue_position": 2,
            }
        )
        status_obj = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="fal_ai",
        )
        assert status_obj.status == "in_progress"
        assert status_obj.progress == 2

    def test_transform_video_status_response_maps_failed_with_error(self):
        mock_response = _fal_status_response(
            {
                "request_id": "abc-123",
                "status": "FAILED",
                "error": "model timed out",
            }
        )
        status_obj = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="fal_ai",
        )
        assert status_obj.status == "failed"
        assert status_obj.error is not None
        assert status_obj.error["message"] == "model timed out"

    def test_transform_video_status_response_tolerates_non_json_body(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.side_effect = ValueError(
            "Expecting value: line 1 column 1 (char 0)"
        )

        status_obj = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="fal_ai",
        )

        assert status_obj.status == "in_progress"

    def test_transform_video_content_request_builds_result_url(self):
        encoded_id = encode_video_id_with_provider("abc-123", "fal_ai", KLING_MODEL_ID)
        url, params = self.config.transform_video_content_request(
            video_id=encoded_id,
            api_base=FAL_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{FAL_API_BASE}/{KLING_QUEUE_NAMESPACE}/requests/abc-123"
        assert params == {}

    def test_transform_video_content_request_reconstructs_from_model_id(self):
        encoded_id = encode_video_id_with_provider("abc-123", "fal_ai", KLING_MODEL_ID)
        url, params = self.config.transform_video_content_request(
            video_id=encoded_id,
            api_base="https://attacker.example.com",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert (
            url
            == f"https://attacker.example.com/{KLING_QUEUE_NAMESPACE}/requests/abc-123"
        )
        assert params == {}

    def test_extract_video_url_handles_video_object(self):
        url = self.config._extract_video_url(
            {"video": {"url": "https://cdn.example.com/v.mp4"}}
        )
        assert url == "https://cdn.example.com/v.mp4"

    def test_extract_video_url_handles_top_level_url(self):
        url = self.config._extract_video_url({"url": "https://cdn.example.com/v.mp4"})
        assert url == "https://cdn.example.com/v.mp4"

    def test_extract_video_url_raises_when_missing(self):
        with pytest.raises(ValueError, match="Video URL not found"):
            self.config._extract_video_url({"status": "IN_PROGRESS"})

    def test_status_request_requires_model_id_in_video_id(self):
        plain_id = encode_video_id_with_provider("abc-123", "fal_ai", None)
        with pytest.raises(ValueError, match="model id encoded"):
            self.config.transform_video_status_retrieve_request(
                video_id=plain_id,
                api_base=FAL_API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_transform_video_delete_request_raises_not_implemented(self):
        encoded_id = encode_video_id_with_provider("abc-123", "fal_ai", KLING_MODEL_ID)
        with pytest.raises(NotImplementedError, match="delete/cancel is not supported"):
            self.config.transform_video_delete_request(
                video_id=encoded_id,
                api_base=FAL_API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_transform_video_delete_response_raises_not_implemented(self):
        mock_response = Mock(spec=httpx.Response)
        with pytest.raises(NotImplementedError, match="delete/cancel is not supported"):
            self.config.transform_video_delete_response(
                raw_response=mock_response,
                logging_obj=self.mock_logging_obj,
            )

    def test_remix_and_list_raise_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.config.transform_video_remix_request(
                video_id="x",
                prompt="p",
                api_base=FAL_API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )
        with pytest.raises(NotImplementedError):
            self.config.transform_video_list_request(
                api_base=FAL_API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_full_video_workflow(self):
        config = FalAIVideoConfig()
        mock_logging_obj = Mock()

        data, _, url = config.transform_video_create_request(
            model=KLING_MODEL,
            prompt="A high quality demo of LiteLLM video gateway",
            api_base=FAL_API_BASE,
            video_create_optional_request_params={
                "duration": "5",
                "aspect_ratio": "16:9",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url.endswith(KLING_MODEL_ID)

        create_response = Mock(spec=httpx.Response)
        create_response.json.return_value = {
            "request_id": "queued-id-1",
            "status": "IN_QUEUE",
        }
        video_obj = config.transform_video_create_response(
            model=KLING_MODEL,
            raw_response=create_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="fal_ai",
            request_data=data,
        )
        assert video_obj.status == "queued"
        assert video_obj.id.startswith("video_")

        status_url, _ = config.transform_video_status_retrieve_request(
            video_id=video_obj.id,
            api_base=FAL_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert status_url.endswith("/requests/queued-id-1/status")

        completed_response = _fal_status_response(
            {
                "request_id": "queued-id-1",
                "status": "COMPLETED",
            },
            request_id="queued-id-1",
        )
        completed_obj = config.transform_video_status_retrieve_response(
            raw_response=completed_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="fal_ai",
        )
        assert completed_obj.status == "completed"


def test_provider_config_manager_returns_fal_ai_video_config():
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_video_config(
        model=SORA_2_MODEL, provider=LlmProviders.FAL_AI
    )
    assert isinstance(config, FalAIVideoConfig)


@pytest.mark.parametrize(
    "model_id,expected_modalities",
    [
        ("fal_ai/fal-ai/kling-video/v3/standard/text-to-video", ("text",)),
        ("fal_ai/fal-ai/kling-video/v3/pro/text-to-video", ("text",)),
        ("fal_ai/bytedance/seedance-2.0/text-to-video", ("text",)),
        ("fal_ai/fal-ai/veo3.1/fast", ("text",)),
        ("fal_ai/fal-ai/kling-video/v3/standard/image-to-video", ("text", "image")),
        ("fal_ai/fal-ai/kling-video/v3/pro/image-to-video", ("text", "image")),
        ("fal_ai/bytedance/seedance-2.0/image-to-video", ("text", "image")),
    ],
)
def test_fal_ai_video_model_registered_with_video_endpoint(
    model_id: str, expected_modalities: tuple
):
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    backup = GetModelCostMap.load_local_model_cost_map()
    entry = backup.get(model_id)
    assert entry is not None, f"{model_id} missing from local backup model cost map"
    assert entry["litellm_provider"] == "fal_ai"
    assert entry["mode"] == "video_generation"
    assert "/v1/videos" in entry["supported_endpoints"]
    assert tuple(entry["supported_modalities"]) == expected_modalities
    assert entry["supported_output_modalities"] == ["video"]
    assert isinstance(entry["output_cost_per_video_per_second"], (int, float))
