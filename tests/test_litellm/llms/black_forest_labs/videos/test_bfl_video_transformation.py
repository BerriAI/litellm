"""
Tests for Black Forest Labs FLUX 3 video generation transformation.

Payload and response shapes are taken from a live FLUX 3 video generation
against https://api.bfl.ai/v1/flux-3-video and its regional polling URL.
"""

from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.black_forest_labs.common_utils import BlackForestLabsError
from litellm.llms.black_forest_labs.videos.transformation import (
    BlackForestLabsVideoConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import extract_original_video_id

API_BASE = "https://api.bfl.ai"
JOB_ID = "90307d3a-deec-47cb-bdb9-bf1c5bae1f04"
POLLING_URL = f"https://api.us7.bfl.ai/v1/get_result?id={JOB_ID}"
SAMPLE_URL = "https://delivery.us7.bfl.ai/durable/2026081720/video.mp4?se=2026-08-17T21%3A26%3A38Z&sig=abc"


def _response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("POST", f"{API_BASE}/v1/flux-3-video"),
    )


class TestBlackForestLabsVideoTransformation:
    def setup_method(self):
        self.config = BlackForestLabsVideoConfig()
        self.mock_logging_obj = Mock()

    def test_text_to_video_request_sets_t2v_mode(self):
        data, files, url = self.config.transform_video_create_request(
            model="flux-3-video",
            prompt="A white kitten chases a butterfly across a sunlit garden.",
            api_base=API_BASE,
            video_create_optional_request_params={
                "duration": 8,
                "resolution": "fhd",
                "aspect_ratio": "16:9",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://api.bfl.ai/v1/flux-3-video"
        assert data["mode"] == "t2v"
        assert data["prompt"] == "A white kitten chases a butterfly across a sunlit garden."
        assert data["duration"] == 8
        assert data["resolution"] == "fhd"
        assert files == []

    @pytest.mark.parametrize(
        "params, expected_mode",
        [
            ({}, "t2v"),
            ({"keyframes": ["https://example.com/first.png"]}, "i2v"),
            ({"start_video": "https://example.com/clip.mp4"}, "v2v"),
            ({"draft_cache": "https://example.com/bundle.bin"}, "draft_enhance"),
        ],
    )
    def test_mode_is_inferred_from_inputs(self, params, expected_mode):
        data, _, _ = self.config.transform_video_create_request(
            model="flux-3-video",
            prompt="a prompt",
            api_base=API_BASE,
            video_create_optional_request_params=dict(params),
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["mode"] == expected_mode

    def test_draft_enhance_drops_the_prompt(self):
        """FLUX 3 rejects a prompt in draft_enhance mode; the bundle carries it."""
        data, _, _ = self.config.transform_video_create_request(
            model="flux-3-video",
            prompt="a prompt the API would reject here",
            api_base=API_BASE,
            video_create_optional_request_params={"draft_cache": "bundle"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["mode"] == "draft_enhance"
        assert "prompt" not in data

    def test_explicit_mode_is_not_overridden(self):
        data, _, _ = self.config.transform_video_create_request(
            model="flux-3-video",
            prompt="a prompt",
            api_base=API_BASE,
            video_create_optional_request_params={
                "mode": "i2v",
                "keyframes": ["https://example.com/first.png"],
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["mode"] == "i2v"

    def test_unknown_model_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown BFL video model"):
            self.config.transform_video_create_request(
                model="flux-9-video",
                prompt="a prompt",
                api_base=API_BASE,
                video_create_optional_request_params={},
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    @pytest.mark.parametrize(
        "size, expected_resolution",
        [
            ("1280x720", "hd"),
            ("1920x1080", "fhd"),
            ("1080x1920", "fhd"),
            ("720x1280", "hd"),
            ("fhd", "fhd"),
        ],
    )
    def test_size_maps_to_a_resolution_tier(self, size, expected_resolution):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"size": size},
            model="flux-3-video",
            drop_params=False,
        )

        assert mapped["resolution"] == expected_resolution

    @pytest.mark.parametrize(
        "seconds, expected_duration",
        [("8", 8), (8, 8), ("2", 5), ("60", 20), ("8.6", 8)],
    )
    def test_seconds_maps_into_the_supported_duration_range(self, seconds, expected_duration):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"seconds": seconds},
            model="flux-3-video",
            drop_params=False,
        )

        assert mapped["duration"] == expected_duration

    def test_input_reference_becomes_a_keyframe(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"input_reference": "https://example.com/first.png"},
            model="flux-3-video",
            drop_params=False,
        )

        assert mapped["keyframes"] == ["https://example.com/first.png"]

    def test_create_response_returns_the_job_handle_and_keeps_the_region(self):
        video = self.config.transform_video_create_response(
            model="flux-3-video",
            raw_response=_response({"id": JOB_ID, "polling_url": POLLING_URL, "cost": None}),
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="black_forest_labs",
            request_data={"duration": 8, "resolution": "fhd"},
        )

        assert video.status == "queued"
        assert video.seconds == "8"
        assert video.size == "fhd"

        # The region survives inside the id, which is all the status call gets.
        status_url, _ = self.config.transform_video_status_retrieve_request(
            video_id=video.id,
            api_base=API_BASE,
            litellm_params=None,
            headers={},
        )
        assert status_url == POLLING_URL

    def test_status_url_falls_back_to_the_api_base_without_a_region(self):
        video = self.config.transform_video_create_response(
            model="flux-3-video",
            raw_response=_response({"id": JOB_ID, "polling_url": None}),
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="black_forest_labs",
        )

        assert extract_original_video_id(video.id) == JOB_ID

        status_url, _ = self.config.transform_video_status_retrieve_request(
            video_id=video.id,
            api_base=API_BASE,
            litellm_params=None,
            headers={},
        )
        assert status_url == f"{API_BASE}/v1/get_result?id={JOB_ID}"

    def test_content_request_targets_the_same_region_as_the_status_call(self):
        video = self.config.transform_video_create_response(
            model="flux-3-video",
            raw_response=_response({"id": JOB_ID, "polling_url": POLLING_URL}),
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="black_forest_labs",
        )

        content_url, _ = self.config.transform_video_content_request(
            video_id=video.id,
            api_base=API_BASE,
            litellm_params=None,
            headers={},
        )
        assert content_url == POLLING_URL

    def test_a_packed_region_outside_bfl_is_rejected(self):
        with pytest.raises(BlackForestLabsError, match="not within the bfl.ai domain"):
            self.config.transform_video_status_retrieve_request(
                video_id=f"{JOB_ID}@attacker.example.com",
                api_base=API_BASE,
                litellm_params=None,
                headers={},
            )

    def test_create_response_without_a_job_id_raises(self):
        with pytest.raises(BlackForestLabsError):
            self.config.transform_video_create_response(
                model="flux-3-video",
                raw_response=_response({"detail": "bad request"}, status_code=422),
                logging_obj=self.mock_logging_obj,
            )

    def test_create_response_rejects_a_polling_url_outside_bfl(self):
        with pytest.raises(BlackForestLabsError, match="not within the bfl.ai domain"):
            self.config.transform_video_create_response(
                model="flux-3-video",
                raw_response=_response(
                    {"id": JOB_ID, "polling_url": "https://attacker.example.com/v1/get_result"}
                ),
                logging_obj=self.mock_logging_obj,
            )

    @pytest.mark.parametrize(
        "bfl_status, expected_status",
        [
            ("Pending", "queued"),
            ("Queued", "queued"),
            ("Reasoning", "in_progress"),
            ("Generating", "in_progress"),
            ("Ready", "completed"),
            ("Error", "failed"),
            ("Content Moderated", "failed"),
            ("Task not found", "failed"),
        ],
    )
    def test_bfl_status_maps_to_openai_status(self, bfl_status, expected_status):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response({"id": JOB_ID, "status": bfl_status, "result": {}}),
            logging_obj=self.mock_logging_obj,
        )

        assert video.status == expected_status

    def test_failed_status_carries_the_bfl_detail(self):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response(
                {"id": JOB_ID, "status": "Content Moderated", "details": "flagged by moderation"}
            ),
            logging_obj=self.mock_logging_obj,
        )

        assert video.status == "failed"
        assert video.error["code"] == "Content Moderated"
        assert video.error["message"] == "flagged by moderation"

    def test_completed_status_has_no_error(self):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response({"id": JOB_ID, "status": "Ready", "result": {"sample": SAMPLE_URL}}),
            logging_obj=self.mock_logging_obj,
        )

        assert video.error is None

    def test_credit_cost_is_reported_as_usage(self):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response({"id": JOB_ID, "status": "Ready", "cost": 30.0}),
            logging_obj=self.mock_logging_obj,
        )

        assert video.usage == {"credits": 30.0}

    def test_fractional_progress_is_reported_as_a_percentage(self):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response({"id": JOB_ID, "status": "Generating", "progress": 0.42}),
            logging_obj=self.mock_logging_obj,
        )

        assert video.progress == 42

    def test_status_request_targets_get_result_with_the_original_job_id(self):
        encoded_id = self.config.transform_video_create_response(
            model="flux-3-video",
            raw_response=_response({"id": JOB_ID, "polling_url": POLLING_URL}),
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="black_forest_labs",
        ).id

        url, params = self.config.transform_video_status_retrieve_request(
            video_id=encoded_id,
            api_base="https://api.us7.bfl.ai",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == POLLING_URL
        assert params == {}

    def test_content_request_targets_get_result(self):
        url, _ = self.config.transform_video_content_request(
            video_id=JOB_ID,
            api_base="https://api.us7.bfl.ai",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == POLLING_URL

    def test_content_response_raises_while_still_generating(self):
        with pytest.raises(BlackForestLabsError, match="still processing"):
            self.config.transform_video_content_response(
                raw_response=_response({"id": JOB_ID, "status": "Generating", "result": {}}),
                logging_obj=self.mock_logging_obj,
            )

    def test_content_response_raises_when_a_terminal_job_has_no_video(self):
        with pytest.raises(BlackForestLabsError, match="did not produce a video"):
            self.config.transform_video_content_response(
                raw_response=_response({"id": JOB_ID, "status": "Error", "result": {}}),
                logging_obj=self.mock_logging_obj,
            )

    def test_validate_environment_sets_the_x_key_header(self):
        headers = self.config.validate_environment(
            headers={},
            model="flux-3-video",
            api_key="test-key",
        )

        assert headers["x-key"] == "test-key"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_without_a_key_raises(self, monkeypatch):
        monkeypatch.delenv("BFL_API_KEY", raising=False)
        monkeypatch.delenv("BLACK_FOREST_LABS_API_KEY", raising=False)
        monkeypatch.setattr("litellm.api_key", None)

        with pytest.raises(BlackForestLabsError, match="BFL_API_KEY is not set"):
            self.config.validate_environment(headers={}, model="flux-3-video")
