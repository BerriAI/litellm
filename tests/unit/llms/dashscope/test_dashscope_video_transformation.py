"""
Tests for DashScope (Wan 3.0 / Wan 2.7 / HappyHorse) video generation transformation.
"""

import base64
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final
from unittest.mock import Mock

import httpx
import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.dashscope.qwen_ai_platform import QwenAIPlatformVideoConfig
from litellm.llms.dashscope.qwencloud import QwenCloudVideoConfig
from litellm.llms.dashscope.videos.transformation import (
    DashScopeVideoConfig,
    DashScopeVideoError,
    _parse_task_response,
    _video_url_from_task,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)
from litellm.videos.main import avideo_generation

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

API_BASE = "https://dashscope.aliyuncs.com"
SYNTHESIS_URL = f"{API_BASE}/api/v1/services/aigc/video-generation/video-synthesis"


def _mock_response(payload: dict, status_code: int = 200) -> Mock:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = payload
    mock_response.status_code = status_code
    mock_response.headers = httpx.Headers()
    return mock_response


def _write_png(tmp_path: Path) -> Path:
    path: Final = tmp_path / "first.png"
    path.write_bytes(PNG_BYTES)
    return path


def _create(params: dict, model: str = "wan3.0-video", prompt: str = "a cat on a roof"):
    return DashScopeVideoConfig().transform_video_create_request(
        model=model,
        prompt=prompt,
        api_base=API_BASE,
        video_create_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )


class TestDashScopeVideoCreateRequest:
    def test_body_is_nested_into_input_and_parameters(self):
        """DashScope rejects a flat Sora-shaped body: prompt and media belong
        under ``input``, everything else under ``parameters``."""
        data, files, url = _create({"resolution": "720P", "duration": 10, "ratio": "16:9"})

        assert url == SYNTHESIS_URL
        assert files == ()
        assert data == {
            "model": "wan3.0-video",
            "input": {"prompt": "a cat on a roof"},
            "parameters": {"resolution": "720P", "ratio": "16:9", "duration": 10},
        }

    def test_prompt_only_request_omits_parameters_block(self):
        """With nothing to configure, DashScope's documented text-to-video body
        is just model + input, so no empty parameters block is sent."""
        data, _, _ = _create({})

        assert data == {"model": "wan3.0-video", "input": {"prompt": "a cat on a roof"}}

    def test_media_array_passes_through_into_input(self):
        media = [
            {"type": "reference_image", "url": "https://x/1.png"},
            {"type": "reference_audio", "url": "https://x/1.mp3"},
        ]
        data, _, _ = _create({"media": media, "duration": 5})

        assert data["input"]["media"] == media
        assert "media" not in data["parameters"]

    def test_negative_prompt_goes_to_input_not_parameters(self):
        data, _, _ = _create({"negative_prompt": "blurry", "seed": 7})

        assert data["input"]["negative_prompt"] == "blurry"
        assert data["parameters"] == {"seed": 7}


class TestDashScopeVideoInputReference:
    @pytest.mark.parametrize(
        "make_reference",
        [
            lambda tmp_path: ("first.png", PNG_BYTES, "image/png"),
            lambda tmp_path: ("first.png", io.BytesIO(PNG_BYTES)),
            lambda tmp_path: _write_png(tmp_path),
        ],
        ids=["filename-bytes-content-type", "filename-file", "path"],
    )
    def test_file_tuple_and_path_references_are_encoded(self, make_reference, tmp_path):
        """Regression: tuples and paths are valid FileTypes but raised before reaching DashScope."""
        data, _, _ = _create({"input_reference": make_reference(tmp_path)})

        url = data["input"]["media"][0]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == PNG_BYTES

    def test_string_inside_a_file_tuple_is_not_read_as_a_path(self, tmp_path):
        path = _write_png(tmp_path)

        with pytest.raises(ValueError, match="input_reference must be"):
            _create({"input_reference": ("first.png", str(path))})

    def test_wan3_file_reference_becomes_first_frame_media_data_uri(self):
        """Wan 3.0 takes references as typed media entries; a file must arrive
        base64-encoded as a data URI rather than as an unusable file object."""
        data, _, _ = _create({"input_reference": io.BytesIO(PNG_BYTES)})

        media = data["input"]["media"]
        assert len(media) == 1
        assert media[0]["type"] == "first_frame"
        assert media[0]["url"].startswith("data:image/png;base64,")
        assert base64.b64decode(media[0]["url"].split(",", 1)[1]) == PNG_BYTES

    def test_wan3_url_reference_is_passed_through_unencoded(self):
        data, _, _ = _create({"input_reference": "https://x/first.png"})

        assert data["input"]["media"] == ({"type": "first_frame", "url": "https://x/first.png"},)

    @pytest.mark.parametrize(
        "model", ["wan2.7-i2v", "wan2.7-i2v-2026-04-25", "happyhorse-1.1-i2v", "happyhorse-1.0-i2v"]
    )
    def test_current_i2v_models_take_the_reference_as_a_first_frame_media_entry(self, model):
        """Wan 2.7 and HappyHorse image-to-video take the typed media array;
        a flat img_url is only understood by the legacy Wan 2.6 models."""
        data, _, _ = _create({"input_reference": "https://x/first.png"}, model=model)

        assert data["input"]["media"] == ({"type": "first_frame", "url": "https://x/first.png"},)
        assert "img_url" not in data["input"]

    @pytest.mark.parametrize("model", ["happyhorse-1.1-r2v", "happyhorse-1.0-r2v", "wan2.7-r2v"])
    def test_reference_to_video_models_take_the_reference_as_a_reference_image(self, model):
        """The r2v models reject first_frame and only accept reference_image."""
        data, _, _ = _create({"input_reference": "https://x/subject.png"}, model=model)

        assert data["input"]["media"] == ({"type": "reference_image", "url": "https://x/subject.png"},)

    @pytest.mark.parametrize(
        "model", ["wan2.6-i2v-flash", "wan2.5-i2v-preview", "wan2.2-i2v-plus", "wanx2.1-i2v-turbo"]
    )
    def test_legacy_models_take_the_reference_on_the_flat_img_url_field(self, model):
        """Wan 2.6 and earlier predate the media array and would reject it."""
        data, _, _ = _create({"input_reference": "https://x/first.png"}, model=model)

        assert data["input"]["img_url"] == "https://x/first.png"
        assert "media" not in data["input"]

    def test_explicit_media_array_wins_over_input_reference(self):
        """Only the full media array can express multi-role references, so a
        caller-supplied one must not be clobbered by input_reference."""
        media = [{"type": "reference_video", "url": "https://x/v.mp4"}]
        data, _, _ = _create({"media": media, "input_reference": "https://x/first.png"})

        assert data["input"]["media"] == media

    def test_legacy_explicit_img_url_wins_over_input_reference(self):
        data, _, _ = _create(
            {"img_url": "https://x/explicit.png", "input_reference": "https://x/other.png"},
            model="wan2.6-i2v-flash",
        )

        assert data["input"]["img_url"] == "https://x/explicit.png"


class TestDashScopeVideoMapOpenAIParams:
    def test_seconds_becomes_duration(self):
        mapped = DashScopeVideoConfig().map_openai_params(
            video_create_optional_params={"seconds": "10"}, model="wan3.0-video", drop_params=False
        )

        assert mapped == {"duration": 10}
        assert "seconds" not in mapped

    def test_size_becomes_both_ratio_and_resolution_tier(self):
        """DashScope takes a ratio plus a named resolution tier, and bills per
        second by that tier, so a pixel size must produce both or the request
        silently falls back to the pricier 1080P default."""
        mapped = DashScopeVideoConfig().map_openai_params(
            video_create_optional_params={"size": "1280x720"}, model="wan3.0-video", drop_params=False
        )

        assert mapped == {"ratio": "16:9", "resolution": "720P"}

    @pytest.mark.parametrize(
        "size,expected_resolution",
        [
            ("854x480", "480P"),
            ("1280x720", "720P"),
            ("1920x1080", "1080P"),
            ("1080x1920", "1080P"),
            ("3840x2160", "1080P"),
        ],
    )
    def test_resolution_tier_is_picked_from_the_shortest_side(self, size, expected_resolution):
        mapped = DashScopeVideoConfig().map_openai_params(
            video_create_optional_params={"size": size}, model="wan3.0-video", drop_params=False
        )

        assert mapped["resolution"] == expected_resolution

    def test_explicit_resolution_and_ratio_win_over_size(self):
        mapped = DashScopeVideoConfig().map_openai_params(
            video_create_optional_params={"size": "1280x720", "ratio": "4:3", "resolution": "1080P"},
            model="wan3.0-video",
            drop_params=False,
        )

        assert mapped["ratio"] == "4:3"
        assert mapped["resolution"] == "1080P"

    def test_request_metadata_is_never_sent_as_a_body_field(self):
        mapped = DashScopeVideoConfig().map_openai_params(
            video_create_optional_params={
                "user": "u1",
                "prompt": "p",
                "model": "wan3.0-video",
                "extra_body": {"seed": 1},
            },
            model="wan3.0-video",
            drop_params=False,
        )

        assert mapped == {}

    @pytest.mark.parametrize("params", [{"characters": [{"id": "c"}]}, {"image": {"gcsUri": "gs://b/i.png"}}])
    def test_fields_dashscope_cannot_express_raise_unless_dropped(self, params):
        with pytest.raises(UnsupportedParamsError, match="drop_params=True"):
            DashScopeVideoConfig().map_openai_params(
                video_create_optional_params=params, model="wan3.0-video", drop_params=False
            )

        assert (
            DashScopeVideoConfig().map_openai_params(
                video_create_optional_params=params, model="wan3.0-video", drop_params=True
            )
            == {}
        )

    def test_parameters_block_is_merged(self):
        mapped = DashScopeVideoConfig().map_openai_params(
            video_create_optional_params={"parameters": {"prompt_extend": True, "audio": False}},
            model="wan3.0-video",
            drop_params=False,
        )

        assert mapped == {"prompt_extend": True, "audio": False}

    def test_parameters_block_must_be_an_object(self):
        with pytest.raises(ValueError, match="parameters must be an object"):
            DashScopeVideoConfig().map_openai_params(
                video_create_optional_params={"parameters": ["not", "an", "object"]},
                model="wan3.0-video",
                drop_params=False,
            )

    @pytest.mark.parametrize("params", [{"seconds": "-1"}, {"parameters": {"duration": -1}}])
    def test_smart_duration_is_rejected_because_it_cannot_be_billed(self, params):
        """duration -1 lets DashScope pick the length after the create call,
        which is the only call litellm bills, so it would record $0 for a
        video DashScope charges for."""
        mapped = DashScopeVideoConfig().map_openai_params(
            video_create_optional_params=params, model="wan3.0-video", drop_params=False
        )

        with pytest.raises(UnsupportedParamsError, match="explicit duration") as raised:
            _create(mapped)

        assert raised.value.status_code == 400


class TestDashScopeVideoCreateResponse:
    def test_task_id_is_encoded_with_provider_and_model(self):
        """The create response only carries task_id, so litellm must wrap it for
        later status/content calls to route back to dashscope."""
        video_obj = DashScopeVideoConfig().transform_video_create_response(
            model="wan3.0-video",
            raw_response=_mock_response(
                {"output": {"task_status": "PENDING", "task_id": "0385dc79-5ff8"}, "request_id": "r1"}
            ),
            logging_obj=None,
            custom_llm_provider="dashscope",
            request_data={"model": "wan3.0-video", "parameters": {"duration": 10, "resolution": "720P"}},
        )

        assert video_obj.status == "queued"
        assert video_obj.model == "wan3.0-video"
        decoded = decode_video_id_with_provider(video_obj.id)
        assert decoded["custom_llm_provider"] == "dashscope"
        assert decoded["model_id"] == "wan3.0-video"
        assert decoded["video_id"] == "0385dc79-5ff8"

    def test_usage_carries_requested_cost_inputs_while_queued(self):
        video_obj = DashScopeVideoConfig().transform_video_create_response(
            model="wan3.0-video",
            raw_response=_mock_response({"output": {"task_status": "PENDING", "task_id": "t1"}}),
            logging_obj=None,
            custom_llm_provider="dashscope",
            request_data={"parameters": {"duration": 4, "resolution": "480P"}},
        )

        assert video_obj.usage == {"duration_seconds": 4.0, "video_resolution": "480p"}

    @pytest.mark.parametrize(
        "parameters,expected_usage",
        [
            (None, {"duration_seconds": 5.0, "video_resolution": "1080p"}),
            ({"resolution": "480P"}, {"duration_seconds": 5.0, "video_resolution": "480p"}),
            ({"duration": 8}, {"duration_seconds": 8.0, "video_resolution": "1080p"}),
        ],
    )
    def test_omitted_duration_and_tier_bill_dashscope_defaults(self, parameters, expected_usage):
        """DashScope fills an omitted duration (5s) and resolution (1080P) and
        bills them; the create call is the billed one, so leaving either out
        recorded the video at $0."""
        video_obj = DashScopeVideoConfig().transform_video_create_response(
            model="wan3.0-video",
            raw_response=_mock_response({"output": {"task_status": "PENDING", "task_id": "t1"}}),
            logging_obj=None,
            custom_llm_provider="dashscope",
            request_data={"model": "wan3.0-video", **({"parameters": parameters} if parameters else {})},
        )

        assert video_obj.usage == expected_usage

    def test_in_body_error_on_a_200_is_raised(self):
        """DashScope reports create failures as a 200 with top-level code and no
        output; without this the caller gets a queued video with an empty id."""
        with pytest.raises(DashScopeVideoError, match="No API-key provided") as raised:
            DashScopeVideoConfig().transform_video_create_response(
                model="wan3.0-video",
                raw_response=_mock_response(
                    {"code": "InvalidApiKey", "message": "No API-key provided.", "request_id": "r1"}
                ),
                logging_obj=None,
                custom_llm_provider="dashscope",
                request_data={},
            )

        # the 200 it arrived on would have the proxy retry a call the provider already rejected
        assert raised.value.status_code == 400


class TestDashScopeVideoStatus:
    SUCCEEDED = {
        "request_id": "78c9b768",
        "output": {
            "task_id": "17ed7e50",
            "task_status": "SUCCEEDED",
            "submit_time": "2026-08-06 10:01:35.452",
            "scheduled_time": "2026-08-06 10:01:35.507",
            "end_time": "2026-08-06 10:13:33.838",
            "orig_prompt": "a golden retriever",
            "video_url": "https://oss.example.com/video.mp4",
        },
        "usage": {
            "video_count": 1,
            "duration": 5.0,
            "input_video_duration": 0.0,
            "output_video_duration": 5.0,
            "fps": 30,
            "SR": 720,
            "ratio": "16:9",
        },
    }

    def test_succeeded_task_mapping(self):
        video_obj = DashScopeVideoConfig().transform_video_status_retrieve_response(
            raw_response=_mock_response(self.SUCCEEDED),
            logging_obj=None,
            custom_llm_provider="dashscope",
        )

        assert video_obj.status == "completed"
        assert video_obj.seconds == "5.0"
        assert video_obj.completed_at is not None
        assert video_obj.created_at is not None
        assert video_obj.completed_at > video_obj.created_at
        assert decode_video_id_with_provider(video_obj.id)["video_id"] == "17ed7e50"

    def test_timestamps_are_read_as_utc_plus_8(self):
        """DashScope stamps times in UTC+8 with no offset; reading them as UTC
        would shift every video's created_at by 8 hours."""
        video_obj = DashScopeVideoConfig().transform_video_status_retrieve_response(
            raw_response=_mock_response(self.SUCCEEDED), logging_obj=None, custom_llm_provider="dashscope"
        )

        submitted = datetime(2026, 8, 6, 10, 1, 35, 452000, tzinfo=timezone(timedelta(hours=8)))
        assert video_obj.created_at == int(submitted.timestamp())

    def test_delivered_resolution_beats_the_requested_one_for_billing(self):
        """ratio adaptive lets DashScope deliver a different tier than asked
        for; usage.SR is what was produced and therefore what is billed."""
        video_obj = DashScopeVideoConfig().transform_video_status_retrieve_response(
            raw_response=_mock_response(self.SUCCEEDED), logging_obj=None, custom_llm_provider="dashscope"
        )

        assert video_obj.usage == {"duration_seconds": 5.0, "video_resolution": "720p"}

    def test_billed_duration_includes_input_video_seconds(self):
        """For edit and extend, DashScope bills usage.duration (output plus
        input video seconds), which is larger than the output alone."""
        payload = {
            "output": {"task_id": "t1", "task_status": "SUCCEEDED", "video_url": "https://x/v.mp4"},
            "usage": {"duration": 12.0, "input_video_duration": 7.0, "output_video_duration": 5.0, "SR": 1080},
        }
        video_obj = DashScopeVideoConfig().transform_video_status_retrieve_response(
            raw_response=_mock_response(payload), logging_obj=None, custom_llm_provider="dashscope"
        )

        assert video_obj.usage["duration_seconds"] == 12.0
        assert video_obj.usage["video_resolution"] == "1080p"

    def test_size_is_rebuilt_from_shortest_side_and_ratio(self):
        """DashScope splits geometry across SR and ratio; OpenAI's size field is
        pixels, so reporting the bare ratio would be meaningless."""
        video_obj = DashScopeVideoConfig().transform_video_status_retrieve_response(
            raw_response=_mock_response(self.SUCCEEDED), logging_obj=None, custom_llm_provider="dashscope"
        )

        assert video_obj.size == "1280x720"

    @pytest.mark.parametrize(
        "task_status,expected",
        [
            ("PENDING", "queued"),
            ("RUNNING", "in_progress"),
            ("SUCCEEDED", "completed"),
            ("FAILED", "failed"),
            ("CANCELED", "cancelled"),
            ("UNKNOWN", "failed"),
        ],
    )
    def test_status_mapping(self, task_status, expected):
        video_obj = DashScopeVideoConfig().transform_video_status_retrieve_response(
            raw_response=_mock_response({"output": {"task_id": "t1", "task_status": task_status}}),
            logging_obj=None,
            custom_llm_provider="dashscope",
        )

        assert video_obj.status == expected

    def test_running_task_has_no_completion_time(self):
        video_obj = DashScopeVideoConfig().transform_video_status_retrieve_response(
            raw_response=_mock_response(
                {"output": {"task_id": "t1", "task_status": "RUNNING", "submit_time": "2026-08-06 10:01:35.452"}}
            ),
            logging_obj=None,
            custom_llm_provider="dashscope",
        )

        assert video_obj.completed_at is None
        assert video_obj.usage is None

    def test_failed_task_maps_error_from_inside_output(self):
        """Unlike creation, task failures nest code and message inside output."""
        video_obj = DashScopeVideoConfig().transform_video_status_retrieve_response(
            raw_response=_mock_response(
                {
                    "output": {
                        "task_id": "eff1443c",
                        "task_status": "FAILED",
                        "code": "InvalidParameter",
                        "message": "The two modes are mutually exclusive.",
                    }
                }
            ),
            logging_obj=None,
            custom_llm_provider="dashscope",
        )

        assert video_obj.status == "failed"
        assert video_obj.error == {
            "code": "InvalidParameter",
            "message": "The two modes are mutually exclusive.",
        }

    def test_polled_id_keeps_the_deployment_it_was_polled_with(self):
        """The proxy resolves the deployment from the model encoded in the id;
        dropping it from the returned id sent follow-up polls and downloads to
        the default provider credentials instead of the deployment's."""
        polled_id = encode_video_id_with_provider("17ed7e50", "dashscope", "wan3-prod-deployment")
        logging_obj = Mock(litellm_params={"video_id": polled_id})

        video_obj = DashScopeVideoConfig().transform_video_status_retrieve_response(
            raw_response=_mock_response(self.SUCCEEDED), logging_obj=logging_obj, custom_llm_provider="dashscope"
        )

        decoded = decode_video_id_with_provider(video_obj.id)
        assert decoded["model_id"] == "wan3-prod-deployment"
        assert decoded["video_id"] == "17ed7e50"
        assert video_obj.model == "wan3-prod-deployment"

    def test_in_body_lookup_error_is_raised_instead_of_a_queued_video(self):
        """An unknown or expired task id comes back as a 200 with top-level code
        and no output; mapping it as a task produced a queued video with an
        empty id that callers would poll forever."""
        with pytest.raises(DashScopeVideoError, match="task not found"):
            DashScopeVideoConfig().transform_video_status_retrieve_response(
                raw_response=_mock_response({"code": "InvalidParameter", "message": "task not found"}),
                logging_obj=None,
                custom_llm_provider="dashscope",
            )

    def test_a_body_with_no_output_and_no_message_still_raises(self):
        with pytest.raises(DashScopeVideoError, match="DashScope video request failed"):
            DashScopeVideoConfig().transform_video_status_retrieve_response(
                raw_response=_mock_response({"request_id": "r1"}),
                logging_obj=None,
                custom_llm_provider="dashscope",
            )

    def test_undocumented_task_status_raises_rather_than_defaulting_to_queued(self):
        """Defaulting it left the caller polling a job that would never resolve."""
        with pytest.raises(DashScopeVideoError, match="Unknown DashScope task status: SCHEDULING") as raised:
            DashScopeVideoConfig().transform_video_status_retrieve_response(
                raw_response=_mock_response({"output": {"task_id": "t1", "task_status": "SCHEDULING"}}),
                logging_obj=None,
                custom_llm_provider="dashscope",
            )

        assert raised.value.status_code == 502

    @pytest.mark.parametrize(
        "payload",
        [
            {"output": {"video_url": "https://oss/video.mp4"}},  # no task_id or task_status
            {"output": {"task_id": "t1"}},
        ],
    )
    def test_an_output_block_missing_task_id_or_status_is_rejected(self, payload):
        """They are read straight into the returned video, so a body without them
        is rejected at the boundary instead of producing an empty id."""
        with pytest.raises(DashScopeVideoError, match="Unexpected DashScope video response") as raised:
            _parse_task_response(_mock_response(payload))

        assert raised.value.status_code == 502

    def test_unparseable_body_raises_the_provider_error_not_a_parse_error(self):
        """A gateway returning html used to escape as a JSONDecodeError, which the
        proxy maps to a generic connection error."""
        response = _mock_response({}, status_code=502)
        response.json.side_effect = json.JSONDecodeError("Expecting value", "<html>502 Bad Gateway</html>", 0)

        with pytest.raises(DashScopeVideoError, match="Unexpected DashScope video response") as raised:
            _parse_task_response(response)

        assert raised.value.status_code == 502

    def test_request_decodes_task_id_from_wrapped_video_id(self):
        encoded = encode_video_id_with_provider("17ed7e50", "dashscope", "wan3.0-video")
        url, params = DashScopeVideoConfig().transform_video_status_retrieve_request(
            video_id=encoded, api_base=API_BASE, litellm_params=GenericLiteLLMParams(), headers={}
        )

        assert url == f"{API_BASE}/api/v1/tasks/17ed7e50"
        assert params == {}


class TestDashScopeVideoContent:
    def test_content_request_targets_the_task_endpoint(self):
        encoded = encode_video_id_with_provider("17ed7e50", "dashscope", "wan3.0-video")
        url, params = DashScopeVideoConfig().transform_video_content_request(
            video_id=encoded, api_base=API_BASE, litellm_params=GenericLiteLLMParams(), headers={}
        )

        assert url == f"{API_BASE}/api/v1/tasks/17ed7e50"
        assert params == {}

    def test_url_is_extracted_from_a_succeeded_task(self):
        response = _mock_response(
            {"output": {"task_id": "t1", "task_status": "SUCCEEDED", "video_url": "https://oss/video.mp4"}}
        )

        assert _video_url_from_task(_parse_task_response(response), response) == "https://oss/video.mp4"

    @pytest.mark.parametrize("task_status", ["PENDING", "RUNNING"])
    def test_pending_task_raises_still_processing(self, task_status):
        """A 4xx, so the proxy reports the caller's timing mistake instead of
        retrying a job that is simply not finished."""
        response = _mock_response({"output": {"task_id": "t1", "task_status": task_status}})

        with pytest.raises(DashScopeVideoError, match="still processing") as raised:
            _video_url_from_task(_parse_task_response(response), response)

        assert raised.value.status_code == 409

    def test_failed_task_surfaces_the_upstream_message(self):
        response = _mock_response(
            {"output": {"task_id": "t1", "task_status": "FAILED", "message": "content policy violation"}}
        )

        with pytest.raises(DashScopeVideoError, match="content policy violation") as raised:
            _video_url_from_task(_parse_task_response(response), response)

        assert raised.value.status_code == 400

    def test_expired_task_explains_the_24h_window(self):
        """A task id older than 24h comes back UNKNOWN with no error, which is
        otherwise indistinguishable from a bad id."""
        response = _mock_response({"output": {"task_id": "t1", "task_status": "UNKNOWN"}})

        with pytest.raises(DashScopeVideoError, match="24 hours") as raised:
            _video_url_from_task(_parse_task_response(response), response)

        assert raised.value.status_code == 404


class TestDashScopeVideoEnvironment:
    def test_async_header_is_always_sent(self):
        """DashScope rejects video-synthesis calls without X-DashScope-Async,
        with 'current user api does not support synchronous calls'."""
        headers = DashScopeVideoConfig().validate_environment(
            headers={}, model="wan3.0-video", api_key="sk-test", litellm_params=GenericLiteLLMParams()
        )

        assert headers["X-DashScope-Async"] == "enable"
        assert headers["Authorization"] == "Bearer sk-test"

    def test_api_key_from_litellm_params(self):
        headers = DashScopeVideoConfig().validate_environment(
            headers={}, model="wan3.0-video", api_key=None, litellm_params=GenericLiteLLMParams(api_key="sk-params")
        )

        assert headers["Authorization"] == "Bearer sk-params"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.setattr("litellm.api_key", None)

        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            DashScopeVideoConfig().validate_environment(
                headers={}, model="wan3.0-video", api_key=None, litellm_params=GenericLiteLLMParams()
            )

    def test_compatible_mode_chat_base_is_stripped_back_to_the_host(self):
        """DASHSCOPE_API_BASE is shared with chat and points at
        /compatible-mode/v1; the video API lives under /api/v1."""
        url = DashScopeVideoConfig().get_complete_url(
            model="wan3.0-video",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            litellm_params={},
        )

        assert url == API_BASE

    def test_workspace_scoped_host_is_preserved(self):
        url = DashScopeVideoConfig().get_complete_url(
            model="wan3.0-video", api_base="https://ws-123.cn-beijing.maas.aliyuncs.com/", litellm_params={}
        )

        assert url == "https://ws-123.cn-beijing.maas.aliyuncs.com"

    def test_default_host_is_beijing(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_BASE_VIDEO", raising=False)

        assert (
            DashScopeVideoConfig().get_complete_url(model="wan3.0-video", api_base=None, litellm_params={}) == API_BASE
        )


class TestDashScopeVideoBrandAliases:
    def test_qwencloud_defaults_to_the_international_host(self, monkeypatch):
        monkeypatch.delenv("QWENCLOUD_API_BASE_VIDEO", raising=False)

        assert (
            QwenCloudVideoConfig().get_complete_url(model="wan3.0-video", api_base=None, litellm_params={})
            == "https://dashscope-intl.aliyuncs.com"
        )

    def test_qwen_ai_platform_defaults_to_the_china_host(self, monkeypatch):
        monkeypatch.delenv("QWEN_AI_PLATFORM_API_BASE_VIDEO", raising=False)

        assert (
            QwenAIPlatformVideoConfig().get_complete_url(model="wan3.0-video", api_base=None, litellm_params={})
            == API_BASE
        )

    def test_brand_alias_falls_back_to_the_shared_dashscope_key(self, monkeypatch):
        monkeypatch.delenv("QWENCLOUD_API_KEY", raising=False)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-shared")
        monkeypatch.setattr("litellm.api_key", None)

        headers = QwenCloudVideoConfig().validate_environment(
            headers={}, model="wan3.0-video", api_key=None, litellm_params=GenericLiteLLMParams()
        )

        assert headers["Authorization"] == "Bearer sk-shared"


class TestDashScopeVideoUnsupportedOperations:
    @pytest.mark.parametrize("operation", ["remix", "list", "delete"])
    def test_operations_dashscope_does_not_expose_raise(self, operation):
        """DashScope publishes no remix, list or delete surface, so these must
        fail loudly instead of silently hitting a made-up endpoint."""
        config = DashScopeVideoConfig()
        calls = {
            "remix": lambda: config.transform_video_remix_request(
                video_id="v", prompt="p", api_base=API_BASE, litellm_params=GenericLiteLLMParams(), headers={}
            ),
            "list": lambda: config.transform_video_list_request(
                api_base=API_BASE, litellm_params=GenericLiteLLMParams(), headers={}
            ),
            "delete": lambda: config.transform_video_delete_request(
                video_id="v", api_base=API_BASE, litellm_params=GenericLiteLLMParams(), headers={}
            ),
        }

        with pytest.raises(NotImplementedError, match="not supported by DashScope"):
            calls[operation]()


class TestDashScopeVideoProviderWiring:
    @pytest.mark.parametrize(
        "provider,expected",
        [
            ("dashscope", DashScopeVideoConfig),
            ("qwencloud", QwenCloudVideoConfig),
            ("qwen_ai_platform", QwenAIPlatformVideoConfig),
        ],
    )
    def test_provider_resolves_to_a_video_config(self, provider, expected):
        """Without this registration litellm answers 'video generation is not
        supported for dashscope' before any transform runs."""
        import litellm
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_video_config(
            model="wan3.0-video", provider=litellm.LlmProviders(provider)
        )

        assert type(config) is expected


VIDEO_MODEL_TIERS: Final = (
    ("wan3.0-video", ("480p", "720p", "1080p")),
    ("wan3.0-video-prime", ("480p", "720p", "1080p")),
    ("happyhorse-1.1-t2v", ("480p", "720p", "1080p")),
    ("happyhorse-1.1-i2v", ("480p", "720p", "1080p")),
    ("happyhorse-1.1-r2v", ("480p", "720p", "1080p")),
    ("happyhorse-1.0-t2v", ("720p", "1080p")),
    ("happyhorse-1.0-i2v", ("720p", "1080p")),
    ("happyhorse-1.0-r2v", ("720p", "1080p")),
    ("wan2.7-t2v", ("720p", "1080p")),
    ("wan2.7-i2v", ("720p", "1080p")),
    ("wan2.7-r2v", ("720p", "1080p")),
)


@pytest.mark.usefixtures("local_model_cost_map")
class TestDashScopeVideoPricing:
    @pytest.mark.parametrize("provider", ["dashscope", "qwencloud", "qwen_ai_platform"])
    @pytest.mark.parametrize("model,tiers", VIDEO_MODEL_TIERS)
    def test_every_supported_tier_bills_its_own_rate(self, provider, model, tiers):
        """A tier without its own rate falls back to the base rate, so a 480P
        video would silently bill at the 1080P price; a missing alias entry
        bills the whole video at zero."""
        from litellm import get_model_info
        from litellm.llms.openai.cost_calculation import video_generation_cost

        info = get_model_info(model=model, custom_llm_provider=provider)

        for tier in tiers:
            tier_rate = info[f"output_cost_per_second_{tier}"]
            assert tier_rate > 0
            cost = video_generation_cost(
                model=model, duration_seconds=5.0, custom_llm_provider=provider, video_resolution=tier
            )
            assert cost == pytest.approx(tier_rate * 5.0)

    @pytest.mark.parametrize("provider", ["dashscope", "qwencloud", "qwen_ai_platform"])
    @pytest.mark.parametrize("model,tiers", VIDEO_MODEL_TIERS)
    def test_higher_tiers_never_bill_less_and_1080p_is_the_default(self, provider, model, tiers):
        """DashScope defaults to 1080P, so an unlabelled video must bill at the
        1080P rate rather than the cheapest tier."""
        from litellm import get_model_info

        info = get_model_info(model=model, custom_llm_provider=provider)
        rates = [info[f"output_cost_per_second_{tier}"] for tier in tiers]

        assert rates == sorted(rates)
        assert info["output_cost_per_second"] == info["output_cost_per_second_1080p"]

    @pytest.mark.parametrize("model,_", VIDEO_MODEL_TIERS)
    def test_each_prefix_is_priced_for_the_region_its_default_host_serves(self, model, _):
        """dashscope and qwen_ai_platform default to the Beijing host and
        qwencloud to the international one, so the two Beijing-routed prefixes
        must share a rate card."""
        from litellm import get_model_info

        dashscope = get_model_info(model=model, custom_llm_provider="dashscope")
        qwen_ai_platform = get_model_info(model=model, custom_llm_provider="qwen_ai_platform")

        assert dashscope["output_cost_per_second_1080p"] == qwen_ai_platform["output_cost_per_second_1080p"]


class TestDashScopeVideoEndToEndRequest:
    """
    Drive the real litellm entrypoint rather than calling the transform directly.

    ``video_generation()`` runs map_openai_params first and passes only the
    *mapped* params to transform_video_create_request, so anything the mapper
    drops never reaches the request builder. Asserting on the JSON the handler
    actually serialized also catches body fragments that aren't JSON-encodable.
    """

    @staticmethod
    async def _wire_request(**kwargs) -> dict:
        sent: Final[list[httpx.Request]] = []

        def respond(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return httpx.Response(200, json={"output": {"task_status": "PENDING", "task_id": "t1"}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
            handler: Final = AsyncHTTPHandler()
            await handler.close()
            handler.client = http_client
            await avideo_generation(api_key="sk-test", api_base=API_BASE, client=handler, **kwargs)

        (request,) = sent
        return {"serialized": json.loads(request.content), "headers": dict(request.headers)}

    @pytest.mark.asyncio
    async def test_input_reference_survives_param_mapping_into_the_body(self):
        """Regression: map_openai_params used to drop input_reference, so an
        image-to-video call silently degraded to text-to-video."""
        captured = await self._wire_request(
            model="dashscope/wan3.0-video",
            prompt="make it move",
            input_reference="https://x/first.png",
            seconds="10",
        )

        assert captured["serialized"]["input"]["media"] == [{"type": "first_frame", "url": "https://x/first.png"}]
        assert captured["serialized"]["parameters"]["duration"] == 10
        assert "input_reference" not in captured["serialized"]["input"]
        assert "input_reference" not in captured["serialized"].get("parameters", {})

    @pytest.mark.asyncio
    async def test_happyhorse_r2v_input_reference_reaches_the_body_as_a_reference_image(self):
        captured = await self._wire_request(
            model="dashscope/happyhorse-1.1-r2v",
            prompt="make it move",
            input_reference="https://x/subject.png",
        )

        assert captured["serialized"]["input"]["media"] == [{"type": "reference_image", "url": "https://x/subject.png"}]

    @pytest.mark.asyncio
    async def test_file_input_reference_is_json_encodable(self):
        """Regression: the first_frame entry was built as a MappingProxyType,
        which json.dumps refuses, so every i2v request raised before sending."""
        captured = await self._wire_request(
            model="dashscope/wan3.0-video",
            prompt="make it move",
            input_reference=io.BytesIO(PNG_BYTES),
        )

        media = captured["serialized"]["input"]["media"]
        assert media[0]["type"] == "first_frame"
        assert media[0]["url"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_auth_and_async_headers_reach_the_wire(self):
        """DashScope rejects the call outright without X-DashScope-Async."""
        captured = await self._wire_request(model="dashscope/wan3.0-video", prompt="p")

        headers = {k.lower(): v for k, v in captured["headers"].items()}
        assert headers["x-dashscope-async"] == "enable"
        assert headers["authorization"] == "Bearer sk-test"

    @pytest.mark.asyncio
    async def test_size_reaches_the_body_as_ratio_plus_resolution(self):
        captured = await self._wire_request(
            model="dashscope/happyhorse-1.1-t2v", prompt="a train", seconds="5", size="1920x1080"
        )

        assert captured["serialized"]["parameters"] == {
            "resolution": "1080P",
            "ratio": "16:9",
            "duration": 5,
        }

    @pytest.mark.asyncio
    async def test_multi_role_media_array_reaches_the_body_intact(self):
        media = [
            {"type": "reference_video", "url": "https://x/v.mp4"},
            {"type": "reference_audio", "url": "https://x/a.mp3"},
        ]
        captured = await self._wire_request(
            model="dashscope/wan3.0-video", prompt="视频1", media=media, duration=15, resolution="1080P"
        )

        assert captured["serialized"]["input"]["media"] == media
        assert captured["serialized"]["parameters"]["duration"] == 15
