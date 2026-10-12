"""
Tests for Volcengine Ark (Doubao-Seedance) video generation transformation.
"""

import base64
import io
import json
from typing import Final
from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.volcengine.videos.transformation import (
    VolcEngineVideoConfig,
    VolcEngineVideoError,
    _parse_task,
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

API_BASE = "https://ark.cn-beijing.volces.com"
TASKS_URL = f"{API_BASE}/api/v3/contents/generations/tasks"
SEEDANCE_2_5 = "doubao-seedance-2-5-260628"
SEEDANCE_2_0 = "doubao-seedance-2-0-260128"


def _mock_response(payload: dict, status_code: int = 200) -> Mock:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = payload
    mock_response.status_code = status_code
    mock_response.headers = httpx.Headers()
    return mock_response


def _create(params: dict, model: str = SEEDANCE_2_5, prompt: str = "a cat on a roof"):
    return VolcEngineVideoConfig().transform_video_create_request(
        model=model,
        prompt=prompt,
        api_base=API_BASE,
        video_create_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )


def _poll(payload: dict, provider: str | None = "volcengine"):
    return VolcEngineVideoConfig().transform_video_status_retrieve_response(
        raw_response=_mock_response(payload),
        logging_obj=Mock(),
        custom_llm_provider=provider,
    )


class TestVolcEngineVideoCreateRequest:
    def test_prompt_becomes_the_text_content_item(self):
        """Ark takes no top-level prompt: it only reads the typed content array."""
        body, files, url = _create({})

        assert body["content"] == [{"type": "text", "text": "a cat on a roof"}]
        assert "prompt" not in body
        assert files == ()
        assert url == TASKS_URL

    def test_input_reference_becomes_a_first_frame_item(self):
        body, _, _ = _create({"input_reference": "https://x/first.png"})

        assert body["content"] == [
            {"type": "text", "text": "a cat on a roof"},
            {"type": "image_url", "image_url": {"url": "https://x/first.png"}, "role": "first_frame"},
        ]

    def test_file_input_reference_becomes_a_data_uri(self):
        body, _, _ = _create({"input_reference": io.BytesIO(PNG_BYTES)})

        item = body["content"][1]
        assert item["role"] == "first_frame"
        assert item["image_url"]["url"].startswith("data:image/png;base64,")

    def test_caller_content_array_wins_and_keeps_the_prompt(self):
        """Roles on a caller-supplied array are the only way to express
        reference-to-video and edit jobs, so it must not be rebuilt."""
        supplied = [
            {"type": "image_url", "image_url": {"url": "https://x/ref.png"}, "role": "reference_image"},
            {"type": "video_url", "video_url": {"url": "https://x/ref.mp4"}, "role": "reference_video"},
        ]
        body, _, _ = _create({"content": supplied})

        assert body["content"] == [{"type": "text", "text": "a cat on a roof"}, *supplied]

    def test_caller_content_with_its_own_text_item_is_not_duplicated(self):
        supplied = [
            {"type": "text", "text": "custom"},
            {"type": "audio_url", "audio_url": {"url": "x"}, "role": "reference_audio"},
        ]
        body, _, _ = _create({"content": supplied})

        assert body["content"] == supplied

    def test_malformed_content_array_raises(self):
        with pytest.raises(ValueError, match="content must be an array"):
            _create({"content": ["not-an-item"]})

    def test_provider_params_land_next_to_the_content_array(self):
        body, _, _ = _create(
            {"resolution": "1080p", "ratio": "9:16", "duration": 8, "generate_audio": True, "watermark": False}
        )

        assert body["resolution"] == "1080p"
        assert body["ratio"] == "9:16"
        assert body["duration"] == 8
        assert body["generate_audio"] is True
        assert body["watermark"] is False

    def test_false_valued_flags_are_not_dropped(self):
        """watermark=False and camera_fixed=False are meaningful; a truthiness
        check instead of an is-None check would silently flip them."""
        body, _, _ = _create({"watermark": False, "camera_fixed": False})

        assert body["watermark"] is False
        assert body["camera_fixed"] is False


class TestVolcEngineVideoParamMapping:
    def test_seconds_becomes_duration(self):
        assert VolcEngineVideoConfig().map_openai_params({"seconds": "5"}, "m", True) == {"duration": 5}

    def test_size_yields_ratio_and_resolution_tier(self):
        """Ark takes a named tier, not a pixel size; dropping it leaves the
        request on the model default and bills at that default's rate."""
        mapped = VolcEngineVideoConfig().map_openai_params({"size": "1920x1080"}, "m", True)

        assert mapped == {"ratio": "16:9", "resolution": "1080p"}

    @pytest.mark.parametrize(
        "size,expected_ratio,expected_resolution",
        [
            ("1280x720", "16:9", "720p"),
            ("854x480", "16:9", "480p"),
            ("1080x1920", "9:16", "1080p"),
            ("1024x1024", "1:1", "1080p"),
            ("3840x2160", "16:9", "4k"),
            ("1080x1440", "3:4", "1080p"),
        ],
    )
    def test_size_snaps_onto_the_ratios_ark_accepts(self, size, expected_ratio, expected_resolution):
        """854x480 reduces to 427:240 by gcd, which Ark rejects, so the closest
        named ratio has to win instead."""
        mapped = VolcEngineVideoConfig().map_openai_params({"size": size}, "m", True)

        assert mapped["ratio"] == expected_ratio
        assert mapped["resolution"] == expected_resolution

    def test_explicit_ratio_and_resolution_beat_the_size_derived_ones(self):
        mapped = VolcEngineVideoConfig().map_openai_params(
            {"size": "1280x720", "ratio": "adaptive", "resolution": "1080p"}, "m", True
        )

        assert mapped == {"ratio": "adaptive", "resolution": "1080p"}

    def test_parameters_block_is_merged_in(self):
        mapped = VolcEngineVideoConfig().map_openai_params({"parameters": {"priority": 5}}, "m", True)

        assert mapped == {"priority": 5}

    def test_openai_only_fields_ark_does_not_define_are_dropped(self):
        mapped = VolcEngineVideoConfig().map_openai_params({"user": "u", "characters": [{"id": "c"}]}, "m", True)

        assert mapped == {}


class TestVolcEngineVideoCreateResponse:
    def test_task_id_becomes_a_queued_video(self):
        config = VolcEngineVideoConfig()
        video = config.transform_video_create_response(
            model=SEEDANCE_2_5,
            raw_response=_mock_response({"id": "cgt-2026-abc", "safety_identifier": None}),
            logging_obj=Mock(),
            custom_llm_provider="volcengine",
        )

        assert video.status == "queued"
        assert video.model == SEEDANCE_2_5
        assert decode_video_id_with_provider(video.id)["video_id"] == "cgt-2026-abc"
        assert decode_video_id_with_provider(video.id)["model_id"] == SEEDANCE_2_5

    def test_missing_task_id_raises_instead_of_returning_an_empty_video(self):
        """A rejected create must not surface as a queued video with an empty id."""
        config = VolcEngineVideoConfig()

        with pytest.raises(VolcEngineVideoError, match="did not return a video task id"):
            config.transform_video_create_response(
                model=SEEDANCE_2_5,
                raw_response=_mock_response({"error": {"code": "ModelNotOpen", "message": "model not activated"}}),
                logging_obj=Mock(),
            )

    def _create_response(self, request_data: dict | None, payload: dict | None = None, model: str = SEEDANCE_2_5):
        return VolcEngineVideoConfig().transform_video_create_response(
            model=model,
            raw_response=_mock_response(payload or {"id": "cgt-2026-abc"}),
            logging_obj=Mock(),
            request_data=request_data,
        )

    def test_create_response_carries_the_requested_usage(self):
        """Only the create call is costed, and Ark's create response has no usage
        block, so the requested values have to be reported here or every video
        is logged at zero cost."""
        video = self._create_response({"duration": 4, "resolution": "480p"})

        assert video.usage == {"duration_seconds": 4.0, "video_resolution": "480p"}

    def test_model_chosen_duration_is_not_reported_as_billable_seconds(self):
        """duration -1 means Ark picks the length, so there is no honest number
        to bill at create time."""
        video = self._create_response({"duration": -1, "resolution": "720p"})

        assert "duration_seconds" not in video.usage
        assert video.usage["video_resolution"] == "720p"

    def test_create_response_without_a_request_body_has_no_usage(self):
        assert self._create_response(None).usage is None

    def test_an_omitted_duration_bills_the_2_0_default_length(self):
        """Leaving seconds off is not free: Ark runs the model's own default
        length, which for the 2.0 family is 5s, so the create response has to
        report it or the video is logged at zero spend."""
        video = self._create_response({"resolution": "480p"}, model=SEEDANCE_2_0)

        assert video.usage == {"duration_seconds": 5.0, "video_resolution": "480p"}

    def test_an_omitted_duration_is_not_invented_for_2_5(self):
        """The 2.5 family defaults duration to -1, so Ark picks the length and
        there is no honest number to bill at create time."""
        video = self._create_response({"resolution": "480p"}, model=SEEDANCE_2_5)

        assert "duration_seconds" not in video.usage

    def test_the_2_0_default_does_not_override_an_explicit_minus_one(self):
        """A caller asking Ark to pick the length must not be billed the model
        default instead."""
        video = self._create_response({"duration": -1, "resolution": "480p"}, model=SEEDANCE_2_0)

        assert "duration_seconds" not in video.usage

    def test_create_response_echoes_the_requested_seconds_and_size(self):
        video = self._create_response({"duration": 4, "resolution": "480p", "ratio": "16:9"}, model=SEEDANCE_2_0)

        assert video.seconds == "4.0"
        assert video.size == "864x496"

    @pytest.mark.usefixtures("local_model_cost_map")
    def test_the_create_call_is_not_billed_at_zero(self):
        """Regression guard for the whole point of the usage block: feed what
        the create response reports back through the cost calculator, which is
        exactly what the proxy does when it logs spend."""
        from litellm.llms.openai.cost_calculation import video_generation_cost
        from litellm.utils import get_model_info

        video = self._create_response({"duration": 4, "resolution": "1080p"})
        cost = video_generation_cost(
            model=SEEDANCE_2_5,
            duration_seconds=video.usage["duration_seconds"],
            custom_llm_provider="volcengine",
            video_resolution=video.usage["video_resolution"],
        )
        info = get_model_info(model=f"volcengine/{SEEDANCE_2_5}", custom_llm_provider="volcengine")

        assert cost == pytest.approx(info["output_cost_per_second_1080p"] * 4.0)
        assert cost > 0


class TestVolcEngineVideoStatus:
    @pytest.mark.parametrize(
        "ark_status,expected",
        [
            ("queued", "queued"),
            ("running", "in_progress"),
            ("succeeded", "completed"),
            ("failed", "failed"),
            ("cancelled", "cancelled"),
            ("expired", "failed"),
            ("Queued", "queued"),
        ],
    )
    def test_status_is_mapped_onto_the_openai_vocabulary(self, ark_status, expected):
        video = _poll({"id": "cgt-1", "status": ark_status})

        assert video.status == expected

    def test_completed_task_reports_delivered_geometry_and_duration(self):
        """Billing reads Ark's own duration and resolution, not the requested
        ones: duration=-1 and ratio=adaptive both let Ark deliver something
        else, and the tier is what the per-second rate is keyed on."""
        video = _poll(
            {
                "id": "cgt-1",
                "status": "succeeded",
                "model": SEEDANCE_2_5,
                "created_at": 1718049470,
                "updated_at": 1718049700,
                "resolution": "1080p",
                "ratio": "16:9",
                "duration": "8",
                "content": {"video_url": "https://tos/v.mp4"},
                "usage": {"completion_tokens": 388800, "total_tokens": 388800},
            }
        )

        assert video.usage["duration_seconds"] == 8.0
        assert video.usage["video_resolution"] == "1080p"
        assert video.usage["total_tokens"] == 388800
        assert video.size == "1920x1080"
        assert video.seconds == "8.0"
        assert video.created_at == 1718049470
        assert video.completed_at == 1718049700

    def test_adaptive_ratio_does_not_produce_a_bogus_pixel_size(self):
        video = _poll({"id": "cgt-1", "status": "succeeded", "resolution": "720p", "ratio": "adaptive"})

        assert video.size is None
        assert video.usage["video_resolution"] == "720p"

    def test_portrait_delivery_reports_portrait_pixels(self):
        video = _poll({"id": "cgt-1", "status": "succeeded", "resolution": "720p", "ratio": "9:16"})

        assert video.size == "720x1280"

    def test_480p_reports_the_width_ark_actually_produces(self):
        """480 * 16 / 9 is 853.33 and the 2.5 family delivers 854x480, so
        truncating would report a width the model never produces."""
        video = _poll(
            {"id": "cgt-1", "status": "succeeded", "model": SEEDANCE_2_5, "resolution": "480p", "ratio": "16:9"}
        )

        assert video.size == "854x480"

    @pytest.mark.parametrize("ratio,expected", [("16:9", "864x496"), ("4:3", "752x560")])
    def test_480p_size_follows_the_2_0_familys_own_frame(self, ratio, expected):
        """The 2.0 family renders a 480p canvas the tier name does not imply
        (16:9 is 864x496, not 854x480), so scaling the tier's nominal shortest
        side up the ratio reports pixels the model never produces."""
        video = _poll(
            {
                "id": "cgt-1",
                "status": "succeeded",
                "model": "doubao-seedance-2-0-260128",
                "resolution": "480p",
                "ratio": ratio,
            }
        )

        assert video.size == expected

    def test_720p_size_is_the_same_for_both_families(self):
        """Only the 480p canvas differs; both families deliver 1280x720."""
        video = _poll(
            {
                "id": "cgt-1",
                "status": "succeeded",
                "model": "doubao-seedance-2-0-260128",
                "resolution": "720p",
                "ratio": "16:9",
            }
        )

        assert video.size == "1280x720"

    @pytest.mark.parametrize("model,expected", [("doubao-seedance-2-0-260128", "496x864"), (SEEDANCE_2_5, "480x854")])
    def test_portrait_480p_follows_the_same_family_frame(self, model, expected):
        """The portrait frame is not just the landscape one transposed: the 2.5
        family's 480p short side is 480, the 2.0 family's is 496."""
        video = _poll({"id": "cgt-1", "status": "succeeded", "model": model, "resolution": "480p", "ratio": "9:16"})

        assert video.size == expected

    def test_a_ratio_off_the_table_still_reports_a_pixel_size(self):
        """A (tier, ratio) the model has no measured frame for falls back to the
        tier's nominal shortest side rather than dropping the size."""
        video = _poll(
            {"id": "cgt-1", "status": "succeeded", "model": SEEDANCE_2_5, "resolution": "1080p", "ratio": "4:3"}
        )

        assert video.size == "1440x1080"

    def test_queued_task_has_no_completion_time(self):
        video = _poll({"id": "cgt-1", "status": "queued", "updated_at": 1718049700})

        assert video.completed_at is None

    def test_failed_task_carries_the_error_block(self):
        video = _poll(
            {
                "id": "cgt-1",
                "status": "failed",
                "error": {"code": "OutputVideoSensitiveContentDetected", "message": "flagged"},
            }
        )

        assert video.status == "failed"
        assert video.error == {"code": "OutputVideoSensitiveContentDetected", "message": "flagged"}


class TestVolcEngineVideoContent:
    def test_content_url_is_read_from_the_task(self):
        task = _parse_task(
            _mock_response({"id": "cgt-1", "status": "succeeded", "content": {"video_url": "https://tos/v.mp4"}})
        )

        assert _video_url_from_task(task) == "https://tos/v.mp4"

    def test_pending_task_raises_a_retry_hint(self):
        task = _parse_task(_mock_response({"id": "cgt-1", "status": "running"}))

        with pytest.raises(ValueError, match="still processing"):
            _video_url_from_task(task)

    def test_failed_task_surfaces_the_upstream_message(self):
        task = _parse_task(_mock_response({"id": "cgt-1", "status": "failed", "error": {"message": "quota exceeded"}}))

        with pytest.raises(ValueError, match="quota exceeded"):
            _video_url_from_task(task)

    def test_expired_task_points_at_the_retention_window(self):
        task = _parse_task(_mock_response({"id": "cgt-1", "status": "expired"}))

        with pytest.raises(ValueError, match="expire 7 days"):
            _video_url_from_task(task)

    def test_content_request_polls_the_task(self):
        url, params = VolcEngineVideoConfig().transform_video_content_request(
            video_id="cgt-1", api_base=API_BASE, litellm_params=GenericLiteLLMParams(), headers={}
        )

        assert url == f"{TASKS_URL}/cgt-1"
        assert params == {}


class TestVolcEngineVideoList:
    def test_list_hits_the_task_collection_with_ark_paging(self):
        """Ark paginates with page_num/page_size and has no cursor or order, so
        limit maps to page_size and the filters ride on extra_query."""
        url, params = VolcEngineVideoConfig().transform_video_list_request(
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
            limit=20,
            extra_query={"filter.status": "succeeded"},
        )

        assert url == TASKS_URL
        assert params == {"page_size": "20", "filter.status": "succeeded"}

    def test_list_response_is_an_openai_list_body(self):
        response = VolcEngineVideoConfig().transform_video_list_response(
            raw_response=_mock_response(
                {
                    "total": 1,
                    "items": [
                        {
                            "id": "cgt-1",
                            "status": "succeeded",
                            "model": SEEDANCE_2_5,
                            "resolution": "720p",
                            "ratio": "16:9",
                            "duration": 5,
                        }
                    ],
                }
            ),
            logging_obj=Mock(),
            custom_llm_provider="volcengine",
        )

        assert response["object"] == "list"
        assert response["total"] == 1
        assert response["first_id"] == response["last_id"]
        (item,) = response["data"]
        assert item["status"] == "completed"
        assert decode_video_id_with_provider(item["id"])["video_id"] == "cgt-1"

    def test_empty_list_body_still_carries_the_object_type(self):
        response = VolcEngineVideoConfig().transform_video_list_response(
            raw_response=_mock_response({"total": 0, "items": []}),
            logging_obj=Mock(),
        )

        assert response == {"object": "list", "data": [], "total": 0}


class TestVolcEngineVideoDelete:
    def test_delete_targets_the_task_url(self):
        url, params = VolcEngineVideoConfig().transform_video_delete_request(
            video_id=encode_video_id_with_provider("cgt-1", "volcengine", SEEDANCE_2_5),
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == f"{TASKS_URL}/cgt-1"
        assert params == {}

    def test_delete_response_reports_the_deleted_task(self):
        video = VolcEngineVideoConfig().transform_video_delete_response(
            raw_response=_mock_response({"id": "cgt-1", "model": SEEDANCE_2_5}), logging_obj=Mock()
        )

        assert video.id == "cgt-1"
        assert video.status == "deleted"

    def test_delete_response_tolerates_an_empty_body(self):
        """Ark's delete has no documented body, so an empty 200 must not crash."""
        raw_response = Mock(spec=httpx.Response)
        raw_response.json.side_effect = ValueError("no body")

        video = VolcEngineVideoConfig().transform_video_delete_response(raw_response=raw_response, logging_obj=Mock())

        assert video.status == "deleted"


class TestVolcEngineVideoEditAndExtension:
    @staticmethod
    def _video_id(source: str = "cgt-1") -> str:
        return encode_video_id_with_provider(source, "volcengine", SEEDANCE_2_5)

    def test_edit_pins_the_adaptive_geometry_ark_requires(self):
        """Ark rejects an edit task that does not ask it to resolve the source
        geometry itself, so ratio and duration cannot be left to the caller."""
        url, body, files = VolcEngineVideoConfig().transform_video_edit_request(
            prompt="make it brighter",
            video_id=self._video_id(),
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
            extra_body={"video_url": "https://tos/source.mp4"},
        )

        assert url == TASKS_URL
        assert files is None
        assert body["ratio"] == "adaptive"
        assert body["duration"] == -1
        assert body["content"] == [
            {"type": "text", "text": "make it brighter"},
            {"type": "video_url", "video_url": {"url": "https://tos/source.mp4"}, "role": "reference_video"},
        ]

    def test_edit_recovers_the_model_and_the_source_url_from_the_video_id(self):
        """The edit transform gets no model argument, so both the model and, when
        the id wraps a URL, the source video have to come out of the video id."""
        _, body, _ = VolcEngineVideoConfig().transform_video_edit_request(
            prompt="p",
            video_id=self._video_id("https://tos/source.mp4"),
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert body["model"] == SEEDANCE_2_5
        assert body["content"][1]["video_url"]["url"] == "https://tos/source.mp4"

    def test_edit_model_can_be_named_in_extra_body(self):
        _, body, _ = VolcEngineVideoConfig().transform_video_edit_request(
            prompt="p",
            video_id="https://tos/source.mp4",
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
            extra_body={"model": SEEDANCE_2_5},
        )

        assert body["model"] == SEEDANCE_2_5

    def test_edit_model_falls_back_to_the_routed_model(self):
        """A caller editing an arbitrary video URL passes the model once, for
        routing; making them repeat it in extra_body is a trap. The router's
        copy carries litellm's provider prefix, which Ark does not take."""
        _, body, _ = VolcEngineVideoConfig().transform_video_edit_request(
            prompt="p",
            video_id="https://tos/source.mp4",
            api_base=API_BASE,
            litellm_params={"model": f"volcengine/{SEEDANCE_2_5}"},  # pyright: ignore[reportArgumentType]  # the handler hands the transform a mapping, not the GenericLiteLLMParams the base declares
            headers={},
        )

        assert body["model"] == SEEDANCE_2_5

    def test_edit_model_falls_back_to_the_routed_model_object(self):
        """litellm's own entrypoint builds a GenericLiteLLMParams, so the
        fallback has to read both shapes."""
        _, body, _ = VolcEngineVideoConfig().transform_video_edit_request(
            prompt="p",
            video_id="https://tos/source.mp4",
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(model=f"volcengine/{SEEDANCE_2_5}"),
            headers={},
        )

        assert body["model"] == SEEDANCE_2_5

    def test_edit_without_a_model_raises(self):
        with pytest.raises(ValueError, match="need the model that produced the source video"):
            VolcEngineVideoConfig().transform_video_edit_request(
                prompt="p",
                video_id="https://tos/source.mp4",
                api_base=API_BASE,
                litellm_params={},
                headers={},
            )

    def test_edit_without_a_resolvable_source_url_raises(self):
        """Ark's task query is a GET while litellm's edit prefetch hook only
        POSTs, so a bare task id cannot be turned into a video URL."""
        with pytest.raises(ValueError, match="need the source video URL"):
            VolcEngineVideoConfig().transform_video_edit_request(
                prompt="p",
                video_id=self._video_id(),
                api_base=API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_extension_carries_the_seconds_to_add(self):
        url, body = VolcEngineVideoConfig().transform_video_extension_request(
            prompt="keep going",
            video_id=self._video_id("https://tos/source.mp4"),
            seconds="5",
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == TASKS_URL
        assert body["duration"] == 5
        assert body["ratio"] == "adaptive"
        assert body["content"][1]["role"] == "reference_video"

    def test_extension_without_seconds_leaves_the_duration_to_ark(self):
        _, body = VolcEngineVideoConfig().transform_video_extension_request(
            prompt="keep going",
            video_id=self._video_id("https://tos/source.mp4"),
            seconds="",
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "duration" not in body

    def test_remix_raises_rather_than_guessing_an_endpoint(self):
        with pytest.raises(NotImplementedError, match="not supported by Volcengine"):
            VolcEngineVideoConfig().transform_video_remix_request(
                video_id="v", prompt="p", api_base=API_BASE, litellm_params=GenericLiteLLMParams(), headers={}
            )


class TestVolcEngineVideoEnvironment:
    @pytest.mark.parametrize(
        "api_base",
        [
            None,
            API_BASE,
            f"{API_BASE}/",
            f"{API_BASE}/api/v3",
            f"{API_BASE}/api/compatible",
            f"{API_BASE}/api/v3/responses",
            f"{API_BASE}/v1",
        ],
    )
    def test_api_base_is_normalized_to_the_host_root(self, api_base):
        """Every Ark surface shares one credential, so the api_base may carry
        whichever one the caller uses. Leaving the surface on the base produces
        /api/compatible/api/v3/... and a 404 from Ark."""
        assert VolcEngineVideoConfig().get_complete_url("m", api_base, {}) == API_BASE

    def test_a_path_prefix_before_the_api_marker_survives(self):
        """A self-hosted Ark-compatible gateway may sit under a path prefix."""
        url = VolcEngineVideoConfig().get_complete_url("m", "https://gw.internal/ark/api/v3", {})

        assert url == "https://gw.internal/ark"

    def test_bearer_token_reaches_the_headers(self):
        headers = VolcEngineVideoConfig().validate_environment({}, SEEDANCE_2_5, api_key="sk-ark")

        assert headers["Authorization"] == "Bearer sk-ark"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)

        with pytest.raises(ValueError, match="Volcengine API key is required"):
            VolcEngineVideoConfig().validate_environment({}, SEEDANCE_2_5)


class TestVolcEngineVideoProviderWiring:
    def test_provider_resolves_to_the_video_config(self):
        """Without this registration litellm answers 'video generation is not
        supported for volcengine' before any transform runs."""
        import litellm
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_video_config(
            model=SEEDANCE_2_5, provider=litellm.LlmProviders("volcengine")
        )

        assert type(config) is VolcEngineVideoConfig


@pytest.mark.usefixtures("local_model_cost_map")
class TestVolcEngineVideoPricing:
    """
    Ark bills per output token, so the per-second tiers are derived rather than
    published as such. These assert the derivation holds rather than pinning a
    list price that a vendor price change would break.
    """

    @pytest.mark.parametrize(
        "model,tiers",
        [
            ("doubao-seedance-2-5-260628", {"480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080)}),
            (
                "doubao-seedance-2-0-260128",
                {"480p": (864, 496), "720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)},
            ),
            ("doubao-seedance-2-0-fast-260128", {"480p": (864, 496), "720p": (1280, 720)}),
            ("doubao-seedance-2-0-mini-260615", {"480p": (864, 496), "720p": (1280, 720)}),
        ],
    )
    def test_each_tier_rate_sits_at_its_pixel_ratio_to_the_base_rate(self, model, tiers):
        """Ark bills tokens = w*h*24*seconds/1024, so a tier's per-second rate
        must be the base rate scaled by its pixel count. The 2.0 family's 16:9
        480p frame is 864x496, not the 854x480 the tier name implies, which is
        what its 480p rate has to be scaled against. A tier whose key is
        missing silently falls back to the base rate instead."""
        from litellm.utils import get_model_info

        info = get_model_info(model=f"volcengine/{model}", custom_llm_provider="volcengine")
        base_rate: Final = info["output_cost_per_second"]
        base_pixels: Final = 1280 * 720

        for tier, (width, height) in tiers.items():
            rate = info[f"output_cost_per_second_{tier}"]
            assert rate == pytest.approx(base_rate * width * height / base_pixels, rel=1e-5), tier

    def test_cost_calculator_picks_the_reported_tier(self):
        """The transform reports Ark's delivered resolution; if it did not, every
        video would bill at the base tier regardless of what was generated."""
        from litellm.llms.openai.cost_calculation import video_generation_cost
        from litellm.utils import get_model_info

        info = get_model_info(model="volcengine/doubao-seedance-2-5-260628", custom_llm_provider="volcengine")
        cost = video_generation_cost(
            model="doubao-seedance-2-5-260628",
            duration_seconds=4.0,
            custom_llm_provider="volcengine",
            video_resolution="1080p",
        )

        assert cost == pytest.approx(info["output_cost_per_second_1080p"] * 4.0)
        assert cost > info["output_cost_per_second_720p"] * 4.0

    def test_unlabelled_resolution_falls_back_to_the_base_tier(self):
        """An unknown tier must not bill at zero."""
        from litellm.llms.openai.cost_calculation import video_generation_cost
        from litellm.utils import get_model_info

        info = get_model_info(model="volcengine/doubao-seedance-2-5-260628", custom_llm_provider="volcengine")
        cost = video_generation_cost(
            model="doubao-seedance-2-5-260628",
            duration_seconds=3.0,
            custom_llm_provider="volcengine",
            video_resolution=None,
        )

        assert cost == pytest.approx(info["output_cost_per_second"] * 3.0)
        assert cost > 0


class TestVolcEngineVideoEndToEndRequest:
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
            return httpx.Response(200, json={"id": "cgt-2026-abc", "safety_identifier": None})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
            handler: Final = AsyncHTTPHandler()
            await handler.close()
            handler.client = http_client
            await avideo_generation(api_key="sk-test", api_base=API_BASE, client=handler, **kwargs)

        (request,) = sent
        return {"serialized": json.loads(request.content), "headers": dict(request.headers), "url": str(request.url)}

    @pytest.mark.asyncio
    async def test_seconds_and_size_reach_the_body_as_duration_and_tier(self):
        captured = await self._wire_request(
            model=f"volcengine/{SEEDANCE_2_5}", prompt="a train", seconds="5", size="1920x1080"
        )

        assert captured["serialized"] == {
            "model": SEEDANCE_2_5,
            "content": [{"type": "text", "text": "a train"}],
            "resolution": "1080p",
            "ratio": "16:9",
            "duration": 5,
        }
        assert captured["url"] == TASKS_URL

    @pytest.mark.asyncio
    async def test_input_reference_survives_param_mapping_into_the_body(self):
        """Regression: if the mapper drops input_reference an image-to-video call
        silently degrades to text-to-video."""
        captured = await self._wire_request(
            model=f"volcengine/{SEEDANCE_2_5}", prompt="make it move", input_reference="https://x/first.png"
        )

        assert captured["serialized"]["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "https://x/first.png"},
            "role": "first_frame",
        }
        assert "input_reference" not in captured["serialized"]

    @pytest.mark.asyncio
    async def test_extra_body_content_array_reaches_the_wire(self):
        """Reference-to-video has no OpenAI-shaped parameter, so extra_body is
        the only way to express it."""
        captured = await self._wire_request(
            model=f"volcengine/{SEEDANCE_2_5}",
            prompt="use these",
            extra_body={
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://x/a.png"}, "role": "reference_image"},
                    {"type": "audio_url", "audio_url": {"url": "https://x/a.mp3"}, "role": "reference_audio"},
                ]
            },
        )

        assert captured["serialized"]["content"] == [
            {"type": "text", "text": "use these"},
            {"type": "image_url", "image_url": {"url": "https://x/a.png"}, "role": "reference_image"},
            {"type": "audio_url", "audio_url": {"url": "https://x/a.mp3"}, "role": "reference_audio"},
        ]

    @pytest.mark.asyncio
    async def test_bearer_token_reaches_the_wire(self):
        captured = await self._wire_request(model=f"volcengine/{SEEDANCE_2_5}", prompt="p")

        headers = {k.lower(): v for k, v in captured["headers"].items()}
        assert headers["authorization"] == "Bearer sk-test"
        assert headers["content-type"] == "application/json"
