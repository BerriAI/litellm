import base64
import io
import json
from pathlib import Path
from typing import Final
from unittest.mock import Mock

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.exceptions import UnsupportedParamsError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai.cost_calculation import video_generation_cost
from litellm.llms.volcengine.videos.transformation import VolcEngineVideoConfig, VolcEngineVideoError
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider
from litellm.utils import ProviderConfigManager
from litellm.videos.main import avideo_content, avideo_generation

PNG_BYTES: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_DATA_URL: Final = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"

API_BASE: Final = "https://ark.cn-beijing.volces.com"
TASKS_URL: Final = f"{API_BASE}/api/v3/contents/generations/tasks"
SEEDANCE_2_5: Final = "doubao-seedance-2-5-260628"
SEEDANCE_2_0: Final = "doubao-seedance-2-0-260128"
SEEDANCE_2_0_FAST: Final = "doubao-seedance-2-0-fast-260128"
TASK_ID: Final = "cgt-2026-abc"


def _response(payload: object, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=payload, request=httpx.Request("GET", TASKS_URL))


def _text_response(text: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, text=text, request=httpx.Request("GET", TASKS_URL))


def _create_body(
    params: dict[str, object], model: str = SEEDANCE_2_5, drop_params: bool | None = None
) -> dict[str, object]:
    body, files, url = VolcEngineVideoConfig().transform_video_create_request(
        model=model,
        prompt="a cat on a roof",
        api_base=API_BASE,
        video_create_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(drop_params=drop_params),
        headers={},
    )
    assert (files, url) == ((), TASKS_URL)
    return body


def _created(request_data: dict[str, object], model: str = SEEDANCE_2_5, logging_obj: object = None):
    return VolcEngineVideoConfig().transform_video_create_response(
        model=model,
        raw_response=_response({"id": TASK_ID}),
        logging_obj=logging_obj,  # pyright: ignore[reportArgumentType]  # the transform only reads litellm_params off it
        custom_llm_provider="volcengine",
        request_data=request_data,
    )


def _polled(payload: object, logging_obj: object = None):
    return VolcEngineVideoConfig().transform_video_status_retrieve_response(
        raw_response=_response(payload),
        logging_obj=logging_obj,  # pyright: ignore[reportArgumentType]  # the transform only reads litellm_params off it
        custom_llm_provider="volcengine",
    )


def _task(**fields: object) -> dict[str, object]:
    return {"id": TASK_ID, "model": SEEDANCE_2_5, "status": "succeeded", **fields}


@pytest.mark.parametrize(
    "input_reference",
    [
        PNG_BYTES,
        io.BytesIO(PNG_BYTES),
        ("frame.png", PNG_BYTES, "image/png"),
        ("frame.png", io.BytesIO(PNG_BYTES)),
    ],
    ids=["bytes", "file", "tuple", "tuple-with-file"],
)
def test_a_file_input_reference_becomes_a_data_uri_first_frame(input_reference: object):
    body = _create_body({"input_reference": input_reference, "duration": 4})

    assert body == {
        "model": SEEDANCE_2_5,
        "content": [
            {"type": "text", "text": "a cat on a roof"},
            {"type": "image_url", "image_url": {"url": PNG_DATA_URL}, "role": "first_frame"},
        ],
        "resolution": "720p",
        "duration": 4,
    }


def test_a_path_input_reference_is_read_from_disk(tmp_path: Path):
    frame = tmp_path / "frame.png"
    frame.write_bytes(PNG_BYTES)

    body = _create_body({"input_reference": frame, "duration": 4})

    assert body["content"] == [
        {"type": "text", "text": "a cat on a roof"},
        {"type": "image_url", "image_url": {"url": PNG_DATA_URL}, "role": "first_frame"},
    ]


def test_a_url_input_reference_is_passed_as_is_and_never_opened_as_a_path():
    body = _create_body({"input_reference": "https://x/first.png", "duration": 4})

    assert body["content"] == [
        {"type": "text", "text": "a cat on a roof"},
        {"type": "image_url", "image_url": {"url": "https://x/first.png"}, "role": "first_frame"},
    ]


def test_openais_object_input_reference_sends_its_image_url():
    body = _create_body({"input_reference": {"image_url": "https://x/first.png"}, "duration": 4})

    assert body["content"] == [
        {"type": "text", "text": "a cat on a roof"},
        {"type": "image_url", "image_url": {"url": "https://x/first.png"}, "role": "first_frame"},
    ]


def test_a_file_id_input_reference_is_a_bad_request_unless_drop_params():
    with pytest.raises(UnsupportedParamsError, match="file_id is not supported"):
        _ = _create_body({"input_reference": {"file_id": "file-abc"}, "duration": 4})

    body = _create_body({"input_reference": {"file_id": "file-abc"}, "duration": 4}, drop_params=True)

    assert body["content"] == [{"type": "text", "text": "a cat on a roof"}]


def test_a_caller_content_array_wins_over_input_reference_and_the_prompt_leads():
    body = _create_body(
        {
            "content": [{"type": "image_url", "image_url": {"url": "https://x/a.png"}, "role": "reference_image"}],
            "input_reference": "https://x/ignored.png",
            "duration": 4,
        }
    )

    assert body["content"] == [
        {"type": "text", "text": "a cat on a roof"},
        {"type": "image_url", "image_url": {"url": "https://x/a.png"}, "role": "reference_image"},
    ]


IMAGE_ITEM: Final = {"type": "image_url", "image_url": {"url": "https://x/a.png"}, "role": "reference_image"}


def test_a_text_item_in_content_is_rejected_because_input_guardrails_only_scan_the_prompt():
    with pytest.raises(UnsupportedParamsError, match="put the text in prompt"):
        _ = _create_body({"content": [{"type": "text", "text": "text the guardrail never saw"}], "duration": 4})


def test_a_text_item_in_content_is_dropped_under_drop_params_so_ark_never_sees_unscanned_text():
    body = _create_body(
        {"content": [{"type": "text", "text": "text the guardrail never saw"}, IMAGE_ITEM], "duration": 4},
        drop_params=True,
    )

    assert body["content"] == [{"type": "text", "text": "a cat on a roof"}, IMAGE_ITEM]


@pytest.mark.parametrize(
    "item",
    [
        {"type": "video_url", "video_url": {"url": "https://x/v.mp4"}, "role": "reference_video"},
        {"type": "video_url", "video_url": {"url": "https://x/v.mp4"}},
        {"type": "image_url", "image_url": {"url": "https://x/v.png"}, "role": "reference_video"},
    ],
    ids=["reference-video", "untagged-video", "mistagged-role"],
)
def test_a_video_input_is_rejected_because_its_cost_is_not_known_at_create(item: dict[str, object]):
    with pytest.raises(UnsupportedParamsError, match="Video input"):
        _ = _create_body({"content": [item], "duration": 4})

    body = _create_body({"content": [item, IMAGE_ITEM], "duration": 4}, drop_params=True)

    assert body["content"] == [{"type": "text", "text": "a cat on a roof"}, IMAGE_ITEM]


@pytest.mark.parametrize(
    "content",
    ["a string", [{"type": "image_url", "role": 7}], [{"type": 7}]],
    ids=["not-an-array", "non-string-role", "non-string-type"],
)
def test_a_content_value_that_is_not_an_array_of_items_is_rejected_unless_drop_params(content: object):
    with pytest.raises(UnsupportedParamsError, match="content must be an array"):
        _ = _create_body({"content": content, "duration": 4})

    body = _create_body({"content": content, "duration": 4}, drop_params=True)

    assert body["content"] == [{"type": "text", "text": "a cat on a roof"}]


def test_an_omitted_duration_and_tier_are_sent_as_openais_four_seconds_at_720p_so_ark_cannot_pick_what_is_billed():
    body = _create_body({})

    assert body == {
        "model": SEEDANCE_2_5,
        "content": [{"type": "text", "text": "a cat on a roof"}],
        "resolution": "720p",
        "duration": 4,
    }


@pytest.mark.usefixtures("local_model_cost_map")
def test_an_endpoint_id_priced_through_base_model_gets_the_base_models_default_tier():
    body, _, _ = VolcEngineVideoConfig().transform_video_create_request(
        model="ep-20261015-abc",
        prompt="a cat on a roof",
        api_base=API_BASE,
        video_create_optional_request_params={"duration": 4},
        litellm_params=GenericLiteLLMParams(model_info={"base_model": f"volcengine/{SEEDANCE_2_5}"}),
        headers={},
    )

    assert (body["model"], body["resolution"]) == ("ep-20261015-abc", "720p")


@pytest.mark.parametrize("field", ["return_last_frame", "draft", "frame_count"])
def test_a_field_litellm_does_not_send_to_ark_is_rejected(field: str):
    with pytest.raises(UnsupportedParamsError, match=f"does not send {field} to Ark"):
        _ = _create_body({"duration": 4, field: True})


def test_a_field_litellm_does_not_send_to_ark_is_dropped_under_drop_params():
    body = _create_body({"duration": 4, "return_last_frame": True, "seed": 7}, drop_params=True)

    assert "return_last_frame" not in body
    assert body["seed"] == 7


@pytest.mark.parametrize("duration", [-1, "-1", 0, "abc", 4.5])
def test_a_duration_ark_would_pick_or_that_is_not_whole_seconds_is_rejected(duration: object):
    with pytest.raises(UnsupportedParamsError, match="duration must be a positive whole number"):
        _ = _create_body({"duration": duration})

    assert _create_body({"duration": duration}, drop_params=True)["duration"] == 4


def test_frames_is_rejected_because_it_bypasses_the_billed_duration():
    with pytest.raises(UnsupportedParamsError, match="frames"):
        _ = _create_body({"frames": 121, "duration": 6})

    body = _create_body({"frames": 121, "duration": 6}, drop_params=True)

    assert ("frames" not in body, body["duration"]) == (True, 6)


def test_ark_fields_reach_the_body_and_false_flags_are_kept():
    body = _create_body(
        {
            "duration": 4,
            "resolution": "480p",
            "ratio": "9:16",
            "seed": 7,
            "watermark": False,
            "generate_audio": False,
            "service_tier": "flex",
        }
    )

    assert body == {
        "model": SEEDANCE_2_5,
        "content": [{"type": "text", "text": "a cat on a roof"}],
        "resolution": "480p",
        "ratio": "9:16",
        "seed": 7,
        "watermark": False,
        "generate_audio": False,
        "service_tier": "flex",
        "duration": 4,
    }


@pytest.mark.usefixtures("local_model_cost_map")
def test_a_resolution_the_model_has_no_price_for_is_rejected():
    with pytest.raises(UnsupportedParamsError, match="not offered at resolution '1080p', only at 480p, 720p"):
        _ = _create_body({"duration": 4, "resolution": "1080p"}, model=SEEDANCE_2_0_FAST)


@pytest.mark.usefixtures("local_model_cost_map")
def test_an_unpriced_resolution_is_dropped_under_drop_params():
    body = _create_body(
        {"duration": 4, "resolution": "1080p", "ratio": "16:9"}, model=SEEDANCE_2_0_FAST, drop_params=True
    )

    assert body == {
        "model": SEEDANCE_2_0_FAST,
        "content": [{"type": "text", "text": "a cat on a roof"}],
        "resolution": "720p",
        "ratio": "16:9",
        "duration": 4,
    }


@pytest.mark.usefixtures("local_model_cost_map")
def test_a_model_the_router_registered_without_prices_keeps_its_resolution():
    endpoint_id: Final = "ep-20261015-unpriced"
    _ = Router(
        model_list=[
            {"model_name": "ark", "litellm_params": {"model": f"volcengine/{endpoint_id}", "api_key": "sk-test"}}
        ]
    )

    body = _create_body({"duration": 4, "resolution": "480p"}, model=endpoint_id)
    body_without_a_tier = _create_body({"duration": 4}, model=endpoint_id)

    assert body["resolution"] == "480p"
    assert "resolution" not in body_without_a_tier


def test_a_mistyped_ark_field_is_rejected():
    with pytest.raises(UnsupportedParamsError, match="invalid seed"):
        _ = _create_body({"duration": 4, "seed": "abc", "watermark": False})


def test_a_mistyped_ark_field_is_dropped_under_drop_params():
    body = _create_body({"duration": 4, "seed": "abc", "watermark": False}, drop_params=True)

    assert body == {
        "model": SEEDANCE_2_5,
        "content": [{"type": "text", "text": "a cat on a roof"}],
        "resolution": "720p",
        "watermark": False,
        "duration": 4,
    }


@pytest.mark.parametrize(
    "size,expected",
    [
        ("1920x1080", {"ratio": "16:9", "resolution": "1080p"}),
        ("854x480", {"ratio": "16:9", "resolution": "480p"}),
        ("864x496", {"ratio": "16:9", "resolution": "480p"}),
        ("720x1280", {"ratio": "9:16", "resolution": "720p"}),
        ("1024x768", {"ratio": "4:3", "resolution": "720p"}),
        ("640x640", {"ratio": "1:1", "resolution": "480p"}),
        ("960x960", {"ratio": "1:1", "resolution": "720p"}),
        ("1440x1440", {"ratio": "1:1", "resolution": "1080p"}),
        ("1112x834", {"ratio": "4:3", "resolution": "720p"}),
        ("1470x630", {"ratio": "21:9", "resolution": "720p"}),
        ("3840x2160", {"ratio": "16:9", "resolution": "4k"}),
    ],
)
def test_size_maps_onto_an_ark_ratio_and_tier(size: str, expected: dict[str, str]):
    assert VolcEngineVideoConfig().map_openai_params({"size": size}, SEEDANCE_2_5, False) == expected


def test_explicit_ratio_and_resolution_beat_the_size_derived_ones():
    mapped = VolcEngineVideoConfig().map_openai_params(
        {"size": "1280x720", "seconds": "4", "ratio": "1:1", "resolution": "480p", "parameters": {"seed": 3}},
        SEEDANCE_2_5,
        False,
    )

    assert mapped == {"ratio": "1:1", "resolution": "480p", "duration": 4, "seed": 3}


@pytest.mark.parametrize(
    "params,message",
    [
        ({"seconds": "4.5"}, "seconds must be a whole number"),
        ({"size": "big"}, "size must be WIDTHxHEIGHT"),
        ({"size": "16:9"}, "size must be WIDTHxHEIGHT"),
        ({"parameters": "seed=3"}, "parameters must be an object"),
        ({"characters": [{"id": "c"}]}, "'characters' is not supported"),
        ({"image": "https://x/a.png"}, "'image' is not supported"),
    ],
)
def test_a_value_ark_cannot_take_raises_unless_drop_params(params: dict[str, object], message: str):
    with pytest.raises(UnsupportedParamsError, match=message):
        _ = VolcEngineVideoConfig().map_openai_params(params, SEEDANCE_2_5, False)

    assert VolcEngineVideoConfig().map_openai_params(params, SEEDANCE_2_5, True) == {}


def test_request_metadata_is_not_mapped_into_the_body_and_user_becomes_the_safety_identifier():
    mapped = VolcEngineVideoConfig().map_openai_params(
        {"user": "u", "extra_headers": {"x": "y"}, "seconds": 4}, SEEDANCE_2_5, False
    )

    assert mapped == {"safety_identifier": "u", "duration": 4}


def test_an_explicit_safety_identifier_wins_over_user():
    mapped = VolcEngineVideoConfig().map_openai_params(
        {"user": "u", "safety_identifier": "s", "seconds": 4}, SEEDANCE_2_5, False
    )

    assert mapped["safety_identifier"] == "s"


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(
    "model,request_data,seconds,size,usage",
    [
        (
            SEEDANCE_2_5,
            {"duration": 4, "resolution": "480p", "ratio": "16:9"},
            "4",
            "854x480",
            {"duration_seconds": 4.0, "video_resolution": "480p"},
        ),
        (
            SEEDANCE_2_0,
            {"duration": 5, "resolution": "480p", "ratio": "16:9"},
            "5",
            "864x496",
            {"duration_seconds": 5.0, "video_resolution": "480p"},
        ),
        (SEEDANCE_2_5, {"duration": 5}, "5", None, {"duration_seconds": 5.0}),
    ],
    ids=["2.5-480p", "2.0-480p-renders-its-own-frame", "no-geometry"],
)
def test_the_create_response_reports_what_the_task_bills(
    model: str, request_data: dict[str, object], seconds: str, size: str | None, usage: dict[str, object]
):
    video = _created(request_data, model=model)

    assert video.model_dump(include={"status", "seconds", "size", "usage", "model"}) == {
        "status": "queued",
        "seconds": seconds,
        "size": size,
        "usage": usage,
        "model": model,
    }


@pytest.mark.parametrize(
    "usage,expected_cost",
    [
        ({"duration_seconds": 4.0, "video_resolution": "480p"}, 4 * 0.1),
        ({"duration_seconds": 5.0, "video_resolution": "720p"}, 5 * 0.2),
        ({"duration_seconds": 5.0}, 5 * 0.2),
    ],
)
def test_the_create_usage_bills_its_duration_at_its_tier_rate(usage: dict[str, object], expected_cost: float):
    model_info = {
        "output_cost_per_second": 0.2,
        "output_cost_per_second_480p": 0.1,
        "output_cost_per_second_720p": 0.2,
    }
    video = _created({"duration": int(usage["duration_seconds"]), "resolution": usage.get("video_resolution")})

    cost = video_generation_cost(
        model=SEEDANCE_2_5,
        duration_seconds=video.usage["duration_seconds"],
        model_info=model_info,  # pyright: ignore[reportArgumentType]  # a hand-written rate card stands in for ModelInfo
        video_resolution=video.usage.get("video_resolution"),
    )

    assert cost == pytest.approx(expected_cost)


def test_the_created_id_carries_the_requested_and_served_model_groups():
    logging_obj = Mock(
        litellm_params={"litellm_metadata": {"model_group": "seedance", "deployment_model_name": "seedance-account-b"}}
    )

    video = _created({"duration": 5}, logging_obj=logging_obj)

    assert decode_video_id_with_provider(video.id) == {
        "custom_llm_provider": "volcengine",
        "model_id": "seedance",
        "video_id": TASK_ID,
        "served_model": "seedance-account-b",
    }


def test_a_create_without_router_metadata_carries_the_model():
    video = _created({"duration": 5}, logging_obj=Mock(litellm_params={}))

    assert decode_video_id_with_provider(video.id) == {
        "custom_llm_provider": "volcengine",
        "model_id": SEEDANCE_2_5,
        "video_id": TASK_ID,
    }


@pytest.mark.parametrize(
    "raw_response,status_code,message",
    [
        (_response({"error": {"code": "InvalidParameter", "message": "bad ratio"}}), 400, "bad ratio"),
        (_response({"error": {"code": "Forbidden", "message": "no access"}}, 403), 403, "no access"),
        (_text_response("<html>bad gateway</html>"), 502, "Unexpected Volcengine video response"),
    ],
    ids=["error-on-200", "error-status-kept", "not-json"],
)
def test_a_create_response_without_a_task_id_raises_instead_of_queueing_an_empty_video(
    raw_response: httpx.Response, status_code: int, message: str
):
    with pytest.raises(VolcEngineVideoError, match=message) as raised:
        _ = VolcEngineVideoConfig().transform_video_create_response(
            model=SEEDANCE_2_5,
            raw_response=raw_response,
            logging_obj=None,  # pyright: ignore[reportArgumentType]  # the transform only reads litellm_params off it
            custom_llm_provider="volcengine",
            request_data={"duration": 5},
        )

    assert raised.value.status_code == status_code


@pytest.mark.parametrize(
    "ark_status,expected",
    [
        ("queued", "queued"),
        ("running", "in_progress"),
        ("succeeded", "completed"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
        ("expired", "failed"),
    ],
)
def test_the_ark_status_maps_onto_the_openai_vocabulary(ark_status: str, expected: str):
    assert _polled(_task(status=ark_status)).status == expected


@pytest.mark.usefixtures("local_model_cost_map")
def test_a_completed_task_reports_its_delivered_geometry_duration_and_tokens():
    video = _polled(
        _task(
            created_at=1791989938,
            updated_at=1791990031,
            resolution="480p",
            ratio="16:9",
            duration=4,
            usage={"completion_tokens": 40594, "total_tokens": 40594},
            model=SEEDANCE_2_0,
            content={"video_url": "https://tos/v.mp4"},
        )
    )

    assert video.model_dump(exclude={"id"}) | {"id": decode_video_id_with_provider(video.id)["video_id"]} == {
        "id": TASK_ID,
        "object": "video",
        "status": "completed",
        "created_at": 1791989938,
        "completed_at": 1791990031,
        "expires_at": None,
        "error": None,
        "progress": None,
        "remixed_from_video_id": None,
        "seconds": "4",
        "size": "864x496",
        "model": SEEDANCE_2_0,
        "usage": {"duration_seconds": 4.0, "video_resolution": "480p", "total_tokens": 40594},
    }


@pytest.mark.parametrize(
    "resolution,ratio,expected",
    [
        ("720p", "9:16", "720x1280"),
        ("1080p", "16:9", "1920x1080"),
        ("480p", "adaptive", None),
    ],
)
def test_a_frame_the_cost_map_does_not_list_is_scaled_from_the_tier(resolution: str, ratio: str, expected: str | None):
    assert _polled(_task(resolution=resolution, ratio=ratio, model="doubao-seedance-unlisted")).size == expected


def test_a_failed_task_carries_its_error_block():
    video = _polled(_task(status="failed", error={"code": "OutputVideoSensitiveContentDetected", "message": "flagged"}))

    assert (video.status, video.error, video.completed_at) == (
        "failed",
        {"code": "OutputVideoSensitiveContentDetected", "message": "flagged"},
        None,
    )


@pytest.mark.parametrize(
    "payload,message",
    [
        (_task(status="paused"), "Unknown Volcengine task status: paused"),
        ({"status": "running"}, "Unexpected Volcengine video response"),
        ({"id": TASK_ID}, "Unexpected Volcengine video response"),
    ],
    ids=["unmapped-status", "no-id", "no-status"],
)
def test_a_poll_litellm_cannot_read_raises_instead_of_returning_a_job_to_poll_forever(payload: object, message: str):
    with pytest.raises(VolcEngineVideoError, match=message) as raised:
        _ = _polled(payload)

    assert raised.value.status_code == 502


def test_a_poll_keeps_the_ids_requested_and_served_model_when_the_router_served_it():
    video_id = encode_video_id_with_provider(TASK_ID, "volcengine", "seedance", served_model="seedance-account-b")
    logging_obj = Mock(
        litellm_params={
            "video_id": video_id,
            "litellm_metadata": {"model_group": "seedance", "deployment_model_name": "seedance-account-b"},
        }
    )

    video = _polled(_task(status="running"), logging_obj=logging_obj)

    assert video.id == video_id
    assert video.model == "seedance"


def test_a_poll_drops_a_served_model_the_router_did_not_route_to():
    edited = encode_video_id_with_provider(
        TASK_ID, "volcengine", "seedance-account-a", served_model="seedance-account-b"
    )
    logging_obj = Mock(litellm_params={"video_id": edited, "litellm_metadata": {"model_group": "seedance-account-a"}})

    video = _polled(_task(status="running"), logging_obj=logging_obj)

    assert video.id == encode_video_id_with_provider(TASK_ID, "volcengine", "seedance-account-a")


async def _download(task_payload: dict[str, object], video_bytes: bytes = b"mp4") -> bytes:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.host == "tos.example":
            return httpx.Response(200, content=video_bytes)
        return httpx.Response(200, json=task_payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        handler: Final = AsyncHTTPHandler()
        await handler.close()
        handler.client = http_client
        return await avideo_content(
            video_id=TASK_ID,
            custom_llm_provider="volcengine",
            model=f"volcengine/{SEEDANCE_2_5}",
            api_key="sk-test",
            api_base=API_BASE,
            client=handler,
        )


@pytest.mark.asyncio
async def test_the_download_fetches_the_signed_url_through_the_callers_client():
    content = await _download(_task(content={"video_url": "https://tos.example/v.mp4"}), video_bytes=b"the-video")

    assert content == b"the-video"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,message",
    [
        (_task(status="queued"), "still processing"),
        (_task(status="running"), "still processing"),
        (_task(status="failed", error={"message": "flagged"}), "flagged"),
        (_task(status="failed"), "ended as failed"),
        (_task(status="cancelled"), "ended as cancelled"),
        (_task(status="expired"), "Task expired"),
    ],
)
async def test_a_download_with_no_video_yet_is_a_bad_request_so_it_is_neither_retried_nor_cools_the_deployment(
    payload: dict[str, object], message: str
):
    with pytest.raises(litellm.BadRequestError, match=message) as raised:
        _ = await _download(payload)

    assert raised.value.status_code == 400


def test_the_list_maps_ark_paging_and_filters():
    url, params = VolcEngineVideoConfig().transform_video_list_request(
        api_base=API_BASE,
        litellm_params=GenericLiteLLMParams(),
        headers={},
        limit=20,
        extra_query={"filter.status": "succeeded"},
    )

    assert (url, params) == (TASKS_URL, {"page_size": "20", "filter.status": "succeeded"})


@pytest.mark.parametrize("paging", [{"after": "video_abc"}, {"order": "asc"}])
def test_a_list_cursor_or_order_is_rejected_unless_drop_params(paging: dict[str, str]):
    config = VolcEngineVideoConfig()

    with pytest.raises(UnsupportedParamsError, match="pages by page_num"):
        _ = config.transform_video_list_request(
            api_base=API_BASE, litellm_params=GenericLiteLLMParams(), headers={}, **paging
        )
    _, params = config.transform_video_list_request(
        api_base=API_BASE, litellm_params=GenericLiteLLMParams(drop_params=True), headers={}, limit=5, **paging
    )

    assert params == {"page_size": "5"}


def test_the_list_response_is_an_openai_list_body():
    response = VolcEngineVideoConfig().transform_video_list_response(
        raw_response=_response({"total": 1, "items": [_task(status="running")]}),
        logging_obj=Mock(),  # pyright: ignore[reportArgumentType]  # unused by the transform
        custom_llm_provider="volcengine",
    )

    encoded = encode_video_id_with_provider(TASK_ID, "volcengine", SEEDANCE_2_5)
    assert {key: value for key, value in response.items() if key != "data"} == {
        "object": "list",
        "total": 1,
        "has_more": False,
        "first_id": encoded,
        "last_id": encoded,
    }
    assert [(item["id"], item["status"]) for item in response["data"]] == [(encoded, "in_progress")]


def test_an_empty_list_still_carries_the_object_type():
    response = VolcEngineVideoConfig().transform_video_list_response(
        raw_response=_response({"total": 0, "items": []}),
        logging_obj=Mock(),  # pyright: ignore[reportArgumentType]  # unused by the transform
    )

    assert response == {"object": "list", "data": [], "total": 0, "has_more": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("call", ["avideo_edit", "avideo_extension", "avideo_remix"])
async def test_edit_extension_and_remix_are_a_bad_request_because_a_video_input_cannot_be_billed_at_create(call: str):
    with pytest.raises(litellm.BadRequestError, match="Ark bills a task with a video input") as raised:
        _ = await getattr(litellm, call)(
            video_id="https://x/source.mp4",
            prompt="make it brighter",
            seconds="5",
            model=f"volcengine/{SEEDANCE_2_5}",
            custom_llm_provider="volcengine",
            api_key="sk-test",
            api_base=API_BASE,
        )

    assert raised.value.status_code == 400


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
def test_any_ark_surface_on_the_shared_credential_normalizes_to_the_host_root(
    api_base: str | None, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("VOLCENGINE_API_BASE", raising=False)
    monkeypatch.delenv("ARK_API_BASE", raising=False)

    assert VolcEngineVideoConfig().get_complete_url("m", api_base, {}) == API_BASE


@pytest.mark.parametrize("env_var", ["VOLCENGINE_API_BASE", "ARK_API_BASE"])
def test_an_api_base_env_var_routes_a_deployment_without_api_base(env_var: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VOLCENGINE_API_BASE", raising=False)
    monkeypatch.delenv("ARK_API_BASE", raising=False)
    monkeypatch.setenv(env_var, "https://ark.ap-southeast.bytepluses.com/api/v3")

    assert VolcEngineVideoConfig().get_complete_url("m", None, {}) == "https://ark.ap-southeast.bytepluses.com"
    assert VolcEngineVideoConfig().get_complete_url("m", f"{API_BASE}/api/v3", {}) == API_BASE


@pytest.mark.parametrize(
    "api_base,expected",
    [
        ("https://gw.internal/ark/api/v3", "https://gw.internal/ark"),
        ("https://gw.internal/api/ark/api/v3", "https://gw.internal/api/ark"),
    ],
)
def test_a_path_prefix_before_the_api_marker_survives(api_base: str, expected: str):
    assert VolcEngineVideoConfig().get_complete_url("m", api_base, {}) == expected


def test_a_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Volcengine API key is required"):
        _ = VolcEngineVideoConfig().validate_environment({}, SEEDANCE_2_5)


def test_volcengine_resolves_to_the_video_config():
    config = ProviderConfigManager.get_provider_video_config(
        model=SEEDANCE_2_5, provider=litellm.LlmProviders.VOLCENGINE
    )

    assert type(config) is VolcEngineVideoConfig


ARK_VIDEO_TIERS: Final = {
    SEEDANCE_2_5: {"480p": ((854, 480), 70), "720p": ((1280, 720), 70), "1080p": ((1920, 1080), 77)},
    SEEDANCE_2_0: {"480p": ((864, 496), 46), "720p": ((1280, 720), 46), "1080p": ((1920, 1080), 51), "4k": ((3840, 2160), 26)},
    SEEDANCE_2_0_FAST: {"480p": ((864, 496), 37), "720p": ((1280, 720), 37)},
    "doubao-seedance-2-0-mini-260615": {"480p": ((864, 496), 23), "720p": ((1280, 720), 23)},
}
CNY_PER_USD: Final = 7.0


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("model", sorted(ARK_VIDEO_TIERS))
def test_each_tier_rate_is_its_rendered_pixels_at_arks_price_for_that_tier(model: str):
    """Ark prices 1080p and 4k per token above and below the 480p/720p rate, so a tier is not the base
    rate scaled by pixels. Prices and frames are Ark's published per-resolution table, read 2026-10-16
    from https://www.volcengine.com/docs/82379/1544106, whose own 2.0 examples agree: 1080p 16:9 5s bills
    2.48 CNY/s and 4k bills 5.05 CNY/s."""
    info = litellm.get_model_info(model=model, custom_llm_provider="volcengine")
    frames = info["provider_specific_entry"]["volcengine_video_frames"]

    assert {tier: info[f"output_cost_per_second_{tier}"] for tier in ARK_VIDEO_TIERS[model]} == {
        tier: pytest.approx(width * height * 24 / 1024 * cny_per_million / 1e6 / CNY_PER_USD, rel=1e-5)
        for tier, ((width, height), cny_per_million) in ARK_VIDEO_TIERS[model].items()
    }

    assert {
        tier: tuple(int(side) for side in frames[tier]["16:9"].split("x"))
        for tier in ARK_VIDEO_TIERS[model]
        if tier in frames
    } == {tier: frame for tier, (frame, _) in ARK_VIDEO_TIERS[model].items() if tier in frames}


async def _wire_body(**kwargs: object) -> dict[str, object]:
    sent: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"id": TASK_ID})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        handler: Final = AsyncHTTPHandler()
        await handler.close()
        handler.client = http_client
        _ = await avideo_generation(api_key="sk-test", api_base=API_BASE, client=handler, **kwargs)

    (request,) = sent
    assert (str(request.url), request.headers["authorization"]) == (TASKS_URL, "Bearer sk-test")
    return json.loads(request.content)


@pytest.mark.asyncio
async def test_seconds_size_and_extra_body_content_reach_the_wire():
    body = await _wire_body(
        model=f"volcengine/{SEEDANCE_2_5}",
        prompt="use these",
        seconds="4",
        size="854x480",
        extra_body={
            "content": [{"type": "image_url", "image_url": {"url": "https://x/a.png"}, "role": "reference_image"}]
        },
    )

    assert body == {
        "model": SEEDANCE_2_5,
        "content": [
            {"type": "text", "text": "use these"},
            {"type": "image_url", "image_url": {"url": "https://x/a.png"}, "role": "reference_image"},
        ],
        "ratio": "16:9",
        "resolution": "480p",
        "duration": 4,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [{"parameters": {"seed": 3}}, {"extra_body": {"parameters": {"seed": 3}}}],
    ids=["top_level", "extra_body"],
)
async def test_parameters_reach_the_wire_unpacked_whether_sent_top_level_or_in_extra_body(fields: dict[str, object]):
    body = await _wire_body(model=f"volcengine/{SEEDANCE_2_5}", prompt="p", seconds="4", **fields)

    assert body == {
        "model": SEEDANCE_2_5,
        "content": [{"type": "text", "text": "p"}],
        "resolution": "720p",
        "duration": 4,
        "seed": 3,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_body,expected",
    [
        ({"size": "1920x1080"}, {"resolution": "1080p"}),
        ({"seconds": "8"}, {"duration": 8}),
        ({"user": "someone"}, {"safety_identifier": "someone"}),
    ],
    ids=["size", "seconds", "user"],
)
async def test_an_extra_body_field_litellm_maps_itself_is_honoured_and_its_leftover_copy_ignored(
    extra_body: dict[str, object], expected: dict[str, object]
):
    """The shared request builder copies extra_body onto the top-level params, where this adapter maps it, and then
    merges the raw extra_body back over the mapped params, returning the same key a second time as a field of its
    own. The second copy has to be ignored rather than rejected as a field litellm never sends."""
    body = await _wire_body(
        model=f"volcengine/{SEEDANCE_2_5}",
        prompt="p",
        seconds="4",
        size="1280x720",
        extra_body=extra_body,
    )

    assert body == {
        "model": SEEDANCE_2_5,
        "content": [{"type": "text", "text": "p"}],
        "ratio": "16:9",
        "resolution": "720p",
        "duration": 4,
        **expected,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields,misplaced",
    [
        ({"parameters": {"seconds": "8"}}, "seconds"),
        ({"extra_body": {"parameters": {"seconds": "8"}}}, "seconds"),
        ({"extra_body": {"parameters": {"size": "1920x1080"}}}, "size"),
        ({"extra_body": {"parameters": {"user": "someone"}}}, "user"),
        ({"extra_body": {"parameters": {"prompt": "a dog"}}}, "prompt"),
        ({"extra_body": {"parameters": {"parameters": {"seed": 3}}}}, "parameters"),
    ],
    ids=["top_level_seconds", "seconds", "size", "user", "prompt", "nested_parameters"],
)
async def test_a_request_field_litellm_maps_itself_is_rejected_inside_parameters(
    fields: dict[str, object], misplaced: str
):
    with pytest.raises(UnsupportedParamsError, match=f"send {misplaced} as top-level request fields"):
        _ = await _wire_body(model=f"volcengine/{SEEDANCE_2_5}", prompt="p", **fields)


@pytest.mark.asyncio
async def test_a_request_field_inside_parameters_is_dropped_under_drop_params_and_the_rest_is_sent():
    body = await _wire_body(
        model=f"volcengine/{SEEDANCE_2_5}",
        prompt="p",
        drop_params=True,
        extra_body={"parameters": {"seconds": "8", "seed": 3}},
    )

    assert body == {
        "model": SEEDANCE_2_5,
        "content": [{"type": "text", "text": "p"}],
        "resolution": "720p",
        "duration": 4,
        "seed": 3,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_body,field",
    [
        ({"prompt": "a dog"}, "prompt"),
        ({"model": SEEDANCE_2_0_FAST}, "model"),
        ({"extra_headers": {"x-a": "1"}}, "extra_headers"),
    ],
    ids=["prompt", "model", "extra_headers"],
)
async def test_an_extra_body_copy_of_a_request_field_litellm_does_not_map_is_rejected(
    extra_body: dict[str, object], field: str
):
    with pytest.raises(UnsupportedParamsError, match=f"does not send {field} to Ark"):
        _ = await _wire_body(model=f"volcengine/{SEEDANCE_2_5}", prompt="p", extra_body=extra_body)


@pytest.mark.asyncio
async def test_a_user_that_is_not_a_string_is_rejected_rather_than_ignored():
    with pytest.raises(UnsupportedParamsError, match="user must be a string, got int"):
        _ = await _wire_body(model=f"volcengine/{SEEDANCE_2_5}", prompt="p", extra_body={"user": 5})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "control",
    [{"drop_params": False}, {"organization": "org-1"}],
    ids=["drop_params", "organization"],
)
async def test_a_litellm_control_field_is_not_mistaken_for_an_ark_field(control: dict[str, object]):
    body = await _wire_body(model=f"volcengine/{SEEDANCE_2_5}", prompt="p", seconds="4", **control)

    assert body == {
        "model": SEEDANCE_2_5,
        "content": [{"type": "text", "text": "p"}],
        "resolution": "720p",
        "duration": 4,
    }


@pytest.mark.asyncio
async def test_additional_drop_params_removes_the_named_ark_fields_from_the_wire():
    body = await _wire_body(
        model=f"volcengine/{SEEDANCE_2_5}",
        prompt="p",
        seconds="4",
        additional_drop_params=["seed"],
        extra_body={"seed": 3, "watermark": False},
    )

    assert body == {
        "model": SEEDANCE_2_5,
        "content": [{"type": "text", "text": "p"}],
        "resolution": "720p",
        "duration": 4,
        "watermark": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dropped", "request_fields"),
    [
        pytest.param("user", {"user": "u-1"}, id="user"),
        pytest.param("seconds", {"seconds": "8"}, id="seconds"),
        pytest.param("size", {"size": "1920x1080"}, id="size"),
        pytest.param("parameters", {"parameters": {"seed": 3}}, id="parameters"),
        pytest.param("parameters", {"extra_body": {"parameters": {"seed": 3}}}, id="extra_body_parameters"),
        pytest.param("image", {"image": {"gcsUri": "gs://b/a.png"}}, id="unsupported_image"),
    ],
)
async def test_additional_drop_params_naming_a_request_field_sends_what_omitting_it_sends(
    dropped: str, request_fields: dict[str, object]
):
    request: Final = {"model": f"volcengine/{SEEDANCE_2_5}", "prompt": "p"}

    with_drop = await _wire_body(**request, **request_fields, additional_drop_params=[dropped])

    assert with_drop == await _wire_body(**request)


@pytest.mark.asyncio
async def test_an_extra_body_duration_of_minus_one_is_rejected_before_anything_is_sent():
    with pytest.raises(UnsupportedParamsError, match="Ark-picked length"):
        _ = await _wire_body(model=f"volcengine/{SEEDANCE_2_5}", prompt="p", extra_body={"duration": -1})


ACCOUNT_A_HOST: Final = "https://ark-a.example"
ACCOUNT_B_HOST: Final = "https://ark-b.example"


def _two_account_router(**router_kwargs: object) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "seedance-a",
                "litellm_params": {
                    "model": f"volcengine/{SEEDANCE_2_5}",
                    "api_key": "key-a",
                    "api_base": ACCOUNT_A_HOST,
                },
            },
            {
                "model_name": "seedance-b",
                "litellm_params": {
                    "model": f"volcengine/{SEEDANCE_2_5}",
                    "api_key": "key-b",
                    "api_base": ACCOUNT_B_HOST,
                },
            },
        ],
        **router_kwargs,  # pyright: ignore[reportArgumentType]  # forwarded Router options
    )


def _proxy_poll_model(router: Router, video_id: str) -> str | None:
    decoded: Final = decode_video_id_with_provider(video_id)
    requested: Final = router.resolve_model_name_from_model_id(
        decoded.get("model_id"), decoded.get("custom_llm_provider")
    )
    served: Final = decoded.get("served_model")
    return router.resolve_served_model_name(requested, served) if requested and served else requested


@pytest.mark.parametrize("caller_metadata", [None, {"tags": ["proxy-request"]}])
@respx.mock
def test_a_video_is_polled_on_the_ark_account_that_created_it(caller_metadata: dict[str, object] | None):
    router = _two_account_router()
    _ = respx.post(f"{ACCOUNT_B_HOST}/api/v3/contents/generations/tasks").mock(
        return_value=httpx.Response(200, json={"id": TASK_ID})
    )
    task_get = respx.get(f"{ACCOUNT_B_HOST}/api/v3/contents/generations/tasks/{TASK_ID}").mock(
        return_value=httpx.Response(200, json=_task(status="running"))
    )

    created = router.video_generation(model="seedance-b", prompt="a cat", seconds="4", metadata=caller_metadata)
    _ = router.video_status(model=_proxy_poll_model(router, created.id), video_id=created.id)

    assert [call.request.headers["authorization"] for call in task_get.calls] == ["Bearer key-b"]


@respx.mock
def test_a_video_created_through_a_routing_group_is_polled_on_the_member_that_created_it():
    router = _two_account_router(
        routing_groups=[
            {"group_name": "seedance", "models": ["seedance-a", "seedance-b"], "routing_strategy": "simple-shuffle"}
        ]
    )
    for host in (ACCOUNT_A_HOST, ACCOUNT_B_HOST):
        _ = respx.post(f"{host}/api/v3/contents/generations/tasks").mock(
            return_value=httpx.Response(200, json={"id": TASK_ID})
        )
        _ = respx.get(f"{host}/api/v3/contents/generations/tasks/{TASK_ID}").mock(
            return_value=httpx.Response(200, json=_task(status="running"))
        )

    for _ in range(8):
        created = router.video_generation(model="seedance", prompt="a cat", seconds="4")
        polled = router.video_status(model=_proxy_poll_model(router, created.id), video_id=created.id)
        assert polled.id == created.id

    hosts = [call.request.url.host for call in respx.calls]
    assert hosts[1::2] == hosts[0::2]
