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


@pytest.mark.parametrize("drop_params", [None, True])
def test_a_text_item_in_content_is_rejected_because_input_guardrails_only_scan_the_prompt(drop_params: bool | None):
    with pytest.raises(UnsupportedParamsError, match="put the text in prompt"):
        _ = _create_body(
            {"content": [{"type": "text", "text": "text the guardrail never saw"}], "duration": 4},
            drop_params=drop_params,
        )


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
        _ = _create_body({"content": [item], "duration": 4}, drop_params=True)


def test_a_content_value_that_is_not_an_array_of_items_is_rejected():
    with pytest.raises(UnsupportedParamsError, match="content must be an array"):
        _ = _create_body({"content": "a string", "duration": 4})


def test_an_omitted_duration_is_sent_as_five_seconds_so_ark_cannot_pick_the_length():
    body = _create_body({})

    assert body == {"model": SEEDANCE_2_5, "content": [{"type": "text", "text": "a cat on a roof"}], "duration": 5}


@pytest.mark.parametrize("duration", [-1, "-1", 0, "abc", 4.5])
def test_a_duration_ark_would_pick_or_that_is_not_whole_seconds_is_rejected(duration: object):
    with pytest.raises(UnsupportedParamsError, match="duration must be a positive whole number"):
        _ = _create_body({"duration": duration}, drop_params=True)


def test_frames_is_rejected_because_it_bypasses_the_billed_duration():
    with pytest.raises(UnsupportedParamsError, match="frames"):
        _ = _create_body({"frames": 121, "duration": 4})


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
            "litellm_internal": "not for ark",
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

    assert body["resolution"] == "480p"


def test_a_mistyped_ark_field_is_rejected():
    with pytest.raises(UnsupportedParamsError, match="invalid seed"):
        _ = _create_body({"duration": 4, "seed": "abc", "watermark": False})


def test_a_mistyped_ark_field_is_dropped_under_drop_params():
    body = _create_body({"duration": 4, "seed": "abc", "watermark": False}, drop_params=True)

    assert body == {
        "model": SEEDANCE_2_5,
        "content": [{"type": "text", "text": "a cat on a roof"}],
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
        ("960x960", {"ratio": "1:1", "resolution": "1080p"}),
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


def test_request_metadata_is_not_mapped_into_the_body():
    mapped = VolcEngineVideoConfig().map_openai_params(
        {"user": "u", "extra_headers": {"x": "y"}, "seconds": 4}, SEEDANCE_2_5, False
    )

    assert mapped == {"duration": 4}


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(
    "model,request_data,seconds,size,usage",
    [
        (
            SEEDANCE_2_5,
            {"duration": 4, "resolution": "480p", "ratio": "16:9"},
            "4.0",
            "854x480",
            {"duration_seconds": 4.0, "video_resolution": "480p"},
        ),
        (
            SEEDANCE_2_0,
            {"duration": 5, "resolution": "480p", "ratio": "16:9"},
            "5.0",
            "864x496",
            {"duration_seconds": 5.0, "video_resolution": "480p"},
        ),
        (SEEDANCE_2_5, {"duration": 5}, "5.0", None, {"duration_seconds": 5.0}),
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
        "seconds": "4.0",
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
        "first_id": encoded,
        "last_id": encoded,
    }
    assert [(item["id"], item["status"]) for item in response["data"]] == [(encoded, "in_progress")]


def test_an_empty_list_still_carries_the_object_type():
    response = VolcEngineVideoConfig().transform_video_list_response(
        raw_response=_response({"total": 0, "items": []}),
        logging_obj=Mock(),  # pyright: ignore[reportArgumentType]  # unused by the transform
    )

    assert response == {"object": "list", "data": [], "total": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("call", ["avideo_edit", "avideo_extension"])
async def test_edit_and_extension_are_a_bad_request_because_a_video_input_cannot_be_billed_at_create(call: str):
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

    assert VolcEngineVideoConfig().get_complete_url("m", api_base, {}) == API_BASE


def test_the_volcengine_api_base_env_var_routes_a_deployment_without_api_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VOLCENGINE_API_BASE", "https://ark.ap-southeast.bytepluses.com/api/v3")

    assert VolcEngineVideoConfig().get_complete_url("m", None, {}) == "https://ark.ap-southeast.bytepluses.com"
    assert VolcEngineVideoConfig().get_complete_url("m", f"{API_BASE}/api/v3", {}) == API_BASE


def test_a_path_prefix_before_the_api_marker_survives():
    assert (
        VolcEngineVideoConfig().get_complete_url("m", "https://gw.internal/ark/api/v3", {}) == "https://gw.internal/ark"
    )


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


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(
    "model,tiers",
    [
        (SEEDANCE_2_5, {"480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080)}),
        (SEEDANCE_2_0, {"480p": (864, 496), "720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}),
        (SEEDANCE_2_0_FAST, {"480p": (864, 496), "720p": (1280, 720)}),
        ("doubao-seedance-2-0-mini-260615", {"480p": (864, 496), "720p": (1280, 720)}),
    ],
)
def test_each_tier_rate_is_the_base_rate_scaled_by_its_rendered_pixels(model: str, tiers: dict[str, tuple[int, int]]):
    info = litellm.get_model_info(model=model, custom_llm_provider="volcengine")
    base_rate: Final = info["output_cost_per_second"]

    assert {tier: info[f"output_cost_per_second_{tier}"] for tier in tiers} == {
        tier: pytest.approx(base_rate * width * height / (1280 * 720), rel=1e-5)
        for tier, (width, height) in tiers.items()
    }


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("model", [SEEDANCE_2_5, SEEDANCE_2_0, SEEDANCE_2_0_FAST, "doubao-seedance-2-0-mini-260615"])
def test_the_16_9_frame_listed_for_each_tier_is_the_one_its_rate_was_derived_from(model: str):
    info = litellm.get_model_info(model=model, custom_llm_provider="volcengine")
    frames = info["provider_specific_entry"]["volcengine_video_frames"]
    base_rate: Final = info["output_cost_per_second"]

    for tier, by_ratio in frames.items():
        width, height = (int(side) for side in by_ratio["16:9"].split("x"))
        assert info[f"output_cost_per_second_{tier}"] == pytest.approx(
            base_rate * width * height / (1280 * 720), rel=1e-5
        ), tier


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
