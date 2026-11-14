import base64
import json
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, object_value
from integration._support.wire import Reply, Request, wire_server
from integration.providers._dashscope_video_support import (
    API_KEY,
    CREATE_PATH,
    FINISHED_EPOCH,
    MODEL,
    MP4,
    OSS_PATH,
    PROMPT,
    SUBMITTED_EPOCH,
    TASKS_PATH,
    async_openai_client,
    create,
    create_bodies,
    created_id,
    deployed,
    error_message,
    json_reply,
    openai_client,
    pending,
    rate,
    settled_spend,
    spend_stays,
    task_api,
    task_of,
    unknown,
)
from pydantic import JsonValue

_PNG: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_HUGE: Final = "9" * 5120
_720P: Final = "output_cost_per_second_720p"
_1080P: Final = "output_cost_per_second"


def _assert_task_wire(requests: tuple[Request, ...], task_id: str, parameters: Mapping[str, JsonValue]) -> None:
    create_request, poll, content_poll, download = requests
    assert (create_request.method, create_request.target) == ("POST", CREATE_PATH)
    assert create_request.headers["authorization"] == f"Bearer {API_KEY}"
    assert create_request.headers["x-dashscope-async"] == "enable"
    assert create_request.headers["content-type"] == "application/json"
    assert json.loads(create_request.body) == {"model": MODEL, "input": {"prompt": PROMPT}, "parameters": parameters}
    for request in (poll, content_poll):
        assert (request.method, request.target) == ("GET", f"{TASKS_PATH}{task_id}")
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert (download.method, download.target) == ("GET", f"{OSS_PATH}{task_id}.mp4")
    assert "authorization" not in download.headers


def _assert_completed(status: Mapping[str, JsonValue], video_id: str) -> None:
    assert status["id"] == video_id, status
    assert status["status"] == "completed", status
    assert status["created_at"] == SUBMITTED_EPOCH, status
    assert status["completed_at"] == FINISHED_EPOCH, status
    assert status["size"] == "1280x720", status
    assert status["seconds"] == "5.0", status


def _assert_billed_once_at_720p(key: str, provider: str) -> None:
    assert rate(provider, _720P) != rate(provider, _1080P)
    settled_spend(key, 5 * rate(provider, _720P))


def test_dashscope_openai_sdk_sync_create_poll_download_bills_the_720p_tier_once(gateway: Gateway) -> None:
    with wire_server(task_api()) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        key: Final = scenario.key(models=[model])
        client: Final = openai_client(gateway, key)
        video: Final = client.videos.create(
            model=model,
            prompt=PROMPT,
            seconds="5",  # pyright: ignore[reportArgumentType]  # DashScope takes seconds the OpenAI literal type omits
            size="1280x720",
        )
        assert video.status == "queued", video
        _assert_billed_once_at_720p(key, "dashscope")
        status: Final = client.videos.retrieve(video.id)
        _assert_completed(status.model_dump(), video.id)
        content: Final = client.videos.download_content(video.id)
        assert content.content == MP4
        _assert_task_wire(wire.drain(), task_of(video.id), {"ratio": "16:9", "resolution": "720P", "duration": 5})
        spend_stays(key, 5 * rate("dashscope", _720P))


async def test_qwencloud_openai_sdk_async_create_poll_download_bills_the_qwencloud_720p_tier_once(
    gateway: Gateway,
) -> None:
    with wire_server(task_api()) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"qwencloud/{MODEL}", api_base=wire.url, api_key=API_KEY)
        key: Final = scenario.key(models=[model])
        client: Final = async_openai_client(gateway, key)
        video: Final = await client.videos.create(
            model=model,
            prompt=PROMPT,
            seconds="5",  # pyright: ignore[reportArgumentType]  # DashScope takes seconds the OpenAI literal type omits
            size="1280x720",
        )
        assert video.status == "queued", video
        assert rate("qwencloud", _720P) != rate("dashscope", _720P)
        _assert_billed_once_at_720p(key, "qwencloud")
        status: Final = await client.videos.retrieve(video.id)
        _assert_completed(status.model_dump(), video.id)
        content: Final = await client.videos.download_content(video.id)
        assert content.content == MP4
        _assert_task_wire(wire.drain(), task_of(video.id), {"ratio": "16:9", "resolution": "720P", "duration": 5})
        spend_stays(key, 5 * rate("qwencloud", _720P))


def test_qwen_ai_platform_raw_http_create_poll_download_bills_the_720p_tier_once(gateway: Gateway) -> None:
    with wire_server(task_api()) as wire, gateway.scenario() as scenario:
        model: Final = deployed(
            gateway, scenario, model=f"qwen_ai_platform/{MODEL}", api_base=wire.url, api_key=API_KEY
        )
        key: Final = scenario.key(models=[model])
        video_id: Final = created_id(create(gateway, {"model": model, "seconds": 5, "size": "1280x720"}, key))
        _assert_billed_once_at_720p(key, "qwen_ai_platform")
        status: Final = gateway.request("GET", f"/v1/videos/{video_id}", key=key)
        assert status.status_code == 200, status.text
        _assert_completed(JSON_OBJECT.validate_json(status.content), video_id)
        content: Final = gateway.request("GET", f"/v1/videos/{video_id}/content", key=key)
        assert content.status_code == 200, content.text
        assert content.headers["content-type"].startswith("video/mp4")
        assert content.content == MP4
        _assert_task_wire(wire.drain(), task_of(video_id), {"ratio": "16:9", "resolution": "720P", "duration": 5})
        expected: Final = 5 * rate("qwen_ai_platform", _720P)
        spend_stays(key, expected)


def test_multipart_input_reference_reaches_dashscope_as_a_first_frame_media_entry(gateway: Gateway) -> None:
    with wire_server(task_api()) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        response: Final = gateway.request_multipart(
            "/v1/videos",
            {"model": model, "prompt": PROMPT, "seconds": "5", "size": "1280x720"},
            {"input_reference": ("frame.png", _PNG, "image/png")},
        )
        created_id(response)
        (body,) = create_bodies(wire.drain())
        assert body == {
            "model": MODEL,
            "input": {
                "prompt": PROMPT,
                "media": [{"type": "first_frame", "url": "data:image/png;base64," + base64.b64encode(_PNG).decode()}],
            },
            "parameters": {"ratio": "16:9", "resolution": "720P", "duration": 5},
        }


def test_caller_parameters_reach_dashscope_as_written_beside_the_mapped_size_and_seconds(gateway: Gateway) -> None:
    with wire_server(task_api()) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        created_id(
            create(
                gateway,
                {
                    "model": model,
                    "seconds": "8",
                    "size": "720x1280",
                    "negative_prompt": "blurry, low contrast",
                    "parameters": {"prompt_extend": False, "watermark": False, "seed": 42, "shot_type": "multi"},
                },
            )
        )
        (body,) = create_bodies(wire.drain())
        assert body == {
            "model": MODEL,
            "input": {"prompt": PROMPT, "negative_prompt": "blurry, low contrast"},
            "parameters": {
                "ratio": "9:16",
                "resolution": "720P",
                "duration": 8,
                "prompt_extend": False,
                "watermark": False,
                "seed": 42,
                "shot_type": "multi",
            },
        }


def test_omitted_seconds_and_size_bill_dashscopes_five_second_1080p_default(gateway: Gateway) -> None:
    with wire_server(task_api()) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        key: Final = scenario.key(models=[model])
        created_id(create(gateway, {"model": model}, key))
        (body,) = create_bodies(wire.drain())
        assert body == {"model": MODEL, "input": {"prompt": PROMPT}}
        settled_spend(key, 5 * rate("dashscope", _1080P))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seconds", "10s", "seconds must be a whole number of seconds"),
        ("seconds", "", "seconds must be a whole number of seconds"),
        ("seconds", _HUGE, "seconds must be a whole number of seconds"),
        ("seconds", ["5"], "seconds must be a whole number of seconds"),
        ("seconds", True, "seconds must be a whole number of seconds"),
        ("seconds", 2.5, "seconds must be a whole number of seconds"),
        ("size", "abc", "size must be WIDTHxHEIGHT or a WIDTH:HEIGHT ratio"),
        ("size", _HUGE, "size must be WIDTHxHEIGHT or a WIDTH:HEIGHT ratio"),
        ("size", 1280, "size must be WIDTHxHEIGHT or a WIDTH:HEIGHT ratio"),
        ("size", "0x720", "size must be WIDTHxHEIGHT or a WIDTH:HEIGHT ratio"),
        ("parameters", "seed=1", "parameters must be an object of DashScope request fields"),
        ("parameters", [1, 2], "parameters must be an object of DashScope request fields"),
        ("parameters", {"seed": "abc"}, "invalid seed"),
        ("parameters", {"seed": -1}, "invalid seed"),
        ("parameters", {"audio": "yes"}, "invalid audio"),
        ("parameters", {"duration": "5"}, "invalid duration"),
    ],
    ids=[
        "seconds-suffixed",
        "seconds-empty",
        "seconds-5kb",
        "seconds-list",
        "seconds-bool",
        "seconds-fraction",
        "size-word",
        "size-5kb",
        "size-int",
        "size-zero-side",
        "parameters-string",
        "parameters-list",
        "parameters-seed-string",
        "parameters-seed-negative",
        "parameters-audio-string",
        "parameters-duration-string",
    ],
)
def test_malformed_create_field_answers_400_before_any_task_is_created(
    gateway: Gateway, field: str, value: JsonValue, message: str
) -> None:
    with wire_server(task_api()) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        rejected: Final = create(gateway, {"model": model, "seconds": "5", "size": "1280x720", field: value})
        assert rejected.status_code == 400, rejected.text
        assert message in error_message(rejected), rejected.text
        assert wire.drain() == ()
        created_id(create(gateway, {"model": model, "seconds": "5", "size": "1280x720"}))
        assert len(create_bodies(wire.drain())) == 1


def test_the_same_field_sent_twice_takes_the_last_value_and_still_validates(gateway: Gateway) -> None:
    with wire_server(task_api()) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        raw: Final = (
            f'{{"model": "{model}", "prompt": "{PROMPT}", "seconds": "5", "seconds": "5s", "size": "1280x720"}}'
        ).encode()
        rejected: Final = gateway.client.post(
            "/v1/videos",
            content=raw,
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        assert rejected.status_code == 400, rejected.text
        assert "seconds must be a whole number of seconds" in error_message(rejected)
        accepted: Final = gateway.client.post(
            "/v1/videos",
            content=raw.replace(b'"5s"', b'"6"'),
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        created_id(accepted)
        (body,) = create_bodies(wire.drain())
        assert object_value(body["parameters"])["duration"] == 6


@pytest.mark.parametrize(
    ("body", "drop", "message"),
    [
        ({"seconds": "-1"}, False, "Smart duration (duration=-1) is not supported"),
        ({"parameters": {"duration": -1}}, True, "Smart duration (duration=-1) is not supported"),
        (
            {"media": [{"type": "reference_video", "url": "https://example.com/in.mp4"}]},
            False,
            "Video input (a reference_video media entry) is not supported",
        ),
        (
            {"media": [{"type": "reference_video", "url": "https://example.com/in.mp4"}]},
            True,
            "Video input (a reference_video media entry) is not supported",
        ),
    ],
    ids=["smart-duration", "smart-duration-drop-params", "reference-video", "reference-video-drop-params"],
)
def test_requests_billed_before_their_length_is_known_answer_400_even_with_drop_params(
    gateway: Gateway, body: Mapping[str, JsonValue], drop: bool, message: str
) -> None:
    with wire_server(task_api()) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        rejected: Final = create(gateway, {"model": model, **body, **({"drop_params": True} if drop else {})})
        assert rejected.status_code == 400, rejected.text
        assert message in error_message(rejected), rejected.text
        assert wire.drain() == ()


@pytest.mark.parametrize(
    ("provider_model", "size", "message"),
    [
        ("wan2.7-t2v", "832x480", "wan2.7-t2v is not offered at resolution '480P', only at 720P, 1080P"),
        ("wan2.6-t2v", "1280x720", "size only maps onto DashScope video models in litellm's cost map"),
    ],
    ids=["unpriced-tier", "unpriced-model"],
)
def test_a_size_litellm_cannot_bill_answers_400_without_a_task(
    gateway: Gateway, provider_model: str, size: str, message: str
) -> None:
    with wire_server(task_api()) as wire, gateway.scenario() as scenario:
        model: Final = deployed(
            gateway, scenario, model=f"dashscope/{provider_model}", api_base=wire.url, api_key=API_KEY
        )
        rejected: Final = create(gateway, {"model": model, "seconds": "5", "size": size})
        assert rejected.status_code == 400, rejected.text
        assert message in error_message(rejected), rejected.text
        assert wire.drain() == ()


def test_a_dashscope_200_carrying_an_error_code_reaches_the_caller_as_400_after_one_attempt(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return json_reply({"request_id": "req-1", "code": "InvalidParameter", "message": "The size is not supported."})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        rejected: Final = create(gateway, {"model": model, "seconds": "5", "size": "1280x720"})
        assert rejected.status_code == 400, rejected.text
        assert "The size is not supported." in error_message(rejected), rejected.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", CREATE_PATH)]


def test_a_dashscope_invalid_api_key_reaches_the_caller_as_401(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return json_reply(
            {"request_id": "req-2", "code": "InvalidApiKey", "message": "Invalid API-key provided."}, status=401
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        rejected: Final = create(gateway, {"model": model, "seconds": "5", "size": "1280x720"})
        assert rejected.status_code == 401, rejected.text
        assert "Invalid API-key provided." in rejected.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", CREATE_PATH)]


def test_an_expired_task_polls_as_400_task_not_found_and_leaves_the_deployment_serving(gateway: Gateway) -> None:
    with wire_server(task_api(unknown)) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        video_id: Final = created_id(create(gateway, {"model": model, "seconds": "5", "size": "1280x720"}))
        missing: Final = tuple(
            gateway.request("GET", path)
            for path in (f"/v1/videos/{video_id}", f"/v1/videos/{video_id}", f"/v1/videos/{video_id}/content")
        )
        assert [response.status_code for response in missing] == [400, 400, 400], [r.text for r in missing]
        assert all(
            "Task not found. DashScope task ids expire 24 hours after creation" in error_message(response)
            for response in missing
        )
        created_id(create(gateway, {"model": model, "seconds": "5", "size": "1280x720"}))
        targets: Final = [(request.method, request.target) for request in wire.drain()]
        assert targets == [
            ("POST", CREATE_PATH),
            ("GET", f"{TASKS_PATH}{task_of(video_id)}"),
            ("GET", f"{TASKS_PATH}{task_of(video_id)}"),
            ("GET", f"{TASKS_PATH}{task_of(video_id)}"),
            ("POST", CREATE_PATH),
        ]


def test_a_download_before_the_task_finishes_answers_400_without_touching_storage(gateway: Gateway) -> None:
    with wire_server(task_api(pending)) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        video_id: Final = created_id(create(gateway, {"model": model, "seconds": "5", "size": "1280x720"}))
        status: Final = gateway.get(f"/v1/videos/{video_id}")
        assert (status["id"], status["status"]) == (video_id, "queued")
        early: Final = gateway.request("GET", f"/v1/videos/{video_id}/content")
        assert early.status_code == 400, early.text
        assert "Video is still processing (status: PENDING)" in error_message(early)
        assert not any(request.target.startswith(OSS_PATH) for request in wire.drain())


def _failed(task_id: str, origin: str) -> Mapping[str, JsonValue]:
    return {
        "request_id": f"req-{task_id}",
        "output": {
            "task_id": task_id,
            "task_status": "FAILED",
            "code": "DataInspectionFailed",
            "message": "Output data may contain inappropriate content.",
        },
    }


def test_a_failed_task_reports_failed_with_dashscopes_error_and_its_download_answers_400(gateway: Gateway) -> None:
    with wire_server(task_api(_failed)) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        video_id: Final = created_id(create(gateway, {"model": model, "seconds": "5", "size": "1280x720"}))
        status: Final = gateway.get(f"/v1/videos/{video_id}")
        assert status["status"] == "failed", status
        assert status["error"] == {
            "code": "DataInspectionFailed",
            "message": "Output data may contain inappropriate content.",
        }
        download: Final = gateway.request("GET", f"/v1/videos/{video_id}/content")
        assert download.status_code == 400, download.text
        assert "Video generation failed: Output data may contain inappropriate content." in error_message(download)
        assert not any(request.target.startswith(OSS_PATH) for request in wire.drain())


def _unexpected_status(task_id: str, origin: str) -> Mapping[str, JsonValue]:
    return {"request_id": f"req-{task_id}", "output": {"task_id": task_id, "task_status": "ARCHIVED"}}


@pytest.mark.parametrize("shape", ["unknown-status", "not-json"])
def test_a_task_body_dashscope_does_not_document_answers_502(gateway: Gateway, shape: str) -> None:
    created: Final = task_api()

    def respond(request: Request) -> Reply:
        if request.method == "POST":
            return created(request)
        if shape == "not-json":
            return Reply(body=b"<html>gateway timeout</html>", content_type="text/html")
        return json_reply(_unexpected_status(request.target.removeprefix(TASKS_PATH), ""))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        video_id: Final = created_id(create(gateway, {"model": model, "seconds": "5", "size": "1280x720"}))
        polled: Final = gateway.request("GET", f"/v1/videos/{video_id}")
        assert polled.status_code == 502, polled.text
        assert (
            "Unknown DashScope task status: ARCHIVED"
            if shape == "unknown-status"
            else "Unexpected DashScope video response"
        ) in error_message(polled)
        assert len(wire.drain()) == 2


def test_storage_refusing_the_download_reaches_the_caller_as_an_error_and_the_proxy_keeps_serving(
    gateway: Gateway,
) -> None:
    refused: Final = Reply(status=403, body=b"<Error><Code>AccessDenied</Code></Error>", content_type="application/xml")
    with wire_server(task_api(oss=refused)) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        video_id: Final = created_id(create(gateway, {"model": model, "seconds": "5", "size": "1280x720"}))
        download: Final = gateway.request("GET", f"/v1/videos/{video_id}/content")
        assert download.status_code == 403, download.text
        assert download.headers["content-type"].startswith("application/json"), download.headers
        assert "AccessDenied" in error_message(download), download.text
        assert gateway.get(f"/v1/videos/{video_id}")["status"] == "completed"
        assert [request.target for request in wire.drain()] == [
            CREATE_PATH,
            f"{TASKS_PATH}{task_of(video_id)}",
            f"{OSS_PATH}{task_of(video_id)}.mp4",
            f"{TASKS_PATH}{task_of(video_id)}",
        ]


def test_the_proxy_has_no_video_delete_route_so_a_dashscope_task_is_never_touched(gateway: Gateway) -> None:
    with wire_server(task_api()) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        video_id: Final = created_id(create(gateway, {"model": model, "seconds": "5", "size": "1280x720"}))
        wire.drain()
        response: Final = gateway.request("DELETE", f"/v1/videos/{video_id}")
        assert response.status_code == 405, response.text
        assert wire.drain() == ()
        assert gateway.get(f"/v1/videos/{video_id}")["status"] == "completed"


@pytest.mark.parametrize(
    ("operation", "message"),
    [("remix", "Video remix is not supported by DashScope"), ("list", "DashScope API key is required")],
)
def test_operations_dashscope_has_no_api_for_answer_an_error_and_send_nothing_upstream(
    gateway: Gateway, operation: str, message: str
) -> None:
    with wire_server(task_api()) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        video_id: Final = created_id(create(gateway, {"model": model, "seconds": "5", "size": "1280x720"}))
        wire.drain()
        response: Final = (
            gateway.request("POST", f"/v1/videos/{video_id}/remix", {"prompt": "now at dusk"})
            if operation == "remix"
            else gateway.request("GET", "/v1/videos", params={"custom_llm_provider": "dashscope"})
        )
        assert response.status_code >= 400, response.text
        assert message in error_message(response), response.text
        assert wire.drain() == ()
        assert gateway.get(f"/v1/videos/{video_id}")["status"] == "completed"


@pytest.mark.parametrize("level", ["request", "deployment"])
def test_drop_params_drops_malformed_fields_and_unpriced_tiers_instead_of_answering_400(
    gateway: Gateway, level: str
) -> None:
    with wire_server(task_api()) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, 
            model="dashscope/wan2.7-t2v",
            api_base=wire.url,
            api_key=API_KEY,
            **({"drop_params": True} if level == "deployment" else {}),
        )
        key: Final = scenario.key(models=[model])
        created_id(
            create(
                gateway,
                {
                    "model": model,
                    "seconds": "10s",
                    "size": "832x480",
                    "parameters": {"seed": "abc", "watermark": True},
                    **({"drop_params": True} if level == "request" else {}),
                },
                key,
            )
        )
        (body,) = create_bodies(wire.drain())
        assert body == {
            "model": "wan2.7-t2v",
            "input": {"prompt": PROMPT},
            "parameters": {"ratio": "16:9", "watermark": True},
        }
        expected: Final = 5 * rate("dashscope", _1080P, "wan2.7-t2v")
        settled_spend(key, expected)


@pytest.mark.parametrize(
    "parameters", [{}, None, "missing"], ids=["empty-object", "null", "missing"]
)
def test_empty_null_and_missing_parameters_send_the_same_task_body(gateway: Gateway, parameters: JsonValue) -> None:
    with wire_server(task_api()) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        created_id(
            create(
                gateway,
                {"model": model, "seconds": "5", **({} if parameters == "missing" else {"parameters": parameters})},
            )
        )
        (body,) = create_bodies(wire.drain())
        assert body == {"model": MODEL, "input": {"prompt": PROMPT}, "parameters": {"duration": 5}}


def test_two_identical_creates_are_two_tasks_each_billed_once(gateway: Gateway) -> None:
    with wire_server(task_api()) as wire, gateway.scenario() as scenario:
        model: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=wire.url, api_key=API_KEY)
        key: Final = scenario.key(models=[model])
        first: Final = created_id(create(gateway, {"model": model, "seconds": "5", "size": "832x480"}, key))
        second: Final = created_id(create(gateway, {"model": model, "seconds": "5", "size": "832x480"}, key))
        assert first != second
        assert task_of(first) != task_of(second)
        bodies: Final = create_bodies(wire.drain())
        assert len(bodies) == 2 and bodies[0] == bodies[1]
        expected: Final = 2 * 5 * rate("dashscope", "output_cost_per_second_480p")
        settled_spend(key, expected)


async def test_litellm_sdk_sync_and_async_create_poll_and_download_through_dashscope() -> None:
    import litellm

    with wire_server(task_api()) as wire:
        created: Final = litellm.video_generation(
            model=f"dashscope/{MODEL}", prompt=PROMPT, seconds="5", size="1280x720", api_base=wire.url, api_key=API_KEY
        )
        status: Final = litellm.video_status(video_id=created.id, api_base=wire.url, api_key=API_KEY)
        assert (status.id, status.status) == (created.id, "completed")
        assert litellm.video_content(video_id=created.id, api_base=wire.url, api_key=API_KEY) == MP4
        async_created: Final = await litellm.avideo_generation(
            model=f"dashscope/{MODEL}", prompt=PROMPT, seconds="5", size="1280x720", api_base=wire.url, api_key=API_KEY
        )
        async_status: Final = await litellm.avideo_status(video_id=async_created.id, api_base=wire.url, api_key=API_KEY)
        assert (async_status.id, async_status.status) == (async_created.id, "completed")
        assert await litellm.avideo_content(video_id=async_created.id, api_base=wire.url, api_key=API_KEY) == MP4
        bodies: Final = create_bodies(wire.drain())
        assert bodies == tuple(
            {
                "model": MODEL,
                "input": {"prompt": PROMPT},
                "parameters": {"ratio": "16:9", "resolution": "720P", "duration": 5},
            }
            for _ in range(2)
        )
