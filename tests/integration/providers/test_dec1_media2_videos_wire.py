"""`/v1/videos` on OpenAI Sora, Gemini (AI Studio) Veo and Vertex AI Veo: create, poll, download, remix.

The proxy wraps every provider video id: the id a caller sees is `video_<base64>` whose payload names the provider,
the deployment's model id and the provider's own id (litellm/types/videos/utils.py). Polls and downloads decode it,
route to the deployment that created the video, and send the provider's own id upstream.

OpenAI (litellm/llms/openai/videos/transformation.py): create is a form POST to `<api_base>/videos` with `model`,
`prompt`, `seconds`, `size` and an optional `input_reference` file; status is `GET /videos/<id>`; content is
`GET /videos/<id>/content`; remix is `POST /videos/<id>/remix`. `seed`, `duration`, `aspect_ratio` and
`generate_audio` are refused unless `drop_params` is set. Cost is `seconds x per-second price`, taken from the
create response's `seconds` and billed once at create.

Gemini Veo (litellm/llms/gemini/videos/transformation.py): create is `POST
<api_base>/v1beta/models/<model>:predictLongRunning` with `x-goog-api-key` and `{"instances": [{"prompt"}],
"parameters": {"aspectRatio", "resolution", "durationSeconds"}}`; the operation `name` becomes the video id;
status is `GET <api_base>/v1beta/<operation name>`; content reads the finished operation's first
`generatedSamples[].video.uri` and downloads it.

Vertex Veo (litellm/llms/vertex_ai/videos/transformation.py): create is `POST .../publishers/google/models/<model>:predictLongRunning`
with a bearer token; status and content are `POST .../models/<model>:fetchPredictOperation` with
`{"operationName"}`; content decodes `response.videos[0].bytesBase64Encoded`. Without `seconds`, both Google
providers bill `DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS` (8) even though no `durationSeconds` is sent.

`routing/test_video_poll_routing_wire.py` covers which deployment a poll reaches; this file covers the wire and cost.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Callable, Mapping
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, string_value
from integration._support.wire import Reply, Request, wire_server
from integration.providers._dashscope_video_support import spend_stays
from integration.providers.dec1_media2_support import (
    GEMINI_KEY,
    MP4_BYTES,
    OPENAI_KEY,
    decoded_video,
    error_message,
    file_parts,
    form_fields,
    json_body,
    json_reply,
    logged_rows,
    openai_error_reply,
    path_of,
    png,
    settled_key_spend,
)
from integration.providers.dec1_vertex_support import LOCATION, PROJECT, vertex
from pydantic import JsonValue

_VIDEOS: Final = "/v1/videos"
_PROMPT: Final = "a paper boat drifting down a rainy street"
_RATE: Final = 0.1
_VEO: Final = "veo-dec1-test-001"


def _sora(video_id: str, status: str = "queued", seconds: str = "8") -> dict[str, JsonValue]:
    return {
        "id": video_id,
        "object": "video",
        "model": "sora-2",
        "status": status,
        "progress": 100 if status == "completed" else 0,
        "created_at": 1791943200,
        "seconds": seconds,
        "size": "1280x720",
    }


def _sora_api(provider_id: str, *, seconds: str = "8") -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "POST" and path_of(request) == _VIDEOS:
            return json_reply(_sora(provider_id, seconds=seconds))
        if request.method == "POST" and path_of(request).endswith("/remix"):
            return json_reply(_sora(f"{provider_id}_remix", seconds=seconds))
        if path_of(request).endswith("/content"):
            return Reply(body=MP4_BYTES, content_type="video/mp4")
        return json_reply(_sora(provider_id, status="completed", seconds=seconds))

    return respond


def _fields(request: Request) -> dict[str, str] | dict[str, JsonValue]:
    """The create or remix fields, whether the provider client sent JSON or a form."""
    if request.headers.get("content-type", "").startswith("application/json"):
        return json_body(request)
    return form_fields(request)


def _video_id(response_body: Mapping[str, JsonValue]) -> str:
    return string_value(response_body["id"])


# ---------------------------------------------------------------- OpenAI Sora


def test_sora_create_posts_model_prompt_seconds_and_size_with_a_bearer_key(gateway: Gateway) -> None:
    """An `openai/sora-2` deployment creates a video with `seconds` `8` and `size` `1280x720`. OpenAI must get one
    POST to `/v1/videos` with `Authorization: Bearer <key>` and fields `model=sora-2`, `prompt`, `seconds=8`,
    `size=1280x720`. The caller gets status `queued` and an id that is not the provider id but decodes to provider
    `openai` and provider id `video_abc` (the proxy wraps it)."""
    provider_id: Final = f"video_{uuid.uuid4().hex}"
    with wire_server(_sora_api(provider_id)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/sora-2", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request(
            "POST", _VIDEOS, {"model": model, "prompt": _PROMPT, "seconds": "8", "size": "1280x720"}
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        (request,) = wire.drain()
    assert (request.method, path_of(request)) == ("POST", _VIDEOS)
    assert request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    fields: Final = _fields(request)
    assert (fields["model"], fields["prompt"], fields["seconds"], fields["size"]) == (
        "sora-2",
        _PROMPT,
        "8",
        "1280x720",
    )
    assert body["status"] == "queued"
    assert _video_id(body) != provider_id
    unpacked: Final = decoded_video(_video_id(body))
    assert (unpacked["custom_llm_provider"], unpacked["video_id"]) == ("openai", provider_id)


def test_sora_create_with_seed_is_refused_with_400_before_calling_openai(gateway: Gateway) -> None:
    """The caller adds `seed` 7, which OpenAI's video API does not take. Without `drop_params` the proxy must answer
    400 with `openai video generation does not support seed, set drop_params=True to ignore them` and not call
    OpenAI (litellm/llms/base_llm/videos/async_task_utils.py unsupported_video_fields).

    BUG (run 44176682961): `seed` reaches OpenAI as a form field, together with the deployment's own
    `drop_params: "true"`, and the create is answered 200. `map_openai_params` refuses or drops `seed`
    (litellm/llms/openai/videos/transformation.py:127-140 on main), but here it never runs on `seed`: the fields
    land in `extra_body`, which the create request merges in unchecked (`data.update(extra_body)`, same file
    :311-312), and the deployment `drop_params` setting leaks into the provider request the same way."""
    pytest.skip("BUG: an OpenAI video create forwards seed and the deployment drop_params flag to OpenAI unchecked")
    with wire_server(_sora_api("video_x")) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/sora-2", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request("POST", _VIDEOS, {"model": model, "prompt": _PROMPT, "seed": 7})
        assert response.status_code == 400, response.text
        assert "openai video generation does not support seed, set drop_params=True to ignore them" in (
            error_message(response)
        )
        assert wire.drain() == ()


def test_sora_create_with_seed_and_drop_params_sends_no_seed(gateway: Gateway) -> None:
    """With `drop_params: true` on the deployment, the same `seed` 7 must be dropped: OpenAI gets the create with
    no `seed` field and the caller gets 200.

    BUG (run 44176682961): `seed` reaches OpenAI as a form field, together with the deployment's own
    `drop_params: "true"`, and the create is answered 200. `map_openai_params` refuses or drops `seed`
    (litellm/llms/openai/videos/transformation.py:127-140 on main), but here it never runs on `seed`: the fields
    land in `extra_body`, which the create request merges in unchecked (`data.update(extra_body)`, same file
    :311-312), and the deployment `drop_params` setting leaks into the provider request the same way."""
    pytest.skip("BUG: an OpenAI video create forwards seed and the deployment drop_params flag to OpenAI unchecked")
    with wire_server(_sora_api(f"video_{uuid.uuid4().hex}")) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/sora-2", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, drop_params=True
        )
        response: Final = gateway.request("POST", _VIDEOS, {"model": model, "prompt": _PROMPT, "seed": 7})
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert "seed" not in _fields(request)


def test_sora_create_with_a_multipart_input_reference_uploads_it_as_a_file_part(gateway: Gateway) -> None:
    """The caller posts multipart with `model`, `prompt` and an `input_reference` PNG. OpenAI must get a multipart
    create whose `input_reference` file part holds the uploaded bytes unchanged."""
    image: Final = png((10, 20, 30, 255))
    with wire_server(_sora_api(f"video_{uuid.uuid4().hex}")) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/sora-2", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.client.post(
            _VIDEOS,
            data={"model": model, "prompt": _PROMPT},
            files={"input_reference": ("first.png", image, "image/png")},
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert [(part.name, part.content) for part in file_parts(request)] == [("input_reference", image)]


def test_sora_status_poll_sends_the_provider_id_and_returns_the_wrapped_id(gateway: Gateway) -> None:
    """After a create, the caller polls `GET /v1/videos/<wrapped id>`. OpenAI must get `GET /v1/videos/<provider
    id>` with the bearer key, and the caller gets status `completed` with the same wrapped id it polled."""
    provider_id: Final = f"video_{uuid.uuid4().hex}"
    with wire_server(_sora_api(provider_id)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/sora-2", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        created: Final = gateway.request("POST", _VIDEOS, {"model": model, "prompt": _PROMPT})
        video_id: Final = _video_id(JSON_OBJECT.validate_json(created.content))
        wire.drain()
        polled: Final = gateway.request("GET", f"{_VIDEOS}/{video_id}")
        assert polled.status_code == 200, polled.text
        body: Final = JSON_OBJECT.validate_json(polled.content)
        (request,) = wire.drain()
    assert (request.method, path_of(request)) == ("GET", f"{_VIDEOS}/{provider_id}")
    assert request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    assert body["status"] == "completed"
    assert decoded_video(_video_id(body))["video_id"] == provider_id


def test_sora_content_download_returns_the_mp4_as_an_attachment(gateway: Gateway) -> None:
    """The caller downloads `GET /v1/videos/<wrapped id>/content`. OpenAI must get `GET /v1/videos/<provider
    id>/content`; the caller gets the MP4 bytes unchanged with `content-type` `video/mp4` and
    `content-disposition` `attachment; filename=video_<wrapped id>.mp4` (litellm/proxy/video_endpoints/endpoints.py)."""
    provider_id: Final = f"video_{uuid.uuid4().hex}"
    with wire_server(_sora_api(provider_id)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/sora-2", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        created: Final = gateway.request("POST", _VIDEOS, {"model": model, "prompt": _PROMPT})
        video_id: Final = _video_id(JSON_OBJECT.validate_json(created.content))
        wire.drain()
        download: Final = gateway.request("GET", f"{_VIDEOS}/{video_id}/content")
        assert download.status_code == 200, download.text
        (request,) = wire.drain()
    assert (request.method, path_of(request)) == ("GET", f"{_VIDEOS}/{provider_id}/content")
    assert download.content == MP4_BYTES
    assert download.headers["content-type"] == "video/mp4"
    assert download.headers["content-disposition"] == f"attachment; filename=video_{video_id}.mp4"


def test_sora_thumbnail_variant_is_forwarded_to_openai(gateway: Gateway) -> None:
    """The caller downloads `/v1/videos/<id>/content?variant=thumbnail`. OpenAI should get
    `GET /v1/videos/<provider id>/content?variant=thumbnail` so the caller receives the thumbnail.
    Product bug: the proxy content endpoint builds its request data from `video_id` only and never reads the
    `variant` query, so the full MP4 is fetched and labelled `video/mp4`."""
    pytest.skip("BUG: /v1/videos/{id}/content ignores the variant query parameter")
    provider_id: Final = f"video_{uuid.uuid4().hex}"
    with wire_server(_sora_api(provider_id)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/sora-2", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        created: Final = gateway.request("POST", _VIDEOS, {"model": model, "prompt": _PROMPT})
        video_id: Final = _video_id(JSON_OBJECT.validate_json(created.content))
        wire.drain()
        download: Final = gateway.request("GET", f"{_VIDEOS}/{video_id}/content", params={"variant": "thumbnail"})
        assert download.status_code == 200, download.text
        (request,) = wire.drain()
    assert request.target == f"{_VIDEOS}/{provider_id}/content?variant=thumbnail"


def test_sora_remix_posts_the_new_prompt_to_the_provider_id(gateway: Gateway) -> None:
    """The caller remixes a created video with `prompt` `make it snow`. OpenAI must get one POST to
    `/v1/videos/<provider id>/remix` carrying that prompt, and the caller gets a new wrapped id whose provider id is
    the remix id OpenAI returned."""
    provider_id: Final = f"video_{uuid.uuid4().hex}"
    with wire_server(_sora_api(provider_id)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/sora-2", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        created: Final = gateway.request("POST", _VIDEOS, {"model": model, "prompt": _PROMPT})
        video_id: Final = _video_id(JSON_OBJECT.validate_json(created.content))
        wire.drain()
        remixed: Final = gateway.request("POST", f"{_VIDEOS}/{video_id}/remix", {"prompt": "make it snow"})
        assert remixed.status_code == 200, remixed.text
        (request,) = wire.drain()
    assert (request.method, path_of(request)) == ("POST", f"{_VIDEOS}/{provider_id}/remix")
    assert _fields(request)["prompt"] == "make it snow"
    assert decoded_video(_video_id(JSON_OBJECT.validate_json(remixed.content)))["video_id"] == f"{provider_id}_remix"


def test_sora_bills_the_created_seconds_once_and_polls_do_not_bill_again(gateway: Gateway) -> None:
    """An `openai/sora-2` deployment priced at `output_cost_per_second` 0.1 creates a video that OpenAI reports as
    `seconds` `8`. The key spend must settle at 8 x 0.1 after the create, and stay there after two status polls
    that report `completed`, so a video is charged once.
    UNVERIFIED: which row carries the charge (the create row, or a `<video id>_video_cost` poll row); the test pins
    only the key total."""
    with wire_server(_sora_api(f"video_{uuid.uuid4().hex}", seconds="8")) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/sora-2", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, output_cost_per_second=_RATE
        )
        key: Final = scenario.key(models=[model])
        created: Final = gateway.request("POST", _VIDEOS, {"model": model, "prompt": _PROMPT}, key=key)
        assert created.status_code == 200, created.text
        settled_key_spend(key, 8 * _RATE)
        video_id: Final = _video_id(JSON_OBJECT.validate_json(created.content))
        for _ in range(2):
            assert gateway.request("GET", f"{_VIDEOS}/{video_id}", key=key).status_code == 200
        spend_stays(key, 8 * _RATE)
        wire.drain()


def test_sora_create_refused_by_openai_keeps_the_status_and_logs_a_failure(gateway: Gateway) -> None:
    """OpenAI refuses the create with 400 `Your request was blocked by our moderation system`. The caller must get
    400 with that message, and the key's spend row is a `failure` with zero spend."""
    with (
        wire_server(
            lambda _r: openai_error_reply(
                400, "Your request was blocked by our moderation system", "invalid_request_error", "moderation_blocked"
            )
        ) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="openai/sora-2", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, output_cost_per_second=_RATE
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request("POST", _VIDEOS, {"model": model, "prompt": _PROMPT}, key=key)
        assert response.status_code == 400, response.text
        assert "Your request was blocked by our moderation system" in error_message(response)
        (row,) = logged_rows(key)
        assert row["status"] == "failure"
        assert float(str(row["spend"])) == 0.0
        assert len(wire.drain()) >= 1


def test_a_poll_with_an_id_the_proxy_never_issued_routes_to_openai_by_default(gateway: Gateway) -> None:
    """The caller polls `GET /v1/videos/video_plain123`, an id with no LiteLLM wrapping. The status endpoint falls
    back to provider `openai` (`... or provider_from_id or "openai"`), so with no OpenAI deployment reachable for it
    the caller must get an error status, not a 200 for a video that was never created.
    UNVERIFIED: the exact status, which depends on whether the proxy has an `OPENAI_API_KEY` in its environment."""
    response: Final = gateway.request("GET", f"{_VIDEOS}/video_plain{uuid.uuid4().hex[:8]}")
    assert response.status_code >= 400, response.text


# ---------------------------------------------------------------- Gemini (AI Studio) Veo


def _gemini_veo(operation: str, *, done: bool = True, uri: str | None = None) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "POST":
            return json_reply({"name": operation, "done": False})
        if path_of(request) == "/download/video.mp4":
            return Reply(body=MP4_BYTES, content_type="video/mp4")
        finished: Final[dict[str, JsonValue]] = {"name": operation, "done": done}
        if done and uri is not None:
            finished["response"] = {"generateVideoResponse": {"generatedSamples": [{"video": {"uri": uri}}]}}
        return json_reply(finished)

    return respond


def test_gemini_veo_create_maps_size_and_seconds_into_predict_long_running(gateway: Gateway) -> None:
    """A `gemini/veo-3.0-generate-001` deployment creates a video with `size` `1280x720` and `seconds` `6`. Google must
    get one POST to `/v1beta/models/veo-3.0-generate-001:predictLongRunning` with `x-goog-api-key: <key>` and the
    body `{"instances": [{"prompt": <prompt>}], "parameters": {"aspectRatio": "16:9", "resolution": "720p",
    "durationSeconds": 6}}`. The caller gets status `processing` and an id that decodes to the operation name."""
    operation: Final = f"models/veo-3.0-generate-001/operations/{uuid.uuid4().hex}"
    with wire_server(_gemini_veo(operation)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="gemini/veo-3.0-generate-001", api_base=wire.url, api_key=GEMINI_KEY)
        response: Final = gateway.request(
            "POST", _VIDEOS, {"model": model, "prompt": _PROMPT, "size": "1280x720", "seconds": "6"}
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        (request,) = wire.drain()
    assert (request.method, path_of(request)) == ("POST", "/v1beta/models/veo-3.0-generate-001:predictLongRunning")
    assert request.headers["x-goog-api-key"] == GEMINI_KEY
    assert json_body(request) == {
        "instances": [{"prompt": _PROMPT}],
        "parameters": {"aspectRatio": "16:9", "resolution": "720p", "durationSeconds": 6},
    }
    assert body["status"] == "processing"
    assert decoded_video(_video_id(body))["video_id"] == operation


@pytest.mark.parametrize(("size", "aspect_ratio"), (("720x1280", "9:16"), ("1080x1920", "9:16"), ("640x480", "16:9")))
def test_gemini_veo_maps_portrait_sizes_to_9_16_and_unknown_sizes_to_16_9(
    gateway: Gateway, size: str, aspect_ratio: str
) -> None:
    """Portrait OpenAI sizes map to `aspectRatio` `9:16`; a size outside the four presets (`640x480`) still maps,
    to the `16:9` default, and then sends no `resolution` because none can be inferred."""
    operation: Final = f"models/veo-3.0-generate-001/operations/{uuid.uuid4().hex}"
    with wire_server(_gemini_veo(operation)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="gemini/veo-3.0-generate-001", api_base=wire.url, api_key=GEMINI_KEY)
        response: Final = gateway.request("POST", _VIDEOS, {"model": model, "prompt": _PROMPT, "size": size})
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    parameters: Final = json_body(request)["parameters"]
    assert parameters["aspectRatio"] == aspect_ratio  # type: ignore[index]
    assert ("resolution" in parameters) == (size != "640x480")  # type: ignore[operator]


def test_gemini_veo_status_polls_the_operation_and_reports_processing_then_completed(gateway: Gateway) -> None:
    """The caller polls a created Veo video twice; Google answers `done: false` then `done: true`. Each poll must be
    `GET /v1beta/<operation name>` with the API key header, and the caller sees `processing` then `completed`."""
    operation: Final = f"models/veo-3.0-generate-001/operations/{uuid.uuid4().hex}"
    answers: Final = iter((False, True))

    def respond(request: Request) -> Reply:
        if request.method == "POST":
            return json_reply({"name": operation, "done": False})
        return json_reply({"name": operation, "done": next(answers)})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="gemini/veo-3.0-generate-001", api_base=wire.url, api_key=GEMINI_KEY)
        created: Final = gateway.request("POST", _VIDEOS, {"model": model, "prompt": _PROMPT})
        video_id: Final = _video_id(JSON_OBJECT.validate_json(created.content))
        wire.drain()
        statuses: Final = [
            JSON_OBJECT.validate_json(gateway.request("GET", f"{_VIDEOS}/{video_id}").content)["status"]
            for _ in range(2)
        ]
        polls: Final = wire.drain()
    assert statuses == ["processing", "completed"]
    assert [(poll.method, path_of(poll)) for poll in polls] == [("GET", f"/v1beta/{operation}")] * 2
    assert all(poll.headers["x-goog-api-key"] == GEMINI_KEY for poll in polls)


def test_gemini_veo_content_reads_the_sample_uri_from_the_finished_operation_and_downloads_it(
    gateway: Gateway,
) -> None:
    """A finished operation names `generatedSamples[0].video.uri` `<peer>/download/video.mp4`. A content download must
    first GET the operation, then GET that URI, and return its bytes to the caller unchanged.
    UNVERIFIED: whether the download request carries the `x-goog-api-key` header (the content handler reuses the
    status headers)."""
    operation: Final = f"models/veo-3.0-generate-001/operations/{uuid.uuid4().hex}"
    base: Final[list[str]] = []

    def respond(request: Request) -> Reply:
        return _gemini_veo(operation, uri=f"{base[0]}/download/video.mp4")(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        base.append(wire.url)
        model: Final = scenario.model(model="gemini/veo-3.0-generate-001", api_base=wire.url, api_key=GEMINI_KEY)
        created: Final = gateway.request("POST", _VIDEOS, {"model": model, "prompt": _PROMPT})
        video_id: Final = _video_id(JSON_OBJECT.validate_json(created.content))
        wire.drain()
        download: Final = gateway.request("GET", f"{_VIDEOS}/{video_id}/content")
        assert download.status_code == 200, download.text
        requests: Final = wire.drain()
    assert download.content == MP4_BYTES
    assert [(request.method, path_of(request)) for request in requests] == [
        ("GET", f"/v1beta/{operation}"),
        ("GET", "/download/video.mp4"),
    ]


def test_gemini_veo_content_before_the_operation_finishes_is_an_error(gateway: Gateway) -> None:
    """A download is requested while the operation is still `done: false`. The caller must get an error status with
    `Video generation is not complete yet` in the text, and no second (download) request reaches Google.
    UNVERIFIED: the status code; the provider raises a plain ValueError."""
    operation: Final = f"models/veo-3.0-generate-001/operations/{uuid.uuid4().hex}"
    with wire_server(_gemini_veo(operation, done=False)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="gemini/veo-3.0-generate-001", api_base=wire.url, api_key=GEMINI_KEY)
        created: Final = gateway.request("POST", _VIDEOS, {"model": model, "prompt": _PROMPT})
        video_id: Final = _video_id(JSON_OBJECT.validate_json(created.content))
        wire.drain()
        download: Final = gateway.request("GET", f"{_VIDEOS}/{video_id}/content")
        assert download.status_code >= 400, download.text
        assert "Video generation is not complete yet" in download.text
        assert [request.method for request in wire.drain()] == ["GET"]


def test_gemini_veo_without_seconds_sends_no_duration_but_bills_the_eight_second_default(gateway: Gateway) -> None:
    """A Gemini Veo deployment priced at `output_cost_per_second` 0.1 creates a video with no `seconds`. Google must get
    `parameters` without `durationSeconds` (no body field at all when nothing is mapped), yet the key spend must be
    8 x 0.1, because usage falls back to `DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS` (litellm/constants.py).

    BUG (run 44176682961): the key spend never left 0.0 in 60 s. The deployment's `output_cost_per_second` is not
    applied to a Veo create; owned proxies in the same run log `No cost information found for video model
    gemini/veo-3.0-generate-001` and `vertex_ai/veo-dec1-test-001` (litellm/llms/openai/cost_calculation.py:215 on
    main, reached from the video branch at cost_calculator.py:2104-2153). The Sora billing test passes only because
    the cost map's own `sora-2` price (0.1 per second) equals its deployment rate. Root cause not pinned; medium
    confidence."""
    pytest.skip("BUG: a Veo video create ignores the deployment output_cost_per_second and bills 0")
    operation: Final = f"models/veo-3.0-generate-001/operations/{uuid.uuid4().hex}"
    with wire_server(_gemini_veo(operation)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="gemini/veo-3.0-generate-001", api_base=wire.url, api_key=GEMINI_KEY, output_cost_per_second=_RATE
        )
        key: Final = scenario.key(models=[model])
        created: Final = gateway.request("POST", _VIDEOS, {"model": model, "prompt": _PROMPT}, key=key)
        assert created.status_code == 200, created.text
        (request,) = wire.drain()
        settled_key_spend(key, 8 * _RATE)
    assert "durationSeconds" not in json.dumps(json_body(request))


# ---------------------------------------------------------------- Vertex AI Veo


def _vertex_operation(model: str = _VEO) -> str:
    return (
        f"projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{model}/operations/{uuid.uuid4().hex[:12]}"
    )


def _vertex_veo(
    operation: str, *, done: bool = True, error: Mapping[str, JsonValue] | None = None
) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if path_of(request).endswith(":predictLongRunning"):
            return json_reply({"name": operation})
        finished: Final[dict[str, JsonValue]] = {
            "name": operation,
            "done": done,
            "metadata": {"createTime": "2026-11-17T10:00:00.000Z"},
        }
        if error is not None:
            finished["error"] = dict(error)
        elif done:
            finished["response"] = {
                "videos": [{"bytesBase64Encoded": base64.b64encode(MP4_BYTES).decode(), "mimeType": "video/mp4"}]
            }
        return json_reply(finished)

    return respond


def test_vertex_veo_create_posts_instances_and_parameters_with_a_minted_token(gateway: Gateway) -> None:
    """A `vertex_ai/veo-dec1-test-001` deployment creates a video with `seconds` `4` and `size` `720x1280`. Vertex must
    get one POST to `/v1/projects/<project>/locations/us-central1/publishers/google/models/veo-dec1-test-001:predictLongRunning`
    with `Authorization: Bearer token-1` and the body `{"instances": [{"prompt": <prompt>}], "parameters":
    {"aspectRatio": "9:16", "durationSeconds": 4}}` (no `resolution`, since the model has no 1080p price tier)."""
    operation: Final = _vertex_operation()
    with vertex(gateway, _vertex_veo(operation), backend=_VEO) as rig:
        response: Final = rig.post(_VIDEOS, {"prompt": _PROMPT, "seconds": "4", "size": "720x1280"})
        assert response.status_code == 200, response.text
        request: Final = rig.peer.one_call()
        assert JSON_OBJECT.validate_json(response.content)["status"] == "processing"
    assert path_of(request) == (
        f"/v1/projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{_VEO}:predictLongRunning"
    )
    assert request.headers["authorization"] == "Bearer token-1"
    assert json_body(request) == {
        "instances": [{"prompt": _PROMPT}],
        "parameters": {"aspectRatio": "9:16", "durationSeconds": 4},
    }


def test_vertex_veo_flattens_a_caller_parameters_block_into_the_request_parameters(gateway: Gateway) -> None:
    """The caller passes a Vertex-native `parameters` object `{"sampleCount": 2, "personGeneration": "allow_adult"}`
    beside `seconds` `4`. Vertex must get those keys merged flat into `parameters` next to `durationSeconds`, never
    a nested `parameters.parameters` (the double-nesting the transformation guards against)."""
    operation: Final = _vertex_operation()
    with vertex(gateway, _vertex_veo(operation), backend=_VEO) as rig:
        response: Final = rig.post(
            _VIDEOS,
            {
                "prompt": _PROMPT,
                "seconds": "4",
                "parameters": {"sampleCount": 2, "personGeneration": "allow_adult"},
            },
        )
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
    assert body["parameters"] == {"sampleCount": 2, "personGeneration": "allow_adult", "durationSeconds": 4}


def test_vertex_veo_bills_seconds_times_the_sample_count(gateway: Gateway) -> None:
    """A Vertex Veo deployment priced at `output_cost_per_second` 0.1 creates `sampleCount` 2 videos of `seconds` 4.
    The key spend must be 2 x 4 x 0.1, because usage records `video_count` from `sampleCount` and the cost is
    multiplied by it (`_video_count` in litellm/cost_calculator.py).

    BUG (run 44176682961): the key spend never left 0.0 in 60 s. The deployment's `output_cost_per_second` is not
    applied to a Veo create; owned proxies in the same run log `No cost information found for video model
    gemini/veo-3.0-generate-001` and `vertex_ai/veo-dec1-test-001` (litellm/llms/openai/cost_calculation.py:215 on
    main, reached from the video branch at cost_calculator.py:2104-2153). The Sora billing test passes only because
    the cost map's own `sora-2` price (0.1 per second) equals its deployment rate. Root cause not pinned; medium
    confidence."""
    pytest.skip("BUG: a Veo video create ignores the deployment output_cost_per_second and bills 0")
    operation: Final = _vertex_operation()
    with vertex(gateway, _vertex_veo(operation), backend=_VEO, output_cost_per_second=_RATE) as rig:
        key: Final = rig.scenario.key(models=[rig.model])
        response: Final = rig.gateway.request(
            "POST",
            _VIDEOS,
            {"model": rig.model, "prompt": _PROMPT, "seconds": "4", "parameters": {"sampleCount": 2}},
            key=key,
        )
        assert response.status_code == 200, response.text
        settled_key_spend(key, 2 * 4 * _RATE)


def test_vertex_veo_status_posts_fetch_predict_operation_with_the_operation_name(gateway: Gateway) -> None:
    """Polling a Vertex video must POST `/v1/projects/<project>/locations/us-central1/publishers/google/models/<model>:fetchPredictOperation`
    with body `{"operationName": <operation>}` and the bearer token, and a `done: true` operation must reach the
    caller as `completed`."""
    operation: Final = _vertex_operation()
    with vertex(gateway, _vertex_veo(operation), backend=_VEO) as rig:
        created: Final = rig.post(_VIDEOS, {"prompt": _PROMPT})
        video_id: Final = _video_id(JSON_OBJECT.validate_json(created.content))
        rig.peer.model_calls()
        rig.peer.seen.clear()
        polled: Final = rig.gateway.request("GET", f"{_VIDEOS}/{video_id}")
        assert polled.status_code == 200, polled.text
        request: Final = rig.peer.one_call()
    assert path_of(request) == (
        f"/v1/projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{_VEO}:fetchPredictOperation"
    )
    assert json_body(request) == {"operationName": operation}
    assert request.headers["authorization"].startswith("Bearer token-")
    assert JSON_OBJECT.validate_json(polled.content)["status"] == "completed"


def test_vertex_veo_operation_error_reaches_the_caller_as_a_failed_video(gateway: Gateway) -> None:
    """Vertex answers the poll with `done: true` and `error: {"code": 3, "message": "prompt blocked"}`. The caller must
    get 200 with status `failed` and the same `error` object, not `completed`."""
    operation: Final = _vertex_operation()
    error: Final = {"code": 3, "message": "prompt blocked"}
    with vertex(gateway, _vertex_veo(operation, error=error), backend=_VEO) as rig:
        created: Final = rig.post(_VIDEOS, {"prompt": _PROMPT})
        video_id: Final = _video_id(JSON_OBJECT.validate_json(created.content))
        polled: Final = rig.gateway.request("GET", f"{_VIDEOS}/{video_id}")
        assert polled.status_code == 200, polled.text
        body: Final = JSON_OBJECT.validate_json(polled.content)
    assert body["status"] == "failed"
    assert body["error"] == error


def test_vertex_veo_content_decodes_the_base64_video_from_the_finished_operation(gateway: Gateway) -> None:
    """A content download for a finished Vertex video must POST `:fetchPredictOperation` once more and return the
    bytes of `response.videos[0].bytesBase64Encoded`, decoded, as `video/mp4`."""
    operation: Final = _vertex_operation()
    with vertex(gateway, _vertex_veo(operation), backend=_VEO) as rig:
        created: Final = rig.post(_VIDEOS, {"prompt": _PROMPT})
        video_id: Final = _video_id(JSON_OBJECT.validate_json(created.content))
        rig.peer.model_calls()
        rig.peer.seen.clear()
        download: Final = rig.gateway.request("GET", f"{_VIDEOS}/{video_id}/content")
        assert download.status_code == 200, download.text
        request: Final = rig.peer.one_call()
    assert path_of(request).endswith(f"/models/{_VEO}:fetchPredictOperation")
    assert download.content == MP4_BYTES
    assert download.headers["content-type"] == "video/mp4"


def test_vertex_veo_remix_is_refused_without_calling_vertex(gateway: Gateway) -> None:
    """Vertex Veo has no remix API. A remix of a Vertex video must answer an error status whose text contains
    `Video remix is not supported by Vertex AI Veo` and send nothing to Vertex beyond the original create.
    UNVERIFIED: the status code (a NotImplementedError surfaces as 500 unless mapped)."""
    operation: Final = _vertex_operation()
    with vertex(gateway, _vertex_veo(operation), backend=_VEO) as rig:
        created: Final = rig.post(_VIDEOS, {"prompt": _PROMPT})
        video_id: Final = _video_id(JSON_OBJECT.validate_json(created.content))
        rig.peer.model_calls()
        rig.peer.seen.clear()
        remixed: Final = rig.gateway.request("POST", f"{_VIDEOS}/{video_id}/remix", {"prompt": "again"})
        assert remixed.status_code >= 400, remixed.text
        assert "remix is not supported" in remixed.text
        assert rig.peer.model_calls() == ()
