"""`POST /v1/images/edits` on Gemini (AI Studio), Vertex AI Gemini, Vertex AI Imagen and Azure AI Foundry gpt-image.

The proxy reads the multipart edit (litellm/proxy/image_endpoints/endpoints.py): `image`/`image[]` and
`mask`/`mask[]` files become in-memory uploads, numeric form fields such as `n` are coerced to integers, and the
provider config builds its own request:

Gemini (litellm/llms/gemini/image_edit/transformation.py) sends JSON to `<api_base>/models/<model>:generateContent`
with `x-goog-api-key`: `{"contents": [{"parts": [<inlineData per image>, {"text": <prompt>}]}], "generationConfig":
{"response_modalities": ["IMAGE", "TEXT"], ...}}`. An edit with no image is a 400 (#52087, landed 2026-11-16).
`n` greater than 1 is served by `n` identical single-image requests whose images and token usage are merged
(#52103, landed 2026-11-16), capped at `MAX_IMAGE_FAN_OUT_REQUESTS` (10).

Vertex Gemini (vertex_ai/image_edit/vertex_gemini_transformation.py) sends `{"contents": {"role": "USER",
"parts": [...]}, "generationConfig": {"response_modalities": ["IMAGE"], ...}}` (contents is an object, not a list)
and reports Gemini token usage (#51038, landed 2026-11-10). Vertex Imagen (vertex_imagen_transformation.py) sends
`:predict` with `referenceImages` (raw images, then the mask as `REFERENCE_TYPE_MASK`) and fixed edit parameters.
For both Vertex configs a deployment `api_base` short-circuits auth: no token is minted and no Authorization header
is sent, and for Vertex Gemini the URL is the `api_base` itself.

Azure AI Foundry gpt-image (azure_ai/image_edit/gpt_transformation.py, #49650 landed 2026-11-12) reuses the Azure
OpenAI wire at `/openai/deployments/<deployment>/images/edits` with the `api-key` header and a default
`api-version` of `2025-04-01-preview`.

`test_dec1_media_image_edit_wire.py` and `test_openai_image_edit_wire.py` cover OpenAI and Azure OpenAI.
"""

from __future__ import annotations

import base64
import uuid
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario
from integration._support.wire import Reply, Request, wire_server
from integration.providers.dec1_media2_support import (
    AZURE_KEY,
    B64_IMAGE,
    GEMINI_KEY,
    error_message,
    file_parts,
    gemini_image_reply,
    image_reply_body,
    json_body,
    json_reply,
    path_of,
    png,
    post_upload,
    query_of,
)
from integration.providers.dec1_vertex_support import GEMINI_IMAGE, IMAGEN, LOCATION, PROJECT, models_path, vertex
from pydantic import JsonValue

_EDITS: Final = "/v1/images/edits"
_GEMINI_MODEL: Final = "gemini-2.5-flash-image"


def _prompt() -> str:
    return f"make the square blue {uuid.uuid4().hex[:8]}"


def _b64(content: bytes) -> str:
    return base64.b64encode(content).decode()


def _gemini_edit(_request: Request) -> Reply:
    return gemini_image_reply(B64_IMAGE, text="here you go")


def _gemini_deployment(scenario: Scenario, wire_url: str) -> str:
    return scenario.model(model=f"gemini/{_GEMINI_MODEL}", api_base=f"{wire_url}/v1beta", api_key=GEMINI_KEY)


# ---------------------------------------------------------------- Gemini (AI Studio)


def test_gemini_image_edit_sends_inline_image_then_prompt_to_generate_content(gateway: Gateway) -> None:
    """A `gemini/gemini-2.5-flash-image` deployment edits one PNG. Gemini must get one POST to
    `/v1beta/models/gemini-2.5-flash-image:generateContent` with `x-goog-api-key: <key>` and `contents[0].parts`
    equal to `[{"inlineData": {"mimeType": "image/png", "data": <base64 of the upload>}}, {"text": <prompt>}]`,
    image first, and `generationConfig.response_modalities` `["IMAGE", "TEXT"]`."""
    image: Final = png((1, 2, 3, 255))
    prompt: Final = _prompt()
    with wire_server(_gemini_edit) as wire, gateway.scenario() as scenario:
        model: Final = _gemini_deployment(scenario, wire.url)
        response: Final = post_upload(
            gateway, _EDITS, {"model": model, "prompt": prompt}, [("image", ("a.png", image, "image/png"))]
        )
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert (request.method, path_of(request)) == ("POST", f"/v1beta/models/{_GEMINI_MODEL}:generateContent")
    assert request.headers["x-goog-api-key"] == GEMINI_KEY
    body: Final = json_body(request)
    assert body["contents"] == [
        {"parts": [{"inlineData": {"mimeType": "image/png", "data": _b64(image)}}, {"text": prompt}]}
    ]
    assert body["generationConfig"]["response_modalities"] == ["IMAGE", "TEXT"]  # type: ignore[index]


def test_gemini_image_edit_sends_every_uploaded_image_as_its_own_inline_part(gateway: Gateway) -> None:
    """The caller uploads two images under `image[]`. Gemini must get two `inlineData` parts in upload order,
    followed by the text prompt, so multi-image composition keeps working."""
    first: Final = png((10, 0, 0, 255))
    second: Final = png((0, 10, 0, 255))
    with wire_server(_gemini_edit) as wire, gateway.scenario() as scenario:
        model: Final = _gemini_deployment(scenario, wire.url)
        response: Final = post_upload(
            gateway,
            _EDITS,
            {"model": model, "prompt": _prompt()},
            [("image[]", ("a.png", first, "image/png")), ("image[]", ("b.png", second, "image/png"))],
        )
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    parts: Final = json_body(request)["contents"][0]["parts"]  # type: ignore[index]
    assert [part["inlineData"]["data"] for part in parts[:2]] == [_b64(first), _b64(second)]  # type: ignore[index]
    assert "text" in parts[2]  # type: ignore[operator]


def test_gemini_image_edit_returns_only_the_inline_images_as_b64_json(gateway: Gateway) -> None:
    """Gemini answers one text part and one inline image. The caller must get `data` with exactly one entry whose
    `b64_json` is the image data; the text part is not returned as an image."""
    with wire_server(_gemini_edit) as wire, gateway.scenario() as scenario:
        model: Final = _gemini_deployment(scenario, wire.url)
        response: Final = post_upload(
            gateway, _EDITS, {"model": model, "prompt": _prompt()}, [("image", ("a.png", png(), "image/png"))]
        )
        assert response.status_code == 200, response.text
        data: Final = JSON_OBJECT.validate_json(response.content)["data"]
        assert len(wire.drain()) == 1
    assert [entry["b64_json"] for entry in data] == [B64_IMAGE]  # type: ignore[index, union-attr]


def test_gemini_image_edit_without_an_image_answers_400_before_calling_gemini(gateway: Gateway) -> None:
    """The caller posts an edit with a prompt and no image part. The proxy must answer 400 with
    `Gemini image edit requires at least one image.` and not call Gemini; before #52087 this was a 500."""
    with wire_server(_gemini_edit) as wire, gateway.scenario() as scenario:
        model: Final = _gemini_deployment(scenario, wire.url)
        response: Final = gateway.client.post(
            _EDITS,
            data={"model": model, "prompt": _prompt()},
            files={"unused": ("x.txt", b"x", "text/plain")},
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
        assert response.status_code == 400, response.text
        assert "Gemini image edit requires at least one image." in error_message(response)
        assert wire.drain() == ()


def test_gemini_image_edit_with_n_2_sends_two_requests_and_merges_both_images(gateway: Gateway) -> None:
    """The caller asks for `n=2`. Gemini rejects a candidate count above one, so the proxy must send two identical
    `generateContent` requests (neither carrying `sampleCount` or `candidateCount`) and answer with both images in
    `data`, and `usage.total_tokens` equal to the two replies' totals added (2 x 1310)."""
    with wire_server(_gemini_edit) as wire, gateway.scenario() as scenario:
        model: Final = _gemini_deployment(scenario, wire.url)
        response: Final = post_upload(
            gateway,
            _EDITS,
            {"model": model, "prompt": _prompt(), "n": "2"},
            [("image", ("a.png", png(), "image/png"))],
        )
        assert response.status_code == 200, response.text
        requests: Final = wire.drain()
        body: Final = JSON_OBJECT.validate_json(response.content)
    assert len(requests) == 2
    assert json_body(requests[0]) == json_body(requests[1])
    config: Final = json_body(requests[0])["generationConfig"]
    assert "candidateCount" not in config and "sampleCount" not in config  # type: ignore[operator]
    assert len(body["data"]) == 2  # type: ignore[arg-type]
    assert body["usage"]["total_tokens"] == 2 * 1310  # type: ignore[index]


def test_gemini_image_edit_with_n_above_the_fan_out_cap_answers_400_without_any_request(gateway: Gateway) -> None:
    """The caller asks for `n=11`, above `MAX_IMAGE_FAN_OUT_REQUESTS` (10). The proxy must answer 400 with
    `n=11 is above the maximum of 10 images per request.` and send nothing to Gemini (litellm/images/fan_out.py)."""
    with wire_server(_gemini_edit) as wire, gateway.scenario() as scenario:
        model: Final = _gemini_deployment(scenario, wire.url)
        response: Final = post_upload(
            gateway,
            _EDITS,
            {"model": model, "prompt": _prompt(), "n": "11"},
            [("image", ("a.png", png(), "image/png"))],
        )
        assert response.status_code == 400, response.text
        assert "n=11 is above the maximum of 10 images per request." in error_message(response)
        assert wire.drain() == ()


def test_gemini_image_edit_fan_out_with_one_failed_request_fails_the_call(gateway: Gateway) -> None:
    """With `n=2`, Gemini answers the first request with an image and the second with 429. The caller must get an
    error status (the failed request's status, 429), not a partial 200 with one image; the served image is billed
    through the failure row instead (`record_partial_image_spend`).
    UNVERIFIED: the two requests run concurrently, so which one fails depends on arrival order; the test fails the
    second request to reach the peer."""
    calls: Final[list[int]] = []

    def respond(_request: Request) -> Reply:
        calls.append(1)
        if len(calls) == 2:
            return json_reply(
                {"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}}, 429
            )
        return gemini_image_reply(B64_IMAGE)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _gemini_deployment(scenario, wire.url)
        response: Final = post_upload(
            gateway,
            _EDITS,
            {"model": model, "prompt": _prompt(), "n": "2"},
            [("image", ("a.png", png(), "image/png"))],
        )
        assert response.status_code == 429, response.text
        assert len(wire.drain()) >= 2


# ---------------------------------------------------------------- Vertex AI Gemini


def test_vertex_gemini_image_edit_posts_contents_as_one_user_object_to_the_api_base(gateway: Gateway) -> None:
    """A `vertex_ai/gemini-2.5-flash-image` deployment whose `api_base` is the full `...:generateContent` URL edits one
    PNG. Vertex must get one POST to exactly that path with `contents` `{"role": "USER", "parts": [<inlineData>,
    {"text": <prompt>}]}` (an object, not a list) and `generationConfig.response_modalities` `["IMAGE"]`. Because
    `api_base` is set, the config mints no token and sends no `Authorization` header."""
    image: Final = png((4, 5, 6, 255))
    prompt: Final = _prompt()
    path: Final = f"{models_path(GEMINI_IMAGE)}:generateContent"
    with vertex(gateway, _gemini_edit, backend=GEMINI_IMAGE, base_suffix=path) as rig:
        response: Final = post_upload(
            gateway, _EDITS, {"model": rig.model, "prompt": prompt}, [("image", ("a.png", image, "image/png"))]
        )
        assert response.status_code == 200, response.text
        request: Final = rig.peer.one_call()
        assert rig.peer.token_calls() == ()
    assert path_of(request) == path
    assert "authorization" not in request.headers
    body: Final = json_body(request)
    assert body["contents"] == {
        "role": "USER",
        "parts": [{"inlineData": {"mimeType": "image/png", "data": _b64(image)}}, {"text": prompt}],
    }
    assert body["generationConfig"]["response_modalities"] == ["IMAGE"]  # type: ignore[index]


def test_vertex_gemini_image_edit_returns_gemini_token_usage(gateway: Gateway) -> None:
    """Vertex answers with `usageMetadata` of 20 prompt tokens and 1290 image candidate tokens. The caller's
    `usage` must report `input_tokens` 20, `output_tokens` 1290 and `total_tokens` 1310, the token usage that
    #51038 started returning so the edit is billed by tokens instead of per image."""
    path: Final = f"{models_path(GEMINI_IMAGE)}:generateContent"
    with vertex(gateway, _gemini_edit, backend=GEMINI_IMAGE, base_suffix=path) as rig:
        response: Final = post_upload(
            gateway, _EDITS, {"model": rig.model, "prompt": _prompt()}, [("image", ("a.png", png(), "image/png"))]
        )
        assert response.status_code == 200, response.text
        usage: Final = JSON_OBJECT.validate_json(response.content)["usage"]
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (20, 1290, 1310)  # type: ignore[index]


def test_vertex_gemini_image_edit_without_an_image_answers_400(gateway: Gateway) -> None:
    """An edit with no image to a Vertex Gemini image deployment must answer 400 with
    `Vertex AI Gemini image edit requires at least one image.` and send nothing to Vertex."""
    path: Final = f"{models_path(GEMINI_IMAGE)}:generateContent"
    with vertex(gateway, _gemini_edit, backend=GEMINI_IMAGE, base_suffix=path) as rig:
        response: Final = gateway.client.post(
            _EDITS,
            data={"model": rig.model, "prompt": _prompt()},
            files={"unused": ("x.txt", b"x", "text/plain")},
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
        assert response.status_code == 400, response.text
        assert "Vertex AI Gemini image edit requires at least one image." in error_message(response)
        assert rig.peer.model_calls() == ()


def test_vertex_gemini_image_edit_with_n_2_sends_two_requests(gateway: Gateway) -> None:
    """Vertex Gemini edits accept `n` since #52103. With `n=2` Vertex must get two identical requests and the caller
    two images in `data`."""
    path: Final = f"{models_path(GEMINI_IMAGE)}:generateContent"
    with vertex(gateway, _gemini_edit, backend=GEMINI_IMAGE, base_suffix=path) as rig:
        response: Final = post_upload(
            gateway,
            _EDITS,
            {"model": rig.model, "prompt": _prompt(), "n": "2"},
            [("image", ("a.png", png(), "image/png"))],
        )
        assert response.status_code == 200, response.text
        assert len(rig.peer.model_calls()) == 2
        assert len(JSON_OBJECT.validate_json(response.content)["data"]) == 2  # type: ignore[arg-type]


# ---------------------------------------------------------------- Vertex AI Imagen


def _imagen_reply(_request: Request) -> Reply:
    return json_reply(
        {"predictions": [{"bytesBase64Encoded": B64_IMAGE, "mimeType": "image/png"}, {"raiFilteredReason": "x"}]}
    )


def test_vertex_imagen_edit_sends_raw_reference_image_mask_and_fixed_edit_parameters(gateway: Gateway) -> None:
    """A `vertex_ai/imagegeneration@006` deployment edits an image with a mask and `n=2`. Vertex must get one POST to
    `/v1/projects/<project>/locations/us-central1/publishers/google/models/imagegeneration@006:predict` with
    `instances[0]` holding the prompt, the image as `REFERENCE_TYPE_RAW` id 1 and the mask as `REFERENCE_TYPE_MASK`
    id 2 with `maskMode` `MASK_MODE_USER_PROVIDED` and `dilation` 0.03, and `parameters` `{"sampleCount": 2,
    "editMode": "EDIT_MODE_INPAINT_INSERTION", "editConfig": {"baseSteps": 50}, "guidanceScale": 7.5, "seed": null}`."""
    image: Final = png((7, 7, 7, 255))
    mask: Final = png((0, 0, 0, 0))
    prompt: Final = _prompt()
    with vertex(gateway, _imagen_reply, backend=IMAGEN) as rig:
        response: Final = post_upload(
            gateway,
            _EDITS,
            {"model": rig.model, "prompt": prompt, "n": "2"},
            [("image", ("a.png", image, "image/png")), ("mask", ("m.png", mask, "image/png"))],
        )
        assert response.status_code == 200, response.text
        request: Final = rig.peer.one_call()
    assert path_of(request) == f"/v1/projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{IMAGEN}:predict"
    expected_instance: Final[dict[str, JsonValue]] = {
        "prompt": prompt,
        "referenceImages": [
            {
                "referenceType": "REFERENCE_TYPE_RAW",
                "referenceId": 1,
                "referenceImage": {"bytesBase64Encoded": _b64(image)},
            },
            {
                "referenceType": "REFERENCE_TYPE_MASK",
                "referenceId": 2,
                "referenceImage": {"bytesBase64Encoded": _b64(mask)},
                "maskImageConfig": {"maskMode": "MASK_MODE_USER_PROVIDED", "dilation": 0.03},
            },
        ],
    }
    body: Final = json_body(request)
    assert body["instances"] == [expected_instance]
    assert body["parameters"] == {
        "sampleCount": 2,
        "editMode": "EDIT_MODE_INPAINT_INSERTION",
        "editConfig": {"baseSteps": 50},
        "guidanceScale": 7.5,
        "seed": None,
    }


def test_vertex_imagen_edit_returns_only_predictions_that_carry_image_bytes(gateway: Gateway) -> None:
    """Imagen answers one prediction with `bytesBase64Encoded` and one filtered prediction without bytes. The caller
    must get `data` with exactly one `b64_json` entry; the filtered prediction is dropped, not returned empty."""
    with vertex(gateway, _imagen_reply, backend=IMAGEN) as rig:
        response: Final = post_upload(
            gateway, _EDITS, {"model": rig.model, "prompt": _prompt()}, [("image", ("a.png", png(), "image/png"))]
        )
        assert response.status_code == 200, response.text
        data: Final = JSON_OBJECT.validate_json(response.content)["data"]
    assert [entry["b64_json"] for entry in data] == [B64_IMAGE]  # type: ignore[index, union-attr]


def test_vertex_imagen_edit_without_n_sends_sample_count_1(gateway: Gateway) -> None:
    """Without `n`, Imagen must get `parameters.sampleCount` 1 (the transformation's default) and no mask reference."""
    with vertex(gateway, _imagen_reply, backend=IMAGEN) as rig:
        response: Final = post_upload(
            gateway, _EDITS, {"model": rig.model, "prompt": _prompt()}, [("image", ("a.png", png(), "image/png"))]
        )
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
    assert body["parameters"]["sampleCount"] == 1  # type: ignore[index]
    references: Final = body["instances"][0]["referenceImages"]  # type: ignore[index]
    assert [reference["referenceType"] for reference in references] == ["REFERENCE_TYPE_RAW"]  # type: ignore[index]


# ---------------------------------------------------------------- Azure AI Foundry gpt-image


def test_azure_ai_gpt_image_edit_uses_the_azure_openai_route_with_the_foundry_default_api_version(
    gateway: Gateway,
) -> None:
    """An `azure_ai/gpt-image-1` deployment with no `api_version` edits an image. Foundry must get one POST to
    `/openai/deployments/gpt-image-1/images/edits?api-version=2025-04-01-preview` with the `api-key` header and the
    upload in a file part, and the caller gets the image back (#49650)."""
    image: Final = png((9, 9, 9, 255))
    with (
        wire_server(lambda _r: Reply(body=image_reply_body())) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model="azure_ai/gpt-image-1", api_base=wire.url, api_key=AZURE_KEY)
        response: Final = post_upload(
            gateway, _EDITS, {"model": model, "prompt": _prompt()}, [("image", ("a.png", image, "image/png"))]
        )
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["data"][0]["b64_json"] == B64_IMAGE  # type: ignore[index]
        (request,) = wire.drain()
    assert (request.method, path_of(request)) == ("POST", "/openai/deployments/gpt-image-1/images/edits")
    assert query_of(request)["api-version"] == ["2025-04-01-preview"]
    assert request.headers["api-key"] == AZURE_KEY
    assert [part.content for part in file_parts(request)] == [image]


def test_azure_ai_gpt_image_edit_honours_a_deployment_api_version(gateway: Gateway) -> None:
    """The same deployment with `api_version` `2025-06-01` must send `api-version=2025-06-01`, so the Foundry default
    applies only when nothing is configured."""
    with wire_server(lambda _r: Reply(body=image_reply_body())) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure_ai/gpt-image-1", api_base=wire.url, api_key=AZURE_KEY, api_version="2025-06-01"
        )
        response: Final = post_upload(
            gateway, _EDITS, {"model": model, "prompt": _prompt()}, [("image", ("a.png", png(), "image/png"))]
        )
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert query_of(request)["api-version"] == ["2025-06-01"]


@pytest.mark.parametrize("field", ("image", "mask"))
def test_image_edit_with_image_and_bracketed_spelling_together_answers_422(gateway: Gateway, field: str) -> None:
    """The caller sends both `<field>` and `<field>[]` file parts. The proxy must answer 422 with
    `Cannot specify both '<field>' and '<field>[]'` before any provider work."""
    with wire_server(_gemini_edit) as wire, gateway.scenario() as scenario:
        model: Final = _gemini_deployment(scenario, wire.url)
        files: Final = [("image", ("a.png", png(), "image/png"))] if field == "mask" else []
        response: Final = post_upload(
            gateway,
            _EDITS,
            {"model": model, "prompt": _prompt()},
            [*files, (field, ("x.png", png(), "image/png")), (f"{field}[]", ("y.png", png(), "image/png"))],
        )
        assert response.status_code == 422, response.text
        assert f"Cannot specify both '{field}' and '{field}[]'" in response.text
        assert wire.drain() == ()
