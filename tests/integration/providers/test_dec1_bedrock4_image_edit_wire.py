"""`POST /v1/images/edits` on AWS Bedrock: Stability edit models and Nova Canvas edit, one JSON InvokeModel call.

The proxy reads the multipart edit into in-memory uploads (`litellm/proxy/image_endpoints/endpoints.py:342`),
and for a `bedrock/...` deployment hands it to `BedrockImageEdit` (`litellm/images/main.py:1054`), which
picks a config by model id (`litellm/llms/bedrock/image_edit/handler.py:63`): a `stability.` id naming an edit
operation (inpaint, outpaint, upscale, erase, remove-background, search-replace, control-sketch, style
transfer, ...) is `BedrockStabilityImageEditConfig`; an id with `supports_nova_canvas_image_edit` in the
cost map is Nova Canvas; anything else is refused. Unlike OpenAI the upstream is JSON, not multipart: the
image and mask become base64 strings in the body sent to `<api_base>/model/<model id>/invoke`.

Form fields arrive as strings; only the OpenAI numeric fields (`n`) are coerced by the proxy. The Stability
config coerces its own numeric fields (`stability_transformation.py:262`); Nova Canvas does not, which is
pinned below as a suspected bug. None of these routes had an integration test before this file.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_bedrock4_support import (
    CONTROL_SKETCH,
    IMAGE_ONE,
    IMAGE_TWO,
    INPAINT,
    NOVA_CANVAS,
    OUTPAINT,
    REMOVE_BACKGROUND,
    STYLE_TRANSFER,
    TITAN_IMAGE_V2,
    b64,
    error_text,
    image_model,
    json_body,
    json_reply,
    path_of,
    png,
    prompt,
)
from integration.providers.dec1_bedrock_support import TOKEN

IMAGE: Final = png((200, 30, 30, 255))
MASK: Final = png((0, 0, 0, 0))


def _stability_ok(_request: Request) -> Reply:
    return json_reply({"images": [IMAGE_ONE], "finish_reasons": [None], "seeds": [7]})


def _nova_ok(_request: Request) -> Reply:
    return json_reply({"images": [IMAGE_ONE, IMAGE_TWO]})


def _image_file() -> tuple[str, tuple[str, bytes, str]]:
    return ("image", ("photo.png", IMAGE, "image/png"))


def _mask_file() -> tuple[str, tuple[str, bytes, str]]:
    return ("mask", ("mask.png", MASK, "image/png"))


# ------------------------------------------------------------------ Stability edit models


def test_stability_inpaint_sends_base64_image_mask_prompt_and_png_output(gateway: Gateway) -> None:
    """A `bedrock/stability.stable-image-inpaint-v1:0` deployment edits one PNG with a mask and a prompt.

    Bedrock must get one JSON POST to `/model/stability.stable-image-inpaint-v1:0/invoke` with the bearer key
    and exactly `{"output_format": "png", "prompt": <prompt>, "image": <base64 image>, "mask": <base64 mask>}`
    (`litellm/llms/bedrock/image_edit/stability_transformation.py:198-258`); the answer's image comes back as
    `data[0].b64_json`.
    """
    text: Final = prompt()
    with image_model(gateway, _stability_ok, backend=INPAINT) as rig:
        response: Final = rig.edit({"prompt": text}, [_image_file(), _mask_file()])
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
    assert path_of(sent) == f"/model/{INPAINT}/invoke"
    assert sent.headers["authorization"] == f"Bearer {TOKEN}"
    assert sent.headers["content-type"].startswith("application/json")
    assert json_body(sent) == {"output_format": "png", "prompt": text, "image": b64(IMAGE), "mask": b64(MASK)}
    assert response.json()["data"][0]["b64_json"] == IMAGE_ONE


def test_stability_outpaint_turns_string_pixel_fields_into_integers(gateway: Gateway) -> None:
    """An outpaint edit sends the form fields `left=128`, `up=64` and `seed=9`, which arrive as strings.

    The Stability config casts `left`, `right`, `up`, `down` and `seed` to int (`stability_transformation.py:262`),
    so the body must carry `"left": 128, "up": 64, "seed": 9` as JSON numbers.
    """
    with image_model(gateway, _stability_ok, backend=OUTPAINT) as rig:
        response: Final = rig.edit({"prompt": prompt(), "left": "128", "up": "64", "seed": "9"}, [_image_file()])
        assert response.status_code == 200, response.text
        body: Final = json_body(rig.one_post())
    assert (body["left"], body["up"], body["seed"]) == (128, 64, 9)


def test_stability_control_strength_is_sent_as_a_float(gateway: Gateway) -> None:
    """A control-sketch edit sends `control_strength=0.6` as a form string.

    It is in the float field list (`stability_transformation.py:263`), so the body must carry
    `"control_strength": 0.6` as a JSON number.
    """
    with image_model(gateway, _stability_ok, backend=CONTROL_SKETCH) as rig:
        response: Final = rig.edit({"prompt": prompt(), "control_strength": "0.6"}, [_image_file()])
        assert response.status_code == 200, response.text
        assert json_body(rig.one_post())["control_strength"] == 0.6


def test_stability_size_becomes_an_aspect_ratio_and_is_not_sent_itself(gateway: Gateway) -> None:
    """An inpaint edit sends `size=1792x1024`.

    `OPENAI_SIZE_TO_STABILITY_ASPECT_RATIO` maps it to `16:9` (`litellm/types/llms/stability.py:177`), so the
    body must carry `"aspect_ratio": "16:9"` and no `size` key (`stability_transformation.py:156`).
    """
    with image_model(gateway, _stability_ok, backend=INPAINT) as rig:
        response: Final = rig.edit({"prompt": prompt(), "size": "1792x1024"}, [_image_file(), _mask_file()])
        assert response.status_code == 200, response.text
        body: Final = json_body(rig.one_post())
    assert body["aspect_ratio"] == "16:9"
    assert "size" not in body


def test_stability_style_transfer_sends_the_upload_as_init_image(gateway: Gateway) -> None:
    """A `stability.stable-style-transfer-v1:0` edit uploads one image.

    For a model id containing `style-transfer` the upload is sent as `init_image`, not `image`
    (`stability_transformation.py:225`), so the body must have `init_image` equal to the base64 upload and
    no `image` key.
    """
    with image_model(gateway, _stability_ok, backend=STYLE_TRANSFER) as rig:
        response: Final = rig.edit({"prompt": prompt()}, [_image_file()])
        assert response.status_code == 200, response.text
        body: Final = json_body(rig.one_post())
    assert body["init_image"] == b64(IMAGE)
    assert "image" not in body


def test_stability_remove_background_without_a_prompt_sends_no_prompt_key(gateway: Gateway) -> None:
    """A remove-background edit uploads an image and sends no prompt at all.

    The proxy defaults `prompt` to `None` (`endpoints.py:47`) and the config adds `prompt` only when it is
    non-empty (`stability_transformation.py:203`), so the body must be exactly `{"output_format": "png",
    "image": <base64 image>}`.
    """
    with image_model(gateway, _stability_ok, backend=REMOVE_BACKGROUND) as rig:
        response: Final = rig.edit({}, [_image_file()])
        assert response.status_code == 200, response.text
        assert json_body(rig.one_post()) == {"output_format": "png", "image": b64(IMAGE)}


def test_stability_content_filtered_finish_reason_is_a_400(gateway: Gateway) -> None:
    """Bedrock answers 200 with `{"images": [], "finish_reasons": ["CONTENT_FILTERED"]}`.

    A truthy first finish reason raises `BedrockError(status_code=400)` with message
    `Bedrock Stability error: CONTENT_FILTERED` (`stability_transformation.py:336`), so the caller must get
    400 with that text, never a 200 with no images.
    """

    def filtered(_request: Request) -> Reply:
        return json_reply({"images": [], "finish_reasons": ["CONTENT_FILTERED"], "seeds": [1]})

    with image_model(gateway, filtered, backend=INPAINT) as rig:
        response: Final = rig.edit({"prompt": prompt()}, [_image_file(), _mask_file()])
        assert response.status_code == 400, response.text
        assert "Bedrock Stability error: CONTENT_FILTERED" in error_text(response), response.text


# ------------------------------------------------------------------ Nova Canvas edit


def test_nova_canvas_edit_without_a_mask_is_an_image_variation(gateway: Gateway) -> None:
    """A `bedrock/amazon.nova-canvas-v1:0` edit uploads one image with a prompt and no mask or task type.

    With no mask the config routes to `IMAGE_VARIATION` (`amazon_nova_canvas_image_edit_transformation.py:131`),
    so Bedrock must get exactly `{"taskType": "IMAGE_VARIATION", "imageVariationParams": {"images":
    [<base64 image>], "text": <prompt>}}` at `/model/amazon.nova-canvas-v1:0/invoke`, and both returned images
    come back in order.
    """
    text: Final = prompt()
    with image_model(gateway, _nova_ok, backend=NOVA_CANVAS) as rig:
        response: Final = rig.edit({"prompt": text}, [_image_file()])
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
    assert path_of(sent) == f"/model/{NOVA_CANVAS}/invoke"
    assert json_body(sent) == {
        "taskType": "IMAGE_VARIATION",
        "imageVariationParams": {"images": [b64(IMAGE)], "text": text},
    }
    assert [image["b64_json"] for image in response.json()["data"]] == [IMAGE_ONE, IMAGE_TWO]


def test_nova_canvas_mask_prompt_routes_to_inpainting(gateway: Gateway) -> None:
    """A Nova Canvas edit sends the form field `maskPrompt=the cat` and no mask file.

    A mask prompt selects `INPAINTING` (`amazon_nova_canvas_image_edit_transformation.py:110`), so the body
    must have `taskType: INPAINTING` and `inPaintingParams` equal to `{"image": <base64 image>, "text": <prompt>,
    "maskPrompt": "the cat"}`.
    """
    text: Final = prompt()
    with image_model(gateway, _nova_ok, backend=NOVA_CANVAS) as rig:
        response: Final = rig.edit({"prompt": text, "maskPrompt": "the cat"}, [_image_file()])
        assert response.status_code == 200, response.text
        body: Final = json_body(rig.one_post())
    assert body["taskType"] == "INPAINTING"
    assert body["inPaintingParams"] == {"image": b64(IMAGE), "text": text, "maskPrompt": "the cat"}


def test_nova_canvas_mask_upload_becomes_the_inpainting_mask_image(gateway: Gateway) -> None:
    """A Nova Canvas edit uploads an image and a `mask` file, with a prompt.

    The mask should become `inPaintingParams.maskImage` (base64 of the mask upload) under `taskType: INPAINTING`
    (`amazon_nova_canvas_image_edit_transformation.py:112`), as the Stability config already does for `mask`.
    """
    pytest.skip(
        "BUG: the proxy passes `mask` as a list of uploads (endpoints.py:343) and _file_types_to_b64 "
        "(amazon_nova_canvas_image_edit_transformation.py:137) has no list branch, so bytes([BytesIO]) raises "
        "TypeError and every masked Nova Canvas edit through the proxy fails"
    )
    text: Final = prompt()
    with image_model(gateway, _nova_ok, backend=NOVA_CANVAS) as rig:
        response: Final = rig.edit({"prompt": text}, [_image_file(), _mask_file()])
        assert response.status_code == 200, response.text
        body: Final = json_body(rig.one_post())
    assert body["taskType"] == "INPAINTING"
    assert body["inPaintingParams"] == {"image": b64(IMAGE), "text": text, "maskImage": b64(MASK)}


def test_nova_canvas_background_removal_never_sends_a_generation_config(gateway: Gateway) -> None:
    """A Nova Canvas edit sends `taskType=BACKGROUND_REMOVAL`, `n=2` and `size=1024x1024`.

    AWS rejects `imageGenerationConfig` on this task, so the body must be exactly `{"taskType":
    "BACKGROUND_REMOVAL", "backgroundRemovalParams": {"image": <base64 image>}}`
    (`amazon_nova_canvas_image_edit_transformation.py:394`) even though `n` and `size` were mapped.
    """
    with image_model(gateway, _nova_ok, backend=NOVA_CANVAS) as rig:
        response: Final = rig.edit({"taskType": "BACKGROUND_REMOVAL", "n": "2", "size": "1024x1024"}, [_image_file()])
        assert response.status_code == 200, response.text
        assert json_body(rig.one_post()) == {
            "taskType": "BACKGROUND_REMOVAL",
            "backgroundRemovalParams": {"image": b64(IMAGE)},
        }


def test_nova_canvas_n_and_size_become_the_generation_config(gateway: Gateway) -> None:
    """A Nova Canvas variation edit sends `n=2` (coerced to an int by the proxy) and `size=512x512`.

    The body's `imageGenerationConfig` must be `{"width": 512, "height": 512, "numberOfImages": 2}`
    (`amazon_nova_canvas_image_edit_transformation.py:269`, `:345`).
    """
    with image_model(gateway, _nova_ok, backend=NOVA_CANVAS) as rig:
        response: Final = rig.edit({"prompt": prompt(), "n": "2", "size": "512x512"}, [_image_file()])
        assert response.status_code == 200, response.text
        assert json_body(rig.one_post())["imageGenerationConfig"] == {"width": 512, "height": 512, "numberOfImages": 2}


def test_nova_canvas_numeric_form_fields_reach_bedrock_as_numbers(gateway: Gateway) -> None:
    """A Nova Canvas variation edit sends `seed=42`, `cfgScale=6.5` and `similarityStrength=0.7` as form fields.

    AWS requires numbers for all three, so the body should carry `imageGenerationConfig.seed: 42`,
    `imageGenerationConfig.cfgScale: 6.5` and `imageVariationParams.similarityStrength: 0.7` as JSON numbers.
    """
    pytest.skip(
        "BUG: the proxy coerces only ImageEditRequestParams numerics (endpoints.py:46) and the Nova Canvas edit "
        "config copies seed, cfgScale and similarityStrength verbatim (amazon_nova_canvas_image_edit_transformation.py:354), "
        "so they reach Bedrock as strings; the Stability config casts its own (stability_transformation.py:262)"
    )
    with image_model(gateway, _nova_ok, backend=NOVA_CANVAS) as rig:
        response: Final = rig.edit(
            {"prompt": prompt(), "seed": "42", "cfgScale": "6.5", "similarityStrength": "0.7"}, [_image_file()]
        )
        assert response.status_code == 200, response.text
        body: Final = json_body(rig.one_post())
    assert body["imageGenerationConfig"] == {"seed": 42, "cfgScale": 6.5}
    assert body["imageVariationParams"]["similarityStrength"] == 0.7


def test_nova_canvas_unknown_edit_task_type_never_reaches_bedrock(gateway: Gateway) -> None:
    """A Nova Canvas edit sends `taskType=TEXT_IMAGE`, a generation-only task.

    `_nova_canvas_task_body` raises `Unsupported Amazon Nova Canvas taskType: 'TEXT_IMAGE'`
    (`amazon_nova_canvas_image_edit_transformation.py:103`) before signing, so the call must fail with that text
    and the peer must see no POST. UNVERIFIED: the status of a ValueError on this route is not pinned.
    """
    with image_model(gateway, _nova_ok, backend=NOVA_CANVAS) as rig:
        response: Final = rig.edit({"prompt": prompt(), "taskType": "TEXT_IMAGE"}, [_image_file()])
        assert response.status_code >= 400, response.text
        assert "Unsupported Amazon Nova Canvas taskType" in error_text(response), response.text
        assert rig.posts() == ()


def test_a_bedrock_model_with_no_edit_config_is_refused_before_bedrock(gateway: Gateway) -> None:
    """A `bedrock/amazon.titan-image-generator-v2:0` deployment is sent an edit.

    Titan is neither a Stability edit model nor flagged `supports_nova_canvas_image_edit`, so config lookup must
    fail with `Unsupported Bedrock image-edit model` (`handler.py:69`,
    `amazon_nova_canvas_image_edit_transformation.py:529`) and no POST may reach the peer. UNVERIFIED: the
    status (400 or 500) is not pinned.
    """
    with image_model(gateway, _nova_ok, backend=TITAN_IMAGE_V2) as rig:
        response: Final = rig.edit({"prompt": prompt()}, [_image_file()])
        assert response.status_code >= 400, response.text
        assert "Unsupported Bedrock image-edit model" in error_text(response), response.text
        assert rig.posts() == ()
