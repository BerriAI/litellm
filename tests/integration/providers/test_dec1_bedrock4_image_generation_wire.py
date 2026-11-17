"""`POST /v1/images/generations` on AWS Bedrock: one InvokeModel call per request, shaped per model family.

The proxy routes a `bedrock/<model id>` image deployment to `BedrockImageGeneration`
(`litellm/images/main.py:600`), which picks a config class by substring of the model id
(`litellm/llms/bedrock/image_generation/image_handler.py:68`): `amazon.titan` is Titan, `amazon.nova-canvas`
is Nova Canvas, `sd3` or `stable-image` is Stability 3 (which includes Stable Image Ultra), and anything else
is Stability 1 (SDXL). Each class has its own short list of OpenAI parameters; any other OpenAI parameter
is a 400 before Bedrock is called (`litellm/utils.py:4827`), while provider fields the caller adds to the
JSON body are passed through to the config (`litellm/utils.py:6399`). The request always goes to
`<api_base>/model/<model id>/invoke`, and the answer always comes back as OpenAI `data[].b64_json`.

`test_dec1_media_image_generation_wire.py` already pins the basic Nova Canvas text-to-image body and the
`style` refusal; this file covers the other families, the Nova Canvas task types and error shapes. Cost is
left alone on purpose: image generation custom prices are currently lost (#52103).
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_bedrock4_support import (
    IMAGE_ONE,
    IMAGE_TWO,
    NOVA_CANVAS,
    SD3_LARGE,
    SDXL_V1,
    STABLE_IMAGE_ULTRA,
    TITAN_IMAGE_V2,
    error_text,
    image_model,
    json_body,
    json_reply,
    path_of,
    prompt,
)
from integration.providers.dec1_bedrock_support import TOKEN, aws_error


def _images(_request: Request) -> Reply:
    return json_reply({"images": [IMAGE_ONE, IMAGE_TWO]})


def _one_image(_request: Request) -> Reply:
    return json_reply({"images": [IMAGE_ONE]})


def _stability3(_request: Request) -> Reply:
    return json_reply({"images": [IMAGE_ONE], "finish_reasons": [None], "seeds": [42]})


def _artifacts(_request: Request) -> Reply:
    return json_reply({"result": "success", "artifacts": [{"base64": IMAGE_ONE, "seed": 1, "finishReason": "SUCCESS"}]})


# ------------------------------------------------------------------ Titan Image Generator


def test_titan_without_options_sends_text_image_with_an_empty_generation_config(gateway: Gateway) -> None:
    """A `bedrock/amazon.titan-image-generator-v2:0` deployment is called with only a prompt.

    Bedrock must get one POST to `/model/amazon.titan-image-generator-v2:0/invoke` with the bearer key and
    exactly `{"taskType": "TEXT_IMAGE", "textToImageParams": {"text": <prompt>}, "imageGenerationConfig": {}}`
    (`litellm/llms/bedrock/image_generation/amazon_titan_transformation.py:111`).
    """
    text: Final = prompt()
    with image_model(gateway, _one_image, backend=TITAN_IMAGE_V2) as rig:
        response: Final = rig.generate(text)
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
    assert path_of(sent) == f"/model/{TITAN_IMAGE_V2}/invoke"
    assert sent.headers["authorization"] == f"Bearer {TOKEN}"
    assert json_body(sent) == {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": text},
        "imageGenerationConfig": {},
    }


def test_titan_maps_size_n_and_hd_quality_to_its_generation_config(gateway: Gateway) -> None:
    """Titan is called with `size: 512x768`, `n: 2`, `quality: hd`.

    `imageGenerationConfig` must be `{"width": 512, "height": 768, "numberOfImages": 2, "quality": "premium"}`
    (`amazon_titan_transformation.py:93`), and both returned images come back as `data[].b64_json` in order.
    """
    with image_model(gateway, _images, backend=TITAN_IMAGE_V2) as rig:
        response: Final = rig.generate(prompt(), size="512x768", n=2, quality="hd")
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
    assert json_body(sent)["imageGenerationConfig"] == {
        "width": 512,
        "height": 768,
        "numberOfImages": 2,
        "quality": "premium",
    }
    assert [image["b64_json"] for image in response.json()["data"]] == [IMAGE_ONE, IMAGE_TWO]


@pytest.mark.parametrize(
    ("quality", "sent"),
    [
        ("high", "premium"),
        ("premium", "premium"),
        ("medium", "standard"),
        ("low", "standard"),
        ("standard", "standard"),
    ],
)
def test_titan_folds_every_openai_quality_into_standard_or_premium(gateway: Gateway, quality: str, sent: str) -> None:
    """Titan has two quality levels. `high` and `premium` must reach Bedrock as `premium`; `medium`, `low`
    and `standard` as `standard` (`amazon_titan_transformation.py:100`). A port that forwards the OpenAI
    word verbatim gets a ValidationException from Titan.
    """
    with image_model(gateway, _one_image, backend=TITAN_IMAGE_V2) as rig:
        response: Final = rig.generate(prompt(), quality=quality)
        assert response.status_code == 200, response.text
        assert json_body(rig.one_post())["imageGenerationConfig"] == {"quality": sent}


def test_titan_auto_quality_is_dropped_rather_than_forwarded(gateway: Gateway) -> None:
    """Titan is called with `quality: auto`, which has no Titan meaning.

    The value matches neither branch of `map_openai_params` (`amazon_titan_transformation.py:100`), so the
    call must succeed and `imageGenerationConfig` must be `{}`: no `quality` key at all.
    """
    with image_model(gateway, _one_image, backend=TITAN_IMAGE_V2) as rig:
        response: Final = rig.generate(prompt(), quality="auto")
        assert response.status_code == 200, response.text
        assert json_body(rig.one_post())["imageGenerationConfig"] == {}


def test_titan_negative_text_moves_into_text_to_image_params(gateway: Gateway) -> None:
    """The caller adds the Titan field `negativeText: "blurry, text"` to the JSON body.

    It reaches `optional_params` as a provider field and must land in `textToImageParams` next to the prompt,
    not at the top level (`amazon_titan_transformation.py:117`).
    """
    text: Final = prompt()
    with image_model(gateway, _one_image, backend=TITAN_IMAGE_V2) as rig:
        response: Final = rig.generate(text, negativeText="blurry, text")
        assert response.status_code == 200, response.text
        body: Final = json_body(rig.one_post())
    assert body["textToImageParams"] == {"text": text, "negativeText": "blurry, text"}
    assert "negativeText" not in body


def test_titan_caller_generation_config_is_merged_with_the_mapped_size(gateway: Gateway) -> None:
    """Titan is called with `size: 1024x1024` and a caller `imageGenerationConfig: {"cfgScale": 8.0, "seed": 7}`.

    `transform_request_body` is written to merge the mapped config with the caller's
    (`amazon_titan_transformation.py:116-126`), so Bedrock should get width, height, cfgScale and seed together.
    """
    pytest.skip(
        "BUG: the caller imageGenerationConfig replaces the mapped one in add_provider_specific_params_to_optional_params "
        "(litellm/utils.py:6403), so size is lost and the second pop at amazon_titan_transformation.py:122 is dead code"
    )
    with image_model(gateway, _one_image, backend=TITAN_IMAGE_V2) as rig:
        response: Final = rig.generate(prompt(), size="1024x1024", imageGenerationConfig={"cfgScale": 8.0, "seed": 7})
        assert response.status_code == 200, response.text
        assert json_body(rig.one_post())["imageGenerationConfig"] == {
            "width": 1024,
            "height": 1024,
            "cfgScale": 8.0,
            "seed": 7,
        }


def test_titan_refuses_response_format_with_a_400_before_bedrock(gateway: Gateway) -> None:
    """Titan supports only `size`, `n` and `quality` (`amazon_titan_transformation.py:84`).

    A call that sets `response_format: b64_json` must answer 400 with a message containing
    `Setting \\`response_format\\` is not supported by bedrock` (`litellm/utils.py:4812`), and the peer must
    see no POST: Bedrock always answers base64, but the option is still refused rather than ignored.
    """
    with image_model(gateway, _one_image, backend=TITAN_IMAGE_V2) as rig:
        response: Final = rig.generate(prompt(), response_format="b64_json")
        assert response.status_code == 400, response.text
        assert "Setting `response_format` is not supported by bedrock" in error_text(response), response.text
        assert rig.posts() == ()


def test_guardrail_fields_become_bedrock_guardrail_headers_and_leave_the_body(gateway: Gateway) -> None:
    """The caller adds `guardrailIdentifier: gr-dec1` and `guardrailVersion: "3"` to a Titan generation.

    `_extract_headers_from_optional_params` (`image_handler.py:192`) must move them to the
    `x-amz-bedrock-guardrail-identifier` and `x-amz-bedrock-guardrail-version` request headers, and neither
    field may appear anywhere in the JSON body (Bedrock rejects unknown body keys).
    """
    with image_model(gateway, _one_image, backend=TITAN_IMAGE_V2) as rig:
        response: Final = rig.generate(prompt(), guardrailIdentifier="gr-dec1", guardrailVersion="3")
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
    assert sent.headers["x-amz-bedrock-guardrail-identifier"] == "gr-dec1"
    assert sent.headers["x-amz-bedrock-guardrail-version"] == "3"
    assert "guardrail" not in sent.body.decode().lower()


# ------------------------------------------------------------------ Nova Canvas task types


def test_nova_canvas_extra_fields_land_in_the_generation_config(gateway: Gateway) -> None:
    """Nova Canvas is called with `size: 1024x1024` plus the provider fields `cfgScale: 6.5` and `seed: 12`.

    Nova merges every leftover optional param into `imageGenerationConfig` (`amazon_nova_canvas_transformation.py:88`),
    so it must be exactly `{"width": 1024, "height": 1024, "cfgScale": 6.5, "seed": 12}` and neither field may
    appear at the top level.
    """
    with image_model(gateway, _one_image, backend=NOVA_CANVAS) as rig:
        response: Final = rig.generate(prompt(), size="1024x1024", cfgScale=6.5, seed=12)
        assert response.status_code == 200, response.text
        body: Final = json_body(rig.one_post())
    assert body["imageGenerationConfig"] == {"width": 1024, "height": 1024, "cfgScale": 6.5, "seed": 12}
    assert set(body) == {"taskType", "textToImageParams", "imageGenerationConfig"}


def test_nova_canvas_caller_text_to_image_params_keep_the_prompt_as_text(gateway: Gateway) -> None:
    """The caller sends `textToImageParams: {"negativeText": "people", "style": "PHOTOREALISM"}`.

    The prompt is put first and the caller's fields merged after it (`amazon_nova_canvas_transformation.py:90`),
    so Bedrock must get `textToImageParams` equal to `{"text": <prompt>, "negativeText": "people",
    "style": "PHOTOREALISM"}` with `taskType: TEXT_IMAGE`.
    """
    text: Final = prompt()
    with image_model(gateway, _one_image, backend=NOVA_CANVAS) as rig:
        response: Final = rig.generate(text, textToImageParams={"negativeText": "people", "style": "PHOTOREALISM"})
        assert response.status_code == 200, response.text
        body: Final = json_body(rig.one_post())
    assert body["taskType"] == "TEXT_IMAGE"
    assert body["textToImageParams"] == {"text": text, "negativeText": "people", "style": "PHOTOREALISM"}


def test_nova_canvas_color_guided_generation_sends_the_colors_with_the_prompt(gateway: Gateway) -> None:
    """The caller sends `taskType: COLOR_GUIDED_GENERATION` and `colorGuidedGenerationParams: {"colors":
    ["#FF5500", "#0055FF"]}` with `n: 1`.

    Bedrock must get `{"taskType": "COLOR_GUIDED_GENERATION", "colorGuidedGenerationParams": {"text": <prompt>,
    "colors": [...]}, "imageGenerationConfig": {"numberOfImages": 1}}` and no `textToImageParams`
    (`amazon_nova_canvas_transformation.py:113`).
    """
    text: Final = prompt()
    with image_model(gateway, _one_image, backend=NOVA_CANVAS) as rig:
        response: Final = rig.generate(
            text,
            n=1,
            taskType="COLOR_GUIDED_GENERATION",
            colorGuidedGenerationParams={"colors": ["#FF5500", "#0055FF"]},
        )
        assert response.status_code == 200, response.text
        body: Final = json_body(rig.one_post())
    assert body == {
        "taskType": "COLOR_GUIDED_GENERATION",
        "colorGuidedGenerationParams": {"text": text, "colors": ["#FF5500", "#0055FF"]},
        "imageGenerationConfig": {"numberOfImages": 1},
    }


def test_nova_canvas_inpainting_generation_sends_aws_in_painting_params(gateway: Gateway) -> None:
    """The caller asks the generation route for `taskType: INPAINTING` with `inpaintingParams: {"image": "aW1n",
    "maskPrompt": "the sky"}` (the key the generation config reads, `amazon_nova_canvas_transformation.py:143`).

    AWS Nova Canvas names the block `inPaintingParams` (capital P), which the edit route already sends
    (`litellm/llms/bedrock/image_edit/amazon_nova_canvas_image_edit_transformation.py:122`), so Bedrock should
    get `inPaintingParams: {"text": <prompt>, "image": "aW1n", "maskPrompt": "the sky"}`.
    """
    pytest.skip(
        "BUG: AmazonNovaCanvasInpaintingRequest declares `inpaintingParams` (litellm/types/llms/bedrock.py:1009), "
        "so the generation route sends a key Nova Canvas does not define instead of `inPaintingParams`"
    )
    text: Final = prompt()
    with image_model(gateway, _one_image, backend=NOVA_CANVAS) as rig:
        response: Final = rig.generate(
            text, taskType="INPAINTING", inpaintingParams={"image": "aW1n", "maskPrompt": "the sky"}
        )
        assert response.status_code == 200, response.text
        body: Final = json_body(rig.one_post())
    assert body["taskType"] == "INPAINTING"
    assert body["inPaintingParams"] == {"text": text, "image": "aW1n", "maskPrompt": "the sky"}
    assert "inpaintingParams" not in body


def test_nova_canvas_unknown_task_type_never_reaches_bedrock(gateway: Gateway) -> None:
    """The caller asks the generation route for `taskType: OUTPAINTING`, which only the edit route knows.

    `transform_request_body` raises `NotImplementedError("Task type OUTPAINTING is not supported")`
    (`amazon_nova_canvas_transformation.py:166`) before the request is signed, so the call must fail and the
    peer must see no POST. UNVERIFIED: the status (likely 500, since NotImplementedError has no mapping)
    is not pinned, only that it is an error that names the task type.
    """
    with image_model(gateway, _one_image, backend=NOVA_CANVAS) as rig:
        response: Final = rig.generate(prompt(), taskType="OUTPAINTING")
        assert response.status_code >= 400, response.text
        assert "OUTPAINTING" in error_text(response), response.text
        assert rig.posts() == ()


def test_nova_canvas_medium_quality_is_silently_not_sent(gateway: Gateway) -> None:
    """Nova Canvas maps only `hd`/`premium` and `standard` (`amazon_nova_canvas_transformation.py:182`).

    `quality: medium` is an accepted OpenAI parameter but matches neither branch, so the call must succeed
    and `imageGenerationConfig` must be `{}`. Unlike Titan, Nova does not fold `medium` into `standard`.
    """
    with image_model(gateway, _one_image, backend=NOVA_CANVAS) as rig:
        response: Final = rig.generate(prompt(), quality="medium")
        assert response.status_code == 200, response.text
        assert json_body(rig.one_post())["imageGenerationConfig"] == {}


# ------------------------------------------------------------------ Stability 3 and Ultra


def test_stability3_sends_the_prompt_and_every_provider_field_verbatim(gateway: Gateway) -> None:
    """A `bedrock/stability.sd3-5-large-v1:0` deployment is called with `aspect_ratio: "16:9"`, `seed: 3`,
    `negative_prompt: "fog"` and `output_format: "jpeg"`.

    Stability 3 maps no OpenAI parameter and spreads the rest into the body
    (`amazon_stability3_transformation.py:241`), so Bedrock must get exactly those four keys plus
    `prompt`, at `/model/stability.sd3-5-large-v1:0/invoke`.
    """
    text: Final = prompt()
    with image_model(gateway, _stability3, backend=SD3_LARGE) as rig:
        response: Final = rig.generate(text, aspect_ratio="16:9", seed=3, negative_prompt="fog", output_format="jpeg")
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
    assert path_of(sent) == f"/model/{SD3_LARGE}/invoke"
    assert json_body(sent) == {
        "prompt": text,
        "aspect_ratio": "16:9",
        "seed": 3,
        "negative_prompt": "fog",
        "output_format": "jpeg",
    }
    assert [image["b64_json"] for image in response.json()["data"]] == [IMAGE_ONE]


def test_stable_image_ultra_uses_the_stability3_body(gateway: Gateway) -> None:
    """`stability.stable-image-ultra-v1:1` contains `stable-image`, so `_is_stability_3_model` is true
    (`amazon_stability3_transformation.py:225`). With only a prompt Bedrock must get exactly `{"prompt": <prompt>}`,
    not the SDXL `text_prompts` list.
    """
    text: Final = prompt()
    with image_model(gateway, _stability3, backend=STABLE_IMAGE_ULTRA) as rig:
        response: Final = rig.generate(text)
        assert response.status_code == 200, response.text
        assert json_body(rig.one_post()) == {"prompt": text}


def test_stability3_refuses_size_with_a_400_before_bedrock(gateway: Gateway) -> None:
    """Stability 3 supports no OpenAI parameter (`amazon_stability3_transformation.py:201`).

    `size: 1024x1024` must answer 400 with `Setting \\`size\\` is not supported by bedrock` and no POST may
    reach the peer: callers must use the provider field `aspect_ratio` instead.
    """
    with image_model(gateway, _stability3, backend=SD3_LARGE) as rig:
        response: Final = rig.generate(prompt(), size="1024x1024")
        assert response.status_code == 400, response.text
        assert "Setting `size` is not supported by bedrock" in error_text(response), response.text
        assert rig.posts() == ()


def test_stability3_finish_reason_is_raised_as_a_400_naming_the_reason(gateway: Gateway) -> None:
    """Bedrock answers 200 with `{"images": [], "finish_reasons": ["Filter reason: prompt"], "seeds": [1]}`.

    Any non-null finish reason is a `BedrockError(status_code=400)` whose message is the joined reasons
    (`amazon_stability3_transformation.py:265`), so the caller must get 400 with `Filter reason: prompt` in
    the error message, never a 200 with an empty `data` list.
    """

    def filtered(_request: Request) -> Reply:
        return json_reply({"images": [], "finish_reasons": ["Filter reason: prompt"], "seeds": [1]})

    with image_model(gateway, filtered, backend=SD3_LARGE) as rig:
        response: Final = rig.generate(prompt())
        assert response.status_code == 400, response.text
        assert "Filter reason: prompt" in error_text(response), response.text


# ------------------------------------------------------------------ Stability 1 (SDXL)


def test_sdxl_sends_weighted_text_prompts_and_the_mapped_size(gateway: Gateway) -> None:
    """A `bedrock/stability.stable-diffusion-xl-v1` deployment is called with `size: 1024x1024` and the
    provider fields `cfg_scale: 10` and `steps: 30`.

    Bedrock must get exactly `{"text_prompts": [{"text": <prompt>, "weight": 1}], "width": 1024,
    "height": 1024, "cfg_scale": 10, "steps": 30}` (`amazon_stability1_transformation.py:116`), and the
    `artifacts[0].base64` of the answer must come back as `data[0].b64_json`.
    """
    text: Final = prompt()
    with image_model(gateway, _artifacts, backend=SDXL_V1) as rig:
        response: Final = rig.generate(text, size="1024x1024", cfg_scale=10, steps=30)
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
    assert path_of(sent) == f"/model/{SDXL_V1}/invoke"
    assert json_body(sent) == {
        "text_prompts": [{"text": text, "weight": 1}],
        "width": 1024,
        "height": 1024,
        "cfg_scale": 10,
        "steps": 30,
    }
    assert response.json()["data"][0]["b64_json"] == IMAGE_ONE


def test_sdxl_replaces_newlines_in_the_prompt_with_spaces(gateway: Gateway) -> None:
    """The prompt is `a red fox\\nin the snow <marker>`.

    `transform_request_body` replaces `os.linesep` with a space (`amazon_stability1_transformation.py:107`),
    so on the Linux proxy `text_prompts[0].text` must be `a red fox in the snow <marker>`.
    """
    marker: Final = prompt("sdxl")
    with image_model(gateway, _artifacts, backend=SDXL_V1) as rig:
        response: Final = rig.generate(f"a red fox\nin the snow {marker}")
        assert response.status_code == 200, response.text
        assert json_body(rig.one_post())["text_prompts"] == [{"text": f"a red fox in the snow {marker}", "weight": 1}]


def test_sdxl_refuses_n_with_a_400_before_bedrock(gateway: Gateway) -> None:
    """Stability 1 supports only `size` (`amazon_stability1_transformation.py:82`).

    `n: 2` must answer 400 with `Setting \\`n\\` is not supported by bedrock` and no POST may reach the peer.
    """
    with image_model(gateway, _artifacts, backend=SDXL_V1) as rig:
        response: Final = rig.generate(prompt(), n=2)
        assert response.status_code == 400, response.text
        assert "Setting `n` is not supported by bedrock" in error_text(response), response.text
        assert rig.posts() == ()


# ------------------------------------------------------------------ Errors


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(400, "ValidationException"), (429, "ThrottlingException"), (403, "AccessDeniedException")],
)
def test_a_bedrock_image_error_keeps_its_status(gateway: Gateway, status: int, error_type: str) -> None:
    """Titan answers `<status>` with an AWS error body of type `<error_type>` and message `dec1 image failure`.

    The image handler raises `BedrockError(status_code=<status>, message=<body text>)` (`image_handler.py:172`),
    so the caller must get the same status with `dec1 image failure` in the message. UNVERIFIED for 403:
    the round 1 index notes `_map_bedrock_exception` has no explicit 403 branch.
    """

    def failing(_request: Request) -> Reply:
        return aws_error(status, error_type, "dec1 image failure")

    with image_model(gateway, failing, backend=TITAN_IMAGE_V2) as rig:
        response: Final = rig.generate(prompt(), num_retries=0)
        assert response.status_code == status, response.text
        assert "dec1 image failure" in error_text(response), response.text
