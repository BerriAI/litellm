"""CI-covered tests for the Nova Canvas conditioned-editing additions (issue #39552).

Self-contained mirror of the branch-added tests in
tests/unit/llms/bedrock/image_edit/test_amazon_nova_canvas_image_edit.py: the
conditioning guard, conditionImage acceptance/precedence, controlStrength
coercion/range, mask/maskPrompt rejection, prompt-required, style forwarding,
supported params, and the litellm-level proxy contracts (400-class validation
errors, style passthrough through litellm.aimage_edit).
"""

import asyncio
import base64
import io
import json
from datetime import datetime
from typing import Final
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.image_edit.amazon_nova_canvas_image_edit_transformation import (
    BedrockAmazonNovaCanvasImageEditConfig,
)

TEST_MODEL = "amazon.nova-canvas-v1:0"


@pytest.fixture(autouse=True)
def ensure_nova_canvas_image_edit_model_cost_flags(monkeypatch):
    """Routing uses ``supports_nova_canvas_image_edit`` on ``litellm.model_cost``.

    Full ``model_prices_and_context_window.json`` includes these flags, but CI or
    alternate cost maps may omit them; merge minimal entries so tests match production.
    """
    from litellm.utils import _invalidate_model_cost_lowercase_map

    for key in (
        "amazon.nova-canvas-v1:0",
        "us.amazon.nova-canvas-v1:0",
    ):
        entry = litellm.model_cost.get(key) or {}
        if entry.get("supports_nova_canvas_image_edit") is True:
            continue
        monkeypatch.setitem(
            litellm.model_cost,
            key,
            {
                **entry,
                "litellm_provider": entry.get("litellm_provider", "bedrock"),
                "mode": entry.get("mode", "image_generation"),
                "supports_nova_canvas_image_edit": True,
            },
        )
        _invalidate_model_cost_lowercase_map()

    yield

    _invalidate_model_cost_lowercase_map()


#################################################
# conditioning guard: controlMode/controlStrength/style need TEXT_IMAGE
#################################################


def test_transform_request_conditioning_fields_without_text_image_task_type_raises():
    """controlMode/controlStrength/style with any non-TEXT_IMAGE resolved taskType must
    fail fast instead of being silently dropped by mask/variation routing."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"cond")
    with pytest.raises(BedrockError) as excinfo:
        config.transform_image_edit_request(
            model=TEST_MODEL,
            prompt="restyle",
            image=img,
            image_edit_optional_request_params={
                "controlMode": "SEGMENTATION",
            },
            litellm_params={},
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "taskType=TEXT_IMAGE" in str(excinfo.value.message)
    assert "controlMode/controlStrength/style" in str(excinfo.value.message)


def test_transform_request_control_strength_without_text_image_task_type_raises():
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"cond")
    with pytest.raises(BedrockError) as excinfo:
        config.transform_image_edit_request(
            model=TEST_MODEL,
            prompt="restyle",
            image=img,
            image_edit_optional_request_params={
                "controlStrength": 0.4,
            },
            litellm_params={},
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "taskType=TEXT_IMAGE" in str(excinfo.value.message)


def test_transform_request_style_without_text_image_task_type_raises():
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"cond")
    with pytest.raises(BedrockError) as excinfo:
        config.transform_image_edit_request(
            model=TEST_MODEL,
            prompt="restyle",
            image=img,
            image_edit_optional_request_params={
                "style": "DESIGN_SKETCH",
            },
            litellm_params={},
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "taskType=TEXT_IMAGE" in str(excinfo.value.message)


def test_transform_request_conditioning_fields_with_explicit_text_image_works():
    """The guard must not fire when taskType is explicitly TEXT_IMAGE."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"cond")
    body, _ = config.transform_image_edit_request(
        model=TEST_MODEL,
        prompt="restyle",
        image=img,
        image_edit_optional_request_params={
            "taskType": "TEXT_IMAGE",
            "controlMode": "CANNY_EDGE",
        },
        litellm_params={},
        headers={},
    )
    assert body["taskType"] == "TEXT_IMAGE"
    assert body["textToImageParams"]["controlMode"] == "CANNY_EDGE"


#################################################
# conditionImage acceptance and precedence
#################################################


def test_transform_request_condition_image_without_multipart_image():
    """conditionImage is an alternative TEXT_IMAGE condition source for JSON-only callers."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    condition_b64: Final = base64.b64encode(b"cond-bytes").decode("utf-8")
    body, _ = config.transform_image_edit_request(
        model=TEST_MODEL,
        prompt="same layout",
        image=None,
        image_edit_optional_request_params={
            "taskType": "TEXT_IMAGE",
            "conditionImage": condition_b64,
            "controlMode": "SEGMENTATION",
        },
        litellm_params={},
        headers={},
    )
    assert body["taskType"] == "TEXT_IMAGE"
    t2i = body["textToImageParams"]
    assert t2i["conditionImage"] == condition_b64
    assert t2i["controlMode"] == "SEGMENTATION"


def test_transform_request_condition_image_bytes_accepted():
    """Raw bytes conditionImage is base64-encoded on the way into the body."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    body, _ = config.transform_image_edit_request(
        model=TEST_MODEL,
        prompt="same layout",
        image=None,
        image_edit_optional_request_params={
            "taskType": "TEXT_IMAGE",
            "conditionImage": b"raw-cond-bytes",
        },
        litellm_params={},
        headers={},
    )
    assert body["textToImageParams"]["conditionImage"] == base64.b64encode(b"raw-cond-bytes").decode("utf-8")


def test_transform_request_multipart_image_wins_over_condition_image():
    """Pinned precedence: when both are supplied the multipart `image` field wins."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    condition_b64: Final = base64.b64encode(b"from-condition-param").decode("utf-8")
    body, _ = config.transform_image_edit_request(
        model=TEST_MODEL,
        prompt="same layout",
        image=io.BytesIO(b"from-multipart"),
        image_edit_optional_request_params={
            "taskType": "TEXT_IMAGE",
            "conditionImage": condition_b64,
        },
        litellm_params={},
        headers={},
    )
    assert body["textToImageParams"]["conditionImage"] == base64.b64encode(b"from-multipart").decode("utf-8")


def test_transform_request_condition_image_with_other_task_type_raises():
    """conditionImage only conditions TEXT_IMAGE; other task types need the image input."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    with pytest.raises(BedrockError) as excinfo:
        config.transform_image_edit_request(
            model=TEST_MODEL,
            prompt="restyle",
            image=None,
            image_edit_optional_request_params={
                "taskType": "IMAGE_VARIATION",
                "conditionImage": base64.b64encode(b"cond").decode("utf-8"),
            },
            litellm_params={},
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "conditionImage is only supported" in str(excinfo.value.message)


#################################################
# controlStrength coercion and range
#################################################


def test_transform_request_text_image_control_strength_string_coerced():
    """controlStrength arriving as a string (multipart form data) is coerced to a float."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"cond")
    body, _ = config.transform_image_edit_request(
        model=TEST_MODEL,
        prompt="restyle",
        image=img,
        image_edit_optional_request_params={
            "taskType": "TEXT_IMAGE",
            "controlStrength": "0.7",
        },
        litellm_params={},
        headers={},
    )
    assert body["textToImageParams"]["controlStrength"] == 0.7


@pytest.mark.parametrize("control_strength", [0.0, 1.0])
def test_transform_request_text_image_control_strength_bounds_pass(control_strength):
    """Boundary controlStrength values 0.0 and 1.0 are accepted."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"cond")
    body, _ = config.transform_image_edit_request(
        model=TEST_MODEL,
        prompt="restyle",
        image=img,
        image_edit_optional_request_params={
            "taskType": "TEXT_IMAGE",
            "controlStrength": control_strength,
        },
        litellm_params={},
        headers={},
    )
    assert body["textToImageParams"]["controlStrength"] == control_strength


def test_transform_request_text_image_control_strength_out_of_range_raises():
    """controlStrength outside 0.0-1.0 fails fast."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"cond")
    with pytest.raises(BedrockError) as excinfo:
        config.transform_image_edit_request(
            model=TEST_MODEL,
            prompt="restyle",
            image=img,
            image_edit_optional_request_params={
                "taskType": "TEXT_IMAGE",
                "controlStrength": 1.5,
            },
            litellm_params={},
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "controlStrength must be between 0.0 and 1.0" in str(excinfo.value.message)


def test_transform_request_text_image_control_strength_non_numeric_string_raises():
    """A non-numeric controlStrength string must fail fast, not TypeError."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"cond")
    with pytest.raises(BedrockError) as excinfo:
        config.transform_image_edit_request(
            model=TEST_MODEL,
            prompt="restyle",
            image=img,
            image_edit_optional_request_params={
                "taskType": "TEXT_IMAGE",
                "controlStrength": "abc",
            },
            litellm_params={},
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "controlStrength must be a number" in str(excinfo.value.message)


#################################################
# TEXT_IMAGE rejects mask inputs and requires a prompt
#################################################


def test_transform_request_text_image_with_mask_raises():
    """TEXT_IMAGE has no mask field; a provided mask must fail fast."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"cond")
    mask = io.BytesIO(b"mask-bytes")
    with pytest.raises(BedrockError) as excinfo:
        config.transform_image_edit_request(
            model=TEST_MODEL,
            prompt="restyle",
            image=img,
            image_edit_optional_request_params={
                "taskType": "TEXT_IMAGE",
                "mask": mask,
            },
            litellm_params={},
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "does not support a mask" in str(excinfo.value.message)
    assert "INPAINTING or OUTPAINTING" in str(excinfo.value.message)


def test_transform_request_text_image_with_mask_prompt_raises():
    """TEXT_IMAGE has no maskPrompt field either; fail fast like the binary mask
    instead of silently dropping it."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"cond")
    with pytest.raises(BedrockError) as excinfo:
        config.transform_image_edit_request(
            model=TEST_MODEL,
            prompt="restyle",
            image=img,
            image_edit_optional_request_params={
                "taskType": "TEXT_IMAGE",
                "maskPrompt": "the sky region",
            },
            litellm_params={},
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "does not support a mask" in str(excinfo.value.message)
    assert "INPAINTING or OUTPAINTING" in str(excinfo.value.message)


def test_transform_request_text_image_empty_prompt_raises():
    """An empty TEXT_IMAGE prompt must fail fast, not silently send a blank."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"cond")
    with pytest.raises(BedrockError) as excinfo:
        config.transform_image_edit_request(
            model=TEST_MODEL,
            prompt="",
            image=img,
            image_edit_optional_request_params={"taskType": "TEXT_IMAGE"},
            litellm_params={},
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "TEXT_IMAGE requires a text prompt" in str(excinfo.value.message)


def test_transform_request_text_image_none_prompt_raises():
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"cond")
    with pytest.raises(BedrockError) as excinfo:
        config.transform_image_edit_request(
            model=TEST_MODEL,
            prompt=None,
            image=img,
            image_edit_optional_request_params={"taskType": "TEXT_IMAGE"},
            litellm_params={},
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "TEXT_IMAGE requires a text prompt" in str(excinfo.value.message)


#################################################
# style forwarding and supported params
#################################################


def test_transform_request_text_image_forwards_style():
    """style is forwarded into textToImageParams for conditioned editing."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"cond")
    body, _ = config.transform_image_edit_request(
        model=TEST_MODEL,
        prompt="a city street in the same layout",
        image=img,
        image_edit_optional_request_params={
            "taskType": "TEXT_IMAGE",
            "controlMode": "SEGMENTATION",
            "style": "DESIGN_SKETCH",
        },
        litellm_params={},
        headers={},
    )
    assert body["taskType"] == "TEXT_IMAGE"
    assert body["textToImageParams"]["style"] == "DESIGN_SKETCH"


def test_transform_request_text_image_omits_style_when_absent():
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"cond")
    body, _ = config.transform_image_edit_request(
        model=TEST_MODEL,
        prompt="restyle",
        image=img,
        image_edit_optional_request_params={"taskType": "TEXT_IMAGE"},
        litellm_params={},
        headers={},
    )
    assert "style" not in body["textToImageParams"]


def test_get_supported_openai_params_includes_conditioning_fields():
    """controlMode/controlStrength are advertised for TEXT_IMAGE conditioned editing."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    supported = config.get_supported_openai_params(TEST_MODEL)
    assert "controlMode" in supported
    assert "controlStrength" in supported


def test_get_supported_openai_params_includes_style():
    """style is advertised for TEXT_IMAGE conditioned editing."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    supported = config.get_supported_openai_params(TEST_MODEL)
    assert "style" in supported


def test_get_supported_openai_params_includes_condition_image():
    config = BedrockAmazonNovaCanvasImageEditConfig()
    supported = config.get_supported_openai_params(TEST_MODEL)
    assert "conditionImage" in supported


#################################################
# litellm-level proxy contracts
#################################################


async def test_aimage_edit_mask_with_text_image_maps_to_bad_request(monkeypatch):
    """Through the litellm image-edit layer, the TEXT_IMAGE mask guard must surface as
    litellm.BadRequestError (400-class), never APIConnectionError/500."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bearer-token-12345")
    with pytest.raises(litellm.BadRequestError) as excinfo:
        await litellm.aimage_edit(
            model=f"bedrock/{TEST_MODEL}",
            prompt="restyle",
            image=io.BytesIO(b"img-bytes"),
            taskType="TEXT_IMAGE",
            mask=io.BytesIO(b"mask-bytes"),
        )
    assert excinfo.value.status_code == 400
    assert "does not support a mask" in str(excinfo.value)


async def test_aimage_edit_forwards_style_to_nova_canvas_transform(monkeypatch):
    """style passed to litellm.aimage_edit must survive the images/main.py param
    filtering and reach the Nova Canvas transform (regression: style used to be
    blocklisted and silently dropped on the bedrock path)."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bearer-token-12345")
    posted: dict[str, object] = {}

    class _FakeAsyncClient:
        async def post(self, url, headers, data):
            posted["url"] = url
            posted["body"] = json.loads(data)
            return httpx.Response(200, json={"images": ["aGk="]}, request=httpx.Request("POST", url))

    import litellm.llms.bedrock.image_edit.handler as bedrock_image_edit_handler

    with patch.object(
        bedrock_image_edit_handler,
        "get_async_httpx_client",
        lambda **kwargs: _FakeAsyncClient(),
    ):
        response = await litellm.aimage_edit(
            model=f"bedrock/{TEST_MODEL}",
            prompt="same layout",
            image=io.BytesIO(b"img-bytes"),
            taskType="TEXT_IMAGE",
            style="DESIGN_SKETCH",
        )
    body = posted["body"]
    assert isinstance(body, dict)
    assert body["textToImageParams"]["style"] == "DESIGN_SKETCH"
    assert response.data[0].b64_json == "aGk="


#################################################
# logging headers redaction (pre_call additional_args)
#################################################


def test_redact_bedrock_headers_for_logging_masks_signed_headers():
    """SigV4 signature material must be replaced with [REDACTED]; safe headers survive."""
    from litellm.llms.bedrock.common_utils import redact_bedrock_headers_for_logging

    signed: Final[dict[str, str]] = {
        "Content-Type": "application/json",
        "Host": "bedrock-runtime.us-east-1.amazonaws.com",
        "Authorization": (
            "AWS4-HMAC-SHA256 Credential=AKIA-test/20260115/us-east-1/bedrock/aws4_request, "
            "SignedHeaders=host;x-amz-date, Signature=deadbeefsecret"
        ),
        "X-Amz-Date": "20260115T103000Z",
        "X-Amz-Security-Token": "session-token-secret",
        "X-Amzn-RequestId": "request-id-not-secret",
    }
    redacted = redact_bedrock_headers_for_logging(signed)
    assert redacted["Content-Type"] == "application/json"
    assert redacted["Host"] == "bedrock-runtime.us-east-1.amazonaws.com"
    assert redacted["X-Amzn-RequestId"] == "request-id-not-secret"
    assert redacted["Authorization"] == "[REDACTED]"
    assert redacted["X-Amz-Date"] == "[REDACTED]"
    assert redacted["X-Amz-Security-Token"] == "[REDACTED]"
    # Every key stays present (log consumers see the full header shape).
    assert set(redacted.keys()) == set(signed.keys())
    # The input mapping is untouched: redaction never mutates the sent headers.
    assert signed["Authorization"].startswith("AWS4-HMAC-SHA256")
    assert signed["X-Amz-Security-Token"] == "session-token-secret"


def test_prepare_request_logging_headers_redacted(monkeypatch):
    """pre_call additional_args must carry the redacted copy; the sent request keeps
    the real bearer Authorization header."""
    from litellm.llms.bedrock.image_edit.handler import BedrockImageEdit
    from litellm.litellm_core_utils.litellm_logging import Logging

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bearer-token-12345")

    captured: dict = {}

    def _capture(model_call_details):
        captured.update(model_call_details)

    logging_obj = Logging(
        model=f"bedrock/{TEST_MODEL}",
        messages=[],
        stream=False,
        call_type="aimage_edit",
        start_time=datetime.now(),
        litellm_call_id="test-call-id",
        function_id="test-function",
        kwargs={"logger_fn": _capture},
    )
    logging_obj.update_environment_variables(
        litellm_params={"logger_fn": _capture},
        optional_params={},
    )

    request = BedrockImageEdit()._prepare_request(
        model=TEST_MODEL,
        image=[io.BytesIO(b"fake-png")],
        prompt="make it warmer",
        optional_params={"aws_region_name": "us-west-2", "aws_profile_name": "litellm-no-such-aws-profile"},
        api_base=None,
        extra_headers=None,
        logging_obj=logging_obj,
        api_key=None,
    )
    logged_headers: Final = captured["additional_args"]["headers"]
    assert logged_headers["Authorization"] == "[REDACTED]"
    assert logged_headers["Content-Type"] == "application/json"
    assert "env-bearer-token-12345" not in str(captured)
    # The sent request still carries the real bearer Authorization header.
    assert request.prepped.headers["Authorization"] == "Bearer env-bearer-token-12345"
