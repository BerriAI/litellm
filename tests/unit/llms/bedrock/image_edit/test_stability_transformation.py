import re
from datetime import datetime
from typing import Final

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.image_edit.stability_transformation import BedrockStabilityImageEditConfig
from litellm.types.utils import ImageResponse
from tests.unit.conftest import in_operator_error

INPAINT_MODEL: Final = "stability.stable-image-inpaint-v1:0"


def test_transform_image_edit_request_returns_the_json_body_and_no_files():
    request = BedrockStabilityImageEditConfig().transform_image_edit_request(
        model="stability.stable-image-inpaint-v1:0",
        prompt="add a red hat",
        image=b"\x89PNG-bytes",
        image_edit_optional_request_params={},
        litellm_params={},
        headers={},
    )

    assert request == ({"output_format": "png", "prompt": "add a red hat", "image": "iVBORy1ieXRlcw=="}, {})


def _edit_response(body: object) -> ImageResponse:
    return BedrockStabilityImageEditConfig().transform_image_edit_response(
        model=INPAINT_MODEL,
        raw_response=httpx.Response(200, json=body),
        logging_obj=Logging(
            model=INPAINT_MODEL,
            messages=[],
            stream=False,
            call_type="image_edit",
            start_time=datetime(2026, 1, 1),
            litellm_call_id="stability-edit-call",
            function_id="stability-edit-function",
        ),
    )


@pytest.mark.usefixtures("local_model_cost_map")
def test_transform_image_edit_response_returns_every_non_empty_image_in_order():
    response = _edit_response({"images": ["aW1hZ2Ux", "", "aW1hZ2Uy"], "finish_reasons": [None], "seeds": [7]})

    assert [(image.b64_json, image.url) for image in response.data] == [("aW1hZ2Ux", None), ("aW1hZ2Uy", None)]


@pytest.mark.parametrize(
    ("body", "status_code", "message"),
    [
        ({"errors": ["prompt was flagged"]}, 200, "Bedrock Stability error: ['prompt was flagged']"),
        (
            {"images": [], "finish_reasons": ["Filter reason: prompt"]},
            400,
            "Bedrock Stability error: Filter reason: prompt",
        ),
    ],
)
def test_transform_image_edit_response_raises_the_error_bedrock_reports(body: object, status_code: int, message: str):
    with pytest.raises(BedrockError) as error:
        _edit_response(body)

    assert (error.value.status_code, error.value.message) == (status_code, message)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (5, in_operator_error(5)),
        ({"finish_reasons": 5}, "'int' object is not subscriptable"),
    ],
)
def test_transform_image_edit_response_rejects_a_scalar_where_bedrock_sends_a_container(body: object, message: str):
    with pytest.raises(TypeError, match=re.escape(message)):
        _edit_response(body)
