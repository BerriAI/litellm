from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.bedrock.image_edit.stability_transformation import BedrockStabilityImageEditConfig


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


@pytest.mark.parametrize(
    ("body", "expected_images"),
    [
        ({"images": ["aW1hZ2U="], "finish_reasons": [None], "seeds": [42]}, ["aW1hZ2U="]),
        ({"images": ["aW1hZ2U="], "finish_reasons": None}, ["aW1hZ2U="]),
        ({"images": None, "finish_reasons": None}, []),
        ({}, []),
    ],
)
def test_transform_image_edit_response_returns_each_image_and_tolerates_null_lists(body, expected_images):
    response = BedrockStabilityImageEditConfig().transform_image_edit_response(
        model="stability.stable-image-inpaint-v1:0",
        raw_response=httpx.Response(200, json=body),
        logging_obj=MagicMock(),
    )

    assert [image.b64_json for image in response.data or []] == expected_images


def test_transform_image_edit_response_raises_the_first_finish_reason():
    with pytest.raises(Exception, match="Bedrock Stability error: Filter reason: prompt"):
        BedrockStabilityImageEditConfig().transform_image_edit_response(
            model="stability.stable-image-inpaint-v1:0",
            raw_response=httpx.Response(200, json={"images": [None], "finish_reasons": ["Filter reason: prompt"]}),
            logging_obj=MagicMock(),
        )
