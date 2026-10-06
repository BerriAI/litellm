import os

import pytest

from litellm.llms.bedrock.image_generation.amazon_stability1_transformation import AmazonStabilityConfig
from litellm.types.utils import ImageResponse


def test_transform_request_body_sends_a_single_line_prompt_with_caller_params_and_no_user() -> None:
    optional_params = {"width": 512, "height": 768, "user": "end-user-1"}

    request_body = AmazonStabilityConfig().transform_request_body(
        text=f"a red fox{os.linesep}in the snow", optional_params=optional_params
    )

    assert request_body == {
        "text_prompts": [{"text": "a red fox in the snow", "weight": 1}],
        "width": 512,
        "height": 768,
    }
    assert optional_params == {"width": 512, "height": 768, "user": "end-user-1"}


@pytest.mark.parametrize(
    ("non_default_params", "expected"),
    [
        ({"size": "512x768"}, {"steps": 30, "width": 512, "height": 768}),
        ({"n": 2}, {"steps": 30}),
    ],
)
def test_map_openai_params_turns_size_into_width_and_height(
    non_default_params: dict[str, object], expected: dict[str, object]
) -> None:
    optional_params = AmazonStabilityConfig().map_openai_params(
        non_default_params=non_default_params, optional_params={"steps": 30}
    )

    assert optional_params == expected


def test_transform_response_dict_returns_one_image_per_artifact_in_order() -> None:
    model_response = ImageResponse()

    image_response = AmazonStabilityConfig().transform_response_dict_to_openai_response(
        model_response=model_response,
        response_dict={"artifacts": [{"base64": "Zmlyc3Q="}, {"base64": "c2Vjb25k"}]},
    )

    assert image_response is model_response
    assert [image.b64_json for image in image_response.data] == ["Zmlyc3Q=", "c2Vjb25k"]
