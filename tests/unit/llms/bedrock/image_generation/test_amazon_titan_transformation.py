import pytest

from litellm.llms.bedrock.image_generation.amazon_titan_transformation import AmazonTitanImageGenerationConfig


@pytest.mark.parametrize(
    ("non_default_params", "expected"),
    [
        (
            {"size": "1024x512", "n": 2, "quality": "hd"},
            {"imageGenerationConfig": {"width": 1024, "height": 512, "numberOfImages": 2, "quality": "premium"}},
        ),
        ({"quality": "low"}, {"imageGenerationConfig": {"quality": "standard"}}),
        ({"quality": "auto", "size": None, "n": None}, {}),
    ],
)
def test_map_openai_params_builds_the_image_generation_config(
    non_default_params: dict[str, object], expected: dict[str, object]
):
    optional_params = AmazonTitanImageGenerationConfig.map_openai_params(
        non_default_params=non_default_params, optional_params={}
    )

    assert optional_params == expected


@pytest.mark.parametrize(
    ("optional_params", "expected"),
    [
        (
            {"imageGenerationConfig": {"width": 512, "numberOfImages": 2}, "negativeText": "blurry"},
            {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {"text": "a cat", "negativeText": "blurry"},
                "imageGenerationConfig": {"width": 512, "numberOfImages": 2},
            },
        ),
        (
            {"taskType": "COLOR_GUIDED_GENERATION", "negativeText": ""},
            {
                "taskType": "COLOR_GUIDED_GENERATION",
                "textToImageParams": {"text": "a cat"},
                "imageGenerationConfig": {},
            },
        ),
    ],
)
def test_transform_request_body_builds_the_titan_request(
    optional_params: dict[str, object], expected: dict[str, object]
):
    request_body = AmazonTitanImageGenerationConfig.transform_request_body(
        text="a cat", optional_params=optional_params
    )

    assert request_body == expected
    assert list(request_body["textToImageParams"]) == list(expected["textToImageParams"])
    assert optional_params == {}
