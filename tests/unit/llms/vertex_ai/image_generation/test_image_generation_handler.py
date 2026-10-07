import pytest
from pydantic import ValidationError

from litellm.llms.vertex_ai.image_generation.image_generation_handler import VertexImageGeneration
from litellm.types.utils import ImageResponse


def _process(json_response: dict[str, object]) -> ImageResponse:
    return VertexImageGeneration().process_image_generation_response(
        json_response, ImageResponse(), "imagegeneration@006"
    )


@pytest.mark.parametrize(
    ("predictions", "expected"),
    [
        ([{"bytesBase64Encoded": "QUJD", "mimeType": "image/png"}, {"bytesBase64Encoded": "REVG"}], ["QUJD", "REVG"]),
        ([{"bytesBase64Encoded": None}], [None]),
        ([], []),
    ],
)
def test_process_image_generation_response_maps_each_prediction(
    predictions: list[dict[str, object]], expected: list[str | None]
):
    response = _process({"predictions": predictions, "deployedModelId": "1"})

    assert [image.b64_json for image in response.data] == expected


@pytest.mark.parametrize(
    "predictions",
    [None, 7, ["QUJD"], [{"bytesBase64Encoded": "QUJD"}, None], [{"bytesBase64Encoded": 7}]],
)
def test_process_image_generation_response_rejects_malformed_predictions(predictions: object):
    with pytest.raises(ValidationError) as exc_info:
        _process({"predictions": predictions})

    assert "QUJD" not in str(exc_info.value)


def test_process_image_generation_response_requires_the_encoded_bytes_key():
    with pytest.raises(KeyError):
        _process({"predictions": [{"mimeType": "image/png"}]})
