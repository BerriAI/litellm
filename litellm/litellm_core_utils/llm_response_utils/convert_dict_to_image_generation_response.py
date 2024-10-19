from typing import Dict, Optional

from litellm.types.utils import ImageResponse


def convert_dict_to_image_generation_response(
    model_response_object: Optional[ImageResponse],
    response_object: Optional[Dict],
    hidden_params: Optional[Dict],
) -> ImageResponse:
    if response_object is None:
        raise Exception("Error in response object format")

    if model_response_object is None:
        model_response_object = ImageResponse()

    if "created" in response_object:
        model_response_object.created = response_object["created"]

    if "data" in response_object:
        model_response_object.data = response_object["data"]

    if hidden_params is not None:
        model_response_object._hidden_params = hidden_params

    return model_response_object
