"""
Translation from OpenAI's `/v1/images/generations` to Bedrock Stability AI's stability.sd3-large-v1:0 API format
"""

import time
import types
from typing import List, Literal, Optional, Required, TypedDict

from ....types.utils import ImageObject, ImageResponse


class StabilitySD3Request(TypedDict, total=False):
    prompt: Required[str]
    mode: Literal["text-to-image"]
    aspect_ratio: str
    output_format: str


class StabilitySD3Response(TypedDict, total=False):
    images: Required[List[str]]
    seeds: List[int]
    finish_reasons: List[Optional[str]]


class AmazonStabilitySD3Config:
    def __init__(
        self,
        cfg_scale: Optional[int] = None,
        seed: Optional[float] = None,
        steps: Optional[List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def _transform_request(
        self, input: str, inference_params: dict
    ) -> StabilitySD3Request:
        response_obj = StabilitySD3Request(prompt=input)

        return response_obj

    def _transform_response(self, response: dict) -> ImageResponse:
        response_obj = StabilitySD3Response(**response)  # type: ignore
        image_response = ImageResponse(
            created=int(time.time()),
            data=[ImageObject(b64_json=image) for image in response_obj["images"]],
        )
        return image_response
