from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from litellm.types.utils import FileTypes


class ImageEditOptionalRequestParams(TypedDict, total=False):
    """
    TypedDict for Optional parameters supported by OpenAI's image edit API.

    Params here: https://platform.openai.com/docs/api-reference/images/createEdit
    """

    background: Literal["transparent", "opaque", "auto"] | None
    input_fidelity: Literal["high", "low"] | None
    mask: str | None
    n: int | None
    quality: Literal["high", "medium", "low", "standard", "auto"] | None
    response_format: Literal["url", "b64_json"] | None
    size: str | None
    user: str | None
    imageConfig: dict[str, object] | None


class ImageEditRequestParams(ImageEditOptionalRequestParams, total=False):
    """
    TypedDict for request parameters supported by OpenAI's image edit API.

    Params here: https://platform.openai.com/docs/api-reference/images/createEdit
    """

    image: FileTypes
    prompt: str
    model: str | None


IMAGE_CONFIG_FIELDS: Final = ("imageConfig", "image_config")


@dataclass(frozen=True, slots=True)
class UnsupportedImageParam:
    param: str
    reason: str


class ImageSVGAuthorDeployment(BaseModel):
    model_config = ConfigDict(frozen=True)

    deployment_id: str
    provider_model: str
