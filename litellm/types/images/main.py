from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from typing_extensions import ReadOnly, TypedDict

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
    imageConfig: dict[str, Any] | None


class ImageEditRequestParams(ImageEditOptionalRequestParams, total=False):
    """
    TypedDict for request parameters supported by OpenAI's image edit API.

    Params here: https://platform.openai.com/docs/api-reference/images/createEdit
    """

    image: FileTypes
    prompt: str
    model: str | None


class ImageVariationOptionalRequestParams(TypedDict, total=False):
    """
    TypedDict for Optional parameters supported by OpenAI's image variation API.

    Params here: https://platform.openai.com/docs/api-reference/images/createVariation
    """

    model: ReadOnly[str | None]
    n: ReadOnly[int | None]
    response_format: ReadOnly[Literal["url", "b64_json"] | None]
    size: ReadOnly[str | None]
    user: ReadOnly[str | None]


class ImageVariationRequestParams(ImageVariationOptionalRequestParams, total=False):
    """
    TypedDict for request parameters supported by OpenAI's image variation API.

    Params here: https://platform.openai.com/docs/api-reference/images/createVariation
    """

    image: ReadOnly[FileTypes]


class ImageVariationRequest(BaseModel):
    """
    Pydantic model for image variation request body schema.
    """

    model_config = ConfigDict(frozen=True)

    image: str
    model: str | None = None
    n: int | None = 1
    response_format: Literal["url", "b64_json"] | None = "url"
    size: str | None = "1024x1024"
    user: str | None = None
