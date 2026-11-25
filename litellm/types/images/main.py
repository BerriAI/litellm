from dataclasses import dataclass
from typing import Final, Literal

from typing_extensions import ReadOnly, TypedDict

from litellm.types.utils import FileTypes


class ImageEditOptionalRequestParams(TypedDict, total=False):
    """
    TypedDict for Optional parameters supported by OpenAI's image edit API.

    Params here: https://platform.openai.com/docs/api-reference/images/createEdit
    """

    background: Literal["transparent", "opaque", "auto"] | None
    input_fidelity: Literal["high", "low"] | None
    markup: ReadOnly[FileTypes | list[FileTypes] | None]  # mutable-ok: a tuple is one upload, a list is several
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
