from typing import Any, Dict, List, Literal, Optional, Union

from typing_extensions import TypedDict

from litellm.types.utils import FileTypes


class ImageEditOptionalRequestParams(TypedDict, total=False):
    """
    TypedDict for Optional parameters supported by OpenAI's image edit API.

    Params here: https://platform.openai.com/docs/api-reference/images/createEdit
    """

    background: Optional[Literal["transparent", "opaque", "auto"]]
    mask: Optional[str]
    n: Optional[int]
    quality: Optional[Literal["high", "medium", "low", "standard", "auto"]]
    response_format: Optional[Literal["url", "b64_json"]]
    size: Optional[str]
    user: Optional[str]


class ImageEditRequestParams(ImageEditOptionalRequestParams, total=False):
    """
    TypedDict for request parameters supported by OpenAI's image edit API.

    Params here: https://platform.openai.com/docs/api-reference/images/createEdit
    """

    image: FileTypes
    prompt: str
    model: Optional[str]
