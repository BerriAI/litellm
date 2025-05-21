from typing import Any, Dict, List, Literal, Optional, TypedDict, Union


class ImageEditOptionalRequestParams(TypedDict, total=False):
    """
    TypedDict for Optional parameters supported by OpenAI's image edit API.

    Params here: https://platform.openai.com/docs/api-reference/images/createEdit
    """

    background: Optional[Literal["transparent", "opaque", "auto"]]
    mask: Optional[str]
    model: Optional[str]
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

    image: Union[str, List[str]]
    prompt: str
