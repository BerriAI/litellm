from typing import Any, Dict, Iterable, List, Literal, Optional, Union

from typing_extensions import Required, TypedDict


class ImageEmbeddingInput(TypedDict, total=False):
    image: Required[str]
    text: str


EncodingFormat = Literal["base64", "binary", "float", "int8", "ubinary", "uint8"]


class ImageEmbeddingRequest(TypedDict, total=False):
    input: Required[List[ImageEmbeddingInput]]
    dimensions: int
    encoding_format: EncodingFormat
