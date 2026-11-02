from typing import Literal

from typing_extensions import NotRequired, ReadOnly, Required, TypedDict


class ImageEmbeddingInput(TypedDict, total=False):
    image: Required[str]
    text: str


EncodingFormat = Literal["base64", "binary", "float", "int8", "ubinary", "uint8"]


class ImageEmbeddingRequest(TypedDict, total=False):
    input: Required[list[ImageEmbeddingInput]]
    dimensions: int
    encoding_format: EncodingFormat


class FoundryAgentReference(TypedDict):
    type: ReadOnly[Literal["agent_reference"]]
    name: ReadOnly[str]
    version: ReadOnly[NotRequired[str]]
