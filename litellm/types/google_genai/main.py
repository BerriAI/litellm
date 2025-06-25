# Import types from the Google GenAI SDK
from typing import TYPE_CHECKING, Any, Optional, TypeAlias, TypedDict

from pydantic import BaseModel


# Define fallback base class
class _FallbackRequestDict(TypedDict):
    """Fallback when google.genai types are unavailable."""


if TYPE_CHECKING:
    # During static type-checking we can rely on the real google-genai types.
    from google.genai import types as _genai_types  # type: ignore

    ContentListUnion = _genai_types.ContentListUnion
    ContentListUnionDict = _genai_types.ContentListUnionDict
    GenerateContentConfigOrDict = _genai_types.GenerateContentConfigOrDict
    GenerateContentResponse = _genai_types.GenerateContentResponse

    GenerateContentContentListUnionDict = _genai_types.ContentListUnionDict
    GenerateContentConfigDict = _genai_types.GenerateContentConfigDict
    GenerateContentRequestParametersDict = _genai_types._GenerateContentParametersDict
else:
    # At runtime we fall back to `Any` to avoid a hard dependency on the SDK.
    ContentListUnion: TypeAlias = Any
    ContentListUnionDict: TypeAlias = Any
    GenerateContentConfigOrDict: TypeAlias = Any
    GenerateContentResponse: TypeAlias = Any

    GenerateContentContentListUnionDict: TypeAlias = Any
    GenerateContentConfigDict: TypeAlias = Any
    GenerateContentRequestParametersDict: TypeAlias = _FallbackRequestDict


class GenerateContentRequestDict(GenerateContentRequestParametersDict):  # type: ignore[misc]
    generationConfig: Optional[Any]