# Import types from the Google GenAI SDK
from typing import Any, Optional, TypedDict

from pydantic import BaseModel


# Define fallback base class
class _FallbackRequestDict(TypedDict):
    pass

try:
    from google.genai import types

    ContentListUnion = types.ContentListUnion
    ContentListUnionDict = types.ContentListUnionDict
    GenerateContentConfigOrDict = types.GenerateContentConfigOrDict
    GenerateContentResponse = types.GenerateContentResponse

    ########################################################
    # Request Type
    ########################################################
    GenerateContentContentListUnionDict = types.ContentListUnionDict
    GenerateContentConfigDict = types.GenerateContentConfigDict
    GenerateContentRequestParametersDict = types._GenerateContentParametersDict

    # When google-genai is available, inherit from their base clas

except ImportError:
    # If google-genai is not installed, we need to define the types manually
    ContentListUnion = Any
    ContentListUnionDict = Any
    GenerateContentConfigOrDict = Any
    GenerateContentResponse = Any
    GenerateContentContentListUnionDict = Any
    GenerateContentConfigDict = Any
    GenerateContentRequestParametersDict = _FallbackRequestDict

class GenerateContentRequestDict(GenerateContentRequestParametersDict):  # type: ignore
    generationConfig: Optional[Any]