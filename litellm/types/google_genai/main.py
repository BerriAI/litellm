# Import types from the Google GenAI SDK
from typing import TYPE_CHECKING, Any, Optional, TypeAlias, TypedDict

# During static type-checking we can rely on the real google-genai types.
from google.genai import types as _genai_types  # type: ignore
from pydantic import BaseModel

ContentListUnion = _genai_types.ContentListUnion
ContentListUnionDict = _genai_types.ContentListUnionDict
GenerateContentConfigOrDict = _genai_types.GenerateContentConfigOrDict
GenerateContentResponse = _genai_types.GenerateContentResponse

GenerateContentContentListUnionDict = _genai_types.ContentListUnionDict
GenerateContentConfigDict = _genai_types.GenerateContentConfigDict
GenerateContentRequestParametersDict = _genai_types._GenerateContentParametersDict

class GenerateContentRequestDict(GenerateContentRequestParametersDict):  # type: ignore[misc]
    generationConfig: Optional[Any]