# Import types from the Google GenAI SDK
from typing import TYPE_CHECKING, Any, Optional, TypeAlias, TypedDict

# During static type-checking we can rely on the real google-genai types.
from google.genai import types as _genai_types  # type: ignore
from pydantic import BaseModel

from litellm.types.llms.openai import BaseLiteLLMOpenAIResponseObject

ContentListUnion = _genai_types.ContentListUnion
ContentListUnionDict = _genai_types.ContentListUnionDict
GenerateContentConfigOrDict = _genai_types.GenerateContentConfigOrDict
GoogleGenAIGenerateContentResponse = _genai_types.GenerateContentResponse

GenerateContentContentListUnionDict = _genai_types.ContentListUnionDict
GenerateContentConfigDict = _genai_types.GenerateContentConfigDict
GenerateContentRequestParametersDict = _genai_types._GenerateContentParametersDict

class GenerateContentRequestDict(GenerateContentRequestParametersDict):  # type: ignore[misc]
    generationConfig: Optional[Any]


class GenerateContentResponse(GoogleGenAIGenerateContentResponse, BaseLiteLLMOpenAIResponseObject): # type: ignore[misc]
    _hidden_params: dict = {}
    pass