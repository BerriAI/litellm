# Import types from the Google GenAI SDK
from typing import TYPE_CHECKING, Any, List, Optional, TypeAlias

# During static type-checking we can rely on the real google-genai types.
from google.genai import types as _genai_types  # type: ignore
from pydantic import BaseModel
from typing_extensions import TypedDict

from litellm.types.llms.openai import BaseLiteLLMOpenAIResponseObject
from litellm.types.llms.vertex_ai import PromptTokensDetails

ContentListUnion = _genai_types.ContentListUnion
ContentListUnionDict = _genai_types.ContentListUnionDict
GenerateContentConfigOrDict = _genai_types.GenerateContentConfigOrDict
GoogleGenAIGenerateContentResponse = _genai_types.GenerateContentResponse

GenerateContentContentListUnionDict = _genai_types.ContentListUnionDict
GenerateContentConfigDict = _genai_types.GenerateContentConfigDict
GenerateContentRequestParametersDict = _genai_types._GenerateContentParametersDict
ToolConfigDict = _genai_types.ToolConfigDict

class GenerateContentRequestDict(GenerateContentRequestParametersDict):  # type: ignore[misc]
    generationConfig: Optional[Any]
    tools: Optional[ToolConfigDict] # type: ignore[assignment]


class GenerateContentResponse(GoogleGenAIGenerateContentResponse, BaseLiteLLMOpenAIResponseObject): # type: ignore[misc]
    _hidden_params: dict = {}
    pass


class TokenCountDetailsResponse(TypedDict):
    """
    Response structure for token count details with modality breakdown.
    
    Example:
        {'totalTokens': 12, 'promptTokensDetails': [{'modality': 'TEXT', 'tokenCount': 12}]}
    """
    totalTokens: int
    promptTokensDetails: List[PromptTokensDetails]
