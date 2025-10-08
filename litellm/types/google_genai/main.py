# Import types from the Google GenAI SDK
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeAlias

from pydantic import BaseModel
from typing_extensions import TypedDict

from litellm.types.llms.openai import BaseLiteLLMOpenAIResponseObject

# During static type-checking we can rely on the real google-genai types.
if TYPE_CHECKING:
    from google.genai import types as _genai_types  # type: ignore

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
else:
    # Fallback types when google.genai is not available
    ContentListUnion = Any
    ContentListUnionDict = Dict[str, Any]
    GenerateContentConfigOrDict = Dict[str, Any]
    GoogleGenAIGenerateContentResponse = Dict[str, Any]
    GenerateContentContentListUnionDict = Dict[str, Any]

    # Create a proper fallback class that can be instantiated
    class GenerateContentConfigDict(dict):  # type: ignore[misc]
        def __init__(self, **kwargs):  # type: ignore
            super().__init__(**kwargs)

    class GenerateContentRequestParametersDict(dict):  # type: ignore[misc]
        def __init__(self, **kwargs):  # type: ignore
            super().__init__(**kwargs)

    ToolConfigDict = Dict[str, Any]

    class GenerateContentRequestDict(GenerateContentRequestParametersDict):  # type: ignore[misc]
        def __init__(self, **kwargs):  # type: ignore
            # Extract specific fields
            self.generationConfig = kwargs.get('generationConfig')
            self.tools = kwargs.get('tools')
            super().__init__(**kwargs)

    class GenerateContentResponse(BaseLiteLLMOpenAIResponseObject): # type: ignore[misc]
        def __init__(self, **kwargs):  # type: ignore
            super().__init__(**kwargs)
            self._hidden_params = kwargs.get('_hidden_params', {})