# Import types from the Google GenAI SDK
from typing import TYPE_CHECKING, Any, Dict, Optional


from litellm.types.llms.openai import BaseLiteLLMOpenAIResponseObject

# During static type-checking we can rely on the real google-genai types.
if TYPE_CHECKING:
    from google.genai import types as _genai_types

    ContentListUnion = _genai_types.ContentListUnion
    ContentListUnionDict = _genai_types.ContentListUnionDict
    GenerateContentConfigOrDict = _genai_types.GenerateContentConfigOrDict
    GoogleGenAIGenerateContentResponse = _genai_types.GenerateContentResponse
    GenerateContentContentListUnionDict = _genai_types.ContentListUnionDict
    GenerateContentConfigDict = _genai_types.GenerateContentConfigDict
    GenerateContentRequestParametersDict = _genai_types._GenerateContentParametersDict
    ToolConfigDict = _genai_types.ToolConfigDict

    class GenerateContentRequestDict(GenerateContentRequestParametersDict):
        generationConfig: Optional[Any]
        tools: Optional[ToolConfigDict]

    class GenerateContentResponse(GoogleGenAIGenerateContentResponse, BaseLiteLLMOpenAIResponseObject):
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
    class GenerateContentConfigDict(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class GenerateContentRequestParametersDict(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    ToolConfigDict = Dict[str, Any]

    class GenerateContentRequestDict(GenerateContentRequestParametersDict):
        def __init__(self, **kwargs):
            # Extract specific fields
            self.generationConfig = kwargs.get("generationConfig")
            self.tools = kwargs.get("tools")
            super().__init__(**kwargs)

    class GenerateContentResponse(BaseLiteLLMOpenAIResponseObject):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._hidden_params = kwargs.get("_hidden_params", {})
