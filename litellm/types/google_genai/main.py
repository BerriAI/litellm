# Import types from the Google GenAI SDK
from typing import Any, Dict, List, Union

try:
    from google.genai import types
    ContentListUnion = types.ContentListUnion
    ContentListUnionDict = types.ContentListUnionDict
    GenerateContentConfigOrDict = types.GenerateContentConfigOrDict
    GenerateContentResponse = types.GenerateContentResponse
except ImportError:
    ContentListUnion = Union[List[Dict[str, Any]], str]
    ContentListUnionDict = Union[List[Dict[str, Any]], str, Dict[str, Any]]
    GenerateContentConfigOrDict = Union[Dict[str, Any], None]
    GenerateContentResponse = Any