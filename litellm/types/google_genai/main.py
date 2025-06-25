# Import types from the Google GenAI SDK
from typing import Optional

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


class GenerateContentRequestDict(types._GenerateContentParametersDict, total=False):
    generationConfig: Optional[GenerateContentConfigDict]