from pydantic import BaseModel, Extra
from typing import Optional, List, Union, Dict, Literal
from datetime import datetime
import uuid
######### Request Class Definition ######
class ProxyChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    tools: Optional[List[str]] = None
    tool_choice: Optional[str] = None
    functions: Optional[List[str]] = None  # soon to be deprecated
    function_call: Optional[str] = None # soon to be deprecated

    # Optional LiteLLM params
    caching: Optional[bool] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    api_key: Optional[str] = None
    num_retries: Optional[int] = None
    context_window_fallback_dict: Optional[Dict[str, str]] = None
    fallbacks: Optional[List[str]] = None
    metadata: Optional[Dict[str, str]] = {}
    deployment_id: Optional[str] = None
    request_timeout: Optional[int] = None

    class Config:
        extra='allow' # allow params not defined here, these fall in litellm.completion(**kwargs)

class ModelInfo(BaseModel):
    id: Optional[str]
    mode: Optional[Literal['embedding', 'chat', 'completion']]
    input_cost_per_token: Optional[float]
    output_cost_per_token: Optional[float]
    max_tokens: Optional[int]
    base_model: Optional[Literal['gpt-4-1106-preview', 'gpt-4-32k', 'gpt-4', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo']]

    class Config:
        extra = Extra.allow  # Allow extra fields
        protected_namespaces = ()

class ModelInfoDelete(BaseModel):
    id: Optional[str]

class ModelParams(BaseModel):
    model_name: str
    litellm_params: dict
    model_info: Optional[ModelInfo]=None
 
    class Config:
        protected_namespaces = ()

class GenerateKeyRequest(BaseModel):
    duration: str = "1h"
    models: list = []
    aliases: dict = {}
    config: dict = {}
    spend: int = 0
    user_id: Optional[str] = None

class GenerateKeyResponse(BaseModel):
    key: str
    expires: datetime
    user_id: str

class _DeleteKeyObject(BaseModel):
    key: str

class DeleteKeyRequest(BaseModel):
    keys: List[_DeleteKeyObject]


class UserAPIKeyAuth(BaseModel): # the expected response object for user api key auth
    api_key: Optional[str] = None
    user_id: Optional[str] = None