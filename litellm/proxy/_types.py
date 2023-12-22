from pydantic import BaseModel, Extra, Field, root_validator
from typing import Optional, List, Union, Dict, Literal
from datetime import datetime
import uuid, json

class LiteLLMBase(BaseModel):
    """
    Implements default functions, all pydantic objects should have.
    """
    def json(self, **kwargs):
        try:
            return self.model_dump() # noqa
        except:
            # if using pydantic v1
            return self.dict()


######### Request Class Definition ######
class ProxyChatCompletionRequest(LiteLLMBase):
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

class ModelInfoDelete(LiteLLMBase):
    id: Optional[str]


class ModelInfo(LiteLLMBase):
    id: Optional[str]
    mode: Optional[Literal['embedding', 'chat', 'completion']]
    input_cost_per_token: Optional[float] = 0.0
    output_cost_per_token: Optional[float] = 0.0
    max_tokens: Optional[int] = 2048 # assume 2048 if not set

    # for azure models we need users to specify the base model, one azure you can call deployments - azure/my-random-model
    # we look up the base model in model_prices_and_context_window.json
    base_model: Optional[Literal
                         [
                             'gpt-4-1106-preview', 
                             'gpt-4-32k', 
                             'gpt-4', 
                             'gpt-3.5-turbo-16k', 
                             'gpt-3.5-turbo',
                             'text-embedding-ada-002',
                         ]
                        ]

    class Config:
        extra = Extra.allow  # Allow extra fields
        protected_namespaces = ()

    
    @root_validator(pre=True)
    def set_model_info(cls, values):
        if values.get("id") is None:
            values.update({"id": str(uuid.uuid4())})
        if values.get("mode") is None: 
            values.update({"mode": None})
        if values.get("input_cost_per_token") is None: 
            values.update({"input_cost_per_token": None})
        if values.get("output_cost_per_token") is None: 
            values.update({"output_cost_per_token": None})
        if values.get("max_tokens") is None:
            values.update({"max_tokens": None})
        if values.get("base_model") is None:
            values.update({"base_model": None})
        return values



class ModelParams(LiteLLMBase):
    model_name: str
    litellm_params: dict
    model_info: ModelInfo
    
    class Config:
        protected_namespaces = ()
    
    @root_validator(pre=True)
    def set_model_info(cls, values):
        if values.get("model_info") is None:
            values.update({"model_info": ModelInfo()})
        return values

class GenerateKeyRequest(LiteLLMBase):
    duration: Optional[str] = "1h"
    models: Optional[list] = []
    aliases: Optional[dict] = {}
    config: Optional[dict] = {}
    spend: Optional[float] = 0
    user_id: Optional[str] = None
    max_parallel_requests: Optional[int] = None
    metadata: Optional[dict] = {}

class UpdateKeyRequest(LiteLLMBase):
    key: str
    duration: Optional[str] = None
    models: Optional[list] = None
    aliases: Optional[dict] = None
    config: Optional[dict] = None
    spend: Optional[float] = None
    user_id: Optional[str] = None
    max_parallel_requests: Optional[int] = None
    metadata: Optional[dict] = {}

class UserAPIKeyAuth(LiteLLMBase): # the expected response object for user api key auth
    """
    Return the row in the db
    """
    api_key: Optional[str] = None
    models: list = []
    aliases: dict = {}
    config: dict = {}
    spend: Optional[float] = 0
    user_id: Optional[str] = None
    max_parallel_requests: Optional[int] = None
    duration: str = "1h"
    metadata: dict = {}

class GenerateKeyResponse(LiteLLMBase):
    key: str
    expires: datetime
    user_id: str

class _DeleteKeyObject(LiteLLMBase):
    key: str

class DeleteKeyRequest(LiteLLMBase):
    keys: List[_DeleteKeyObject]

class NewUserRequest(GenerateKeyRequest):
    max_budget: Optional[float] = None

class NewUserResponse(GenerateKeyResponse):
    max_budget: Optional[float] = None

class ConfigGeneralSettings(LiteLLMBase):
    """
    Documents all the fields supported by `general_settings` in config.yaml
    """
    completion_model: Optional[str] = Field(None, description="proxy level default model for all chat completion calls") 
    use_azure_key_vault: Optional[bool] = Field(None, description="load keys from azure key vault")
    master_key: Optional[str] = Field(None, description="require a key for all calls to proxy")
    database_url: Optional[str] = Field(None, description="connect to a postgres db - needed for generating temporary keys + tracking spend / key")
    otel: Optional[bool] = Field(None, description="[BETA] OpenTelemetry support - this might change, use with caution.")
    custom_auth: Optional[str] = Field(None, description="override user_api_key_auth with your own auth script - https://docs.litellm.ai/docs/proxy/virtual_keys#custom-auth")
    max_parallel_requests: Optional[int] = Field(None, description="maximum parallel requests for each api key")
    infer_model_from_keys: Optional[bool] = Field(None, description="for `/models` endpoint, infers available model based on environment keys (e.g. OPENAI_API_KEY)")
    background_health_checks: Optional[bool] = Field(None, description="run health checks in background")
    health_check_interval: int = Field(300, description="background health check interval in seconds")
    

class ConfigYAML(LiteLLMBase):
    """
    Documents all the fields supported by the config.yaml
    """
    model_list: Optional[List[ModelParams]] = Field(None, description="List of supported models on the server, with model-specific configs")
    litellm_settings: Optional[dict] = Field(None, description="litellm Module settings. See __init__.py for all, example litellm.drop_params=True, litellm.set_verbose=True, litellm.api_base, litellm.cache")
    general_settings: Optional[ConfigGeneralSettings] = None
    class Config:
        protected_namespaces = ()
