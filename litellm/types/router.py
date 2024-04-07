from typing import List, Optional, Union, Dict, Tuple, Literal

from pydantic import BaseModel, validator
from .completion import CompletionRequest
from .embedding import EmbeddingRequest
import uuid


class ModelConfig(BaseModel):
    model_name: str
    litellm_params: Union[CompletionRequest, EmbeddingRequest]
    tpm: int
    rpm: int


class RouterConfig(BaseModel):
    model_list: List[ModelConfig]

    redis_url: Optional[str] = None
    redis_host: Optional[str] = None
    redis_port: Optional[int] = None
    redis_password: Optional[str] = None

    cache_responses: Optional[bool] = False
    cache_kwargs: Optional[Dict] = {}
    caching_groups: Optional[List[Tuple[str, List[str]]]] = None
    client_ttl: Optional[int] = 3600
    num_retries: Optional[int] = 0
    timeout: Optional[float] = None
    default_litellm_params: Optional[Dict[str, str]] = {}
    set_verbose: Optional[bool] = False
    fallbacks: Optional[List] = []
    allowed_fails: Optional[int] = None
    context_window_fallbacks: Optional[List] = []
    model_group_alias: Optional[Dict[str, List[str]]] = {}
    retry_after: Optional[int] = 0
    routing_strategy: Literal[
        "simple-shuffle",
        "least-busy",
        "usage-based-routing",
        "latency-based-routing",
    ] = "simple-shuffle"


class ModelInfo(BaseModel):
    id: Optional[
        str
    ]  # Allow id to be optional on input, but it will always be present as a str in the model instance

    def __init__(self, id: Optional[Union[str, int]] = None, **params):
        if id is None:
            id = str(uuid.uuid4())  # Generate a UUID if id is None or not provided
        elif isinstance(id, int):
            id = str(id)
        super().__init__(id=id, **params)

    class Config:
        extra = "allow"

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class LiteLLM_Params(BaseModel):
    model: str
    tpm: Optional[int] = None
    rpm: Optional[int] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    timeout: Optional[Union[float, str]] = None  # if str, pass in as os.environ/
    stream_timeout: Optional[Union[float, str]] = (
        None  # timeout when making stream=True calls, if str, pass in as os.environ/
    )
    max_retries: int = 2  # follows openai default of 2
    organization: Optional[str] = None  # for openai orgs
    ## VERTEX AI ##
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None
    ## AWS BEDROCK / SAGEMAKER ##
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region_name: Optional[str] = None

    def __init__(self, max_retries: Optional[Union[int, str]] = None, **params):
        if max_retries is None:
            max_retries = 2
        elif isinstance(max_retries, str):
            max_retries = int(max_retries)  # cast to int
        super().__init__(max_retries=max_retries, **params)

    class Config:
        extra = "allow"

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class Deployment(BaseModel):
    model_name: str
    litellm_params: LiteLLM_Params
    model_info: ModelInfo

    def __init__(self, model_info: Optional[ModelInfo] = None, **params):
        if model_info is None:
            model_info = ModelInfo()
        super().__init__(model_info=model_info, **params)

    def to_json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)  # noqa
        except Exception as e:
            # if using pydantic v1
            return self.dict(**kwargs)

    class Config:
        extra = "allow"

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)
