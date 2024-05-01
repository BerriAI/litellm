from typing import List, Optional, Union, Dict, Tuple, Literal
import httpx
from pydantic import BaseModel, validator
from .completion import CompletionRequest
from .embedding import EmbeddingRequest
import uuid, enum


class ModelConfig(BaseModel):
    model_name: str
    litellm_params: Union[CompletionRequest, EmbeddingRequest]
    tpm: int
    rpm: int

    class Config:
        protected_namespaces = ()


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

    class Config:
        protected_namespaces = ()


class UpdateRouterConfig(BaseModel):
    """
    Set of params that you can modify via `router.update_settings()`.
    """

    routing_strategy_args: Optional[dict] = None
    routing_strategy: Optional[str] = None
    allowed_fails: Optional[int] = None
    cooldown_time: Optional[float] = None
    num_retries: Optional[int] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    retry_after: Optional[float] = None
    fallbacks: Optional[List[dict]] = None
    context_window_fallbacks: Optional[List[dict]] = None


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
    timeout: Optional[Union[float, str, httpx.Timeout]] = (
        None  # if str, pass in as os.environ/
    )
    stream_timeout: Optional[Union[float, str]] = (
        None  # timeout when making stream=True calls, if str, pass in as os.environ/
    )
    max_retries: Optional[int] = None
    organization: Optional[str] = None  # for openai orgs
    ## VERTEX AI ##
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None
    ## AWS BEDROCK / SAGEMAKER ##
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region_name: Optional[str] = None

    def __init__(
        self,
        model: str,
        max_retries: Optional[Union[int, str]] = None,
        tpm: Optional[int] = None,
        rpm: Optional[int] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout: Optional[Union[float, str]] = None,  # if str, pass in as os.environ/
        stream_timeout: Optional[Union[float, str]] = (
            None  # timeout when making stream=True calls, if str, pass in as os.environ/
        ),
        organization: Optional[str] = None,  # for openai orgs
        ## VERTEX AI ##
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        ## AWS BEDROCK / SAGEMAKER ##
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        **params
    ):
        args = locals()
        args.pop("max_retries", None)
        args.pop("self", None)
        args.pop("params", None)
        args.pop("__class__", None)
        if max_retries is not None and isinstance(max_retries, str):
            max_retries = int(max_retries)  # cast to int
        super().__init__(max_retries=max_retries, **args, **params)

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

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


class updateLiteLLMParams(BaseModel):
    # This class is used to update the LiteLLM_Params
    # only differece is model is optional
    model: Optional[str] = None
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


class updateDeployment(BaseModel):
    model_name: Optional[str] = None
    litellm_params: Optional[updateLiteLLMParams] = None
    model_info: Optional[ModelInfo] = None

    class Config:
        protected_namespaces = ()


class Deployment(BaseModel):
    model_name: str
    litellm_params: LiteLLM_Params
    model_info: ModelInfo

    def __init__(
        self,
        model_name: str,
        litellm_params: LiteLLM_Params,
        model_info: Optional[Union[ModelInfo, dict]] = None,
        **params
    ):
        if model_info is None:
            model_info = ModelInfo()
        elif isinstance(model_info, dict):
            model_info = ModelInfo(**model_info)
        super().__init__(
            model_info=model_info,
            model_name=model_name,
            litellm_params=litellm_params,
            **params
        )

    def to_json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)  # noqa
        except Exception as e:
            # if using pydantic v1
            return self.dict(**kwargs)

    class Config:
        extra = "allow"
        protected_namespaces = ()

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


class RouterErrors(enum.Enum):
    """
    Enum for router specific errors with common codes
    """

    user_defined_ratelimit_error = "Deployment over user-defined ratelimit."
    no_deployments_available = "No deployments available for selected model"
