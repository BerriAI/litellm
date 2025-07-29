"""
litellm.Router Types - includes RouterConfig, UpdateRouterConfig, ModelInfo etc
"""

import datetime
import enum
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, get_type_hints

import httpx
from httpx import AsyncClient, Client
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Required, TypedDict

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

from ..exceptions import RateLimitError
from .completion import CompletionRequest
from .embedding import EmbeddingRequest
from .llms.openai import OpenAIFileObject
from .llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from .utils import ModelResponse, ProviderSpecificModelInfo


class ConfigurableClientsideParamsCustomAuth(TypedDict):
    api_base: str


CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS = Optional[
    List[Union[str, ConfigurableClientsideParamsCustomAuth]]
]


class ModelConfig(BaseModel):
    model_name: str
    litellm_params: Union[CompletionRequest, EmbeddingRequest]
    tpm: int
    rpm: int

    model_config = ConfigDict(protected_namespaces=())


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

    model_config = ConfigDict(protected_namespaces=())


class UpdateRouterConfig(BaseModel):
    """
    Set of params that you can modify via `router.update_settings()`.
    """

    routing_strategy_args: Optional[dict] = None
    routing_strategy: Optional[str] = None
    model_group_retry_policy: Optional[dict] = None
    allowed_fails: Optional[int] = None
    cooldown_time: Optional[float] = None
    num_retries: Optional[int] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    retry_after: Optional[float] = None
    fallbacks: Optional[List[dict]] = None
    context_window_fallbacks: Optional[List[dict]] = None

    model_config = ConfigDict(protected_namespaces=())


class ModelInfo(BaseModel):
    id: Optional[
        str
    ]  # Allow id to be optional on input, but it will always be present as a str in the model instance
    db_model: bool = (
        False  # used for proxy - to separate models which are stored in the db vs. config.
    )
    updated_at: Optional[datetime.datetime] = None
    updated_by: Optional[str] = None

    created_at: Optional[datetime.datetime] = None
    created_by: Optional[str] = None

    base_model: Optional[str] = (
        None  # specify if the base model is azure/gpt-3.5-turbo etc for accurate cost tracking
    )
    tier: Optional[Literal["free", "paid"]] = None

    """
    Team Model Specific Fields
    """
    # the team id that this model belongs to
    team_id: Optional[str] = None

    # the model_name that can be used by the team when making LLM calls
    team_public_model_name: Optional[str] = None

    def __init__(self, id: Optional[Union[str, int]] = None, **params):
        if id is None:
            id = str(uuid.uuid4())  # Generate a UUID if id is None or not provided
        elif isinstance(id, int):
            id = str(id)
        super().__init__(id=id, **params)

    model_config = ConfigDict(extra="allow")

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


class CredentialLiteLLMParams(BaseModel):
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    ## VERTEX AI ##
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None
    vertex_credentials: Optional[Union[str, dict]] = None
    ## UNIFIED PROJECT/REGION ##
    region_name: Optional[str] = None

    ## AWS BEDROCK / SAGEMAKER ##
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region_name: Optional[str] = None
    ## IBM WATSONX ##
    watsonx_region_name: Optional[str] = None


class CustomPricingLiteLLMParams(BaseModel):
    ## CUSTOM PRICING ##
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    input_cost_per_second: Optional[float] = None
    output_cost_per_second: Optional[float] = None
    input_cost_per_pixel: Optional[float] = None
    output_cost_per_pixel: Optional[float] = None


class GenericLiteLLMParams(CredentialLiteLLMParams, CustomPricingLiteLLMParams):
    """
    LiteLLM Params without 'model' arg (used across completion / assistants api)
    """

    custom_llm_provider: Optional[str] = None
    tpm: Optional[int] = None
    rpm: Optional[int] = None
    timeout: Optional[Union[float, str, httpx.Timeout]] = (
        None  # if str, pass in as os.environ/
    )
    stream_timeout: Optional[Union[float, str]] = (
        None  # timeout when making stream=True calls, if str, pass in as os.environ/
    )
    max_retries: Optional[int] = None
    organization: Optional[str] = None  # for openai orgs
    configurable_clientside_auth_params: CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS = None
    litellm_credential_name: Optional[str] = None

    ## LOGGING PARAMS ##
    litellm_trace_id: Optional[str] = None

    max_file_size_mb: Optional[float] = None

    # Deployment budgets
    max_budget: Optional[float] = None
    budget_duration: Optional[str] = None
    use_in_pass_through: Optional[bool] = False
    use_litellm_proxy: Optional[bool] = False
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    merge_reasoning_content_in_choices: Optional[bool] = False
    model_info: Optional[Dict] = None
    mock_response: Optional[Union[str, ModelResponse, Exception, Any]] = None


    # auto-router params
    auto_router_config_path: Optional[str] = None
    auto_router_config: Optional[str] = None
    auto_router_default_model: Optional[str] = None
    auto_router_embedding_model: Optional[str] = None

    def __init__(
        self,
        custom_llm_provider: Optional[str] = None,
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
        ## LOGGING PARAMS ##
        litellm_trace_id: Optional[str] = None,
        ## UNIFIED PROJECT/REGION ##
        region_name: Optional[str] = None,
        ## VERTEX AI ##
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None,
        ## AWS BEDROCK / SAGEMAKER ##
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        ## IBM WATSONX ##
        watsonx_region_name: Optional[str] = None,
        input_cost_per_token: Optional[float] = None,
        output_cost_per_token: Optional[float] = None,
        input_cost_per_second: Optional[float] = None,
        output_cost_per_second: Optional[float] = None,
        max_file_size_mb: Optional[float] = None,
        # Deployment budgets
        max_budget: Optional[float] = None,
        budget_duration: Optional[str] = None,
        # Pass through params
        use_in_pass_through: Optional[bool] = False,
        # Dynamic param to force using litellm proxy
        use_litellm_proxy: Optional[bool] = False,
        # This will merge the reasoning content in the choices
        merge_reasoning_content_in_choices: Optional[bool] = False,
        model_info: Optional[Dict] = None,
        mock_response: Optional[Union[str, ModelResponse, Exception, Any]] = None,
        # auto-router params
        auto_router_config_path: Optional[str] = None,
        auto_router_config: Optional[str] = None,
        auto_router_default_model: Optional[str] = None,
        auto_router_embedding_model: Optional[str] = None,
        **params,
    ):
        args = locals()
        args.pop("max_retries", None)
        args.pop("self", None)
        args.pop("params", None)
        args.pop("__class__", None)
        if max_retries is not None and isinstance(max_retries, str):
            max_retries = int(max_retries)  # cast to int
        # We need to keep max_retries in args since it's a parameter of GenericLiteLLMParams
        args["max_retries"] = (
            max_retries  # Put max_retries back in args after popping it
        )
        super().__init__(**args, **params)

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


class LiteLLM_Params(GenericLiteLLMParams):
    """
    LiteLLM Params with 'model' requirement - used for completions
    """

    model: str
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def __init__(
        self,
        model: str,
        custom_llm_provider: Optional[str] = None,
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
        # OpenAI / Azure Whisper
        # set a max-size of file that can be passed to litellm proxy
        max_file_size_mb: Optional[float] = None,
        # will use deployment on pass-through endpoints if True
        use_in_pass_through: Optional[bool] = False,
        use_litellm_proxy: Optional[bool] = False,
        **params,
    ):
        args = locals()
        args.pop("max_retries", None)
        args.pop("self", None)
        args.pop("params", None)
        args.pop("__class__", None)
        if max_retries is not None and isinstance(max_retries, str):
            max_retries = int(max_retries)  # cast to int
        args["max_retries"] = max_retries
        super().__init__(**{ **args, **params })

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


class updateLiteLLMParams(GenericLiteLLMParams):
    # This class is used to update the LiteLLM_Params
    # only differece is model is optional
    model: Optional[str] = None


class updateDeployment(BaseModel):
    model_name: Optional[str] = None
    litellm_params: Optional[updateLiteLLMParams] = None
    model_info: Optional[ModelInfo] = None

    model_config = ConfigDict(protected_namespaces=())


class LiteLLMParamsTypedDict(TypedDict, total=False):
    model: str
    custom_llm_provider: Optional[str]
    tpm: Optional[int]
    rpm: Optional[int]
    order: Optional[int]
    weight: Optional[int]
    max_parallel_requests: Optional[int]
    api_key: Optional[str]
    api_base: Optional[str]
    api_version: Optional[str]
    timeout: Optional[Union[float, str, httpx.Timeout]]
    stream_timeout: Optional[Union[float, str]]
    max_retries: Optional[int]
    organization: Optional[Union[List, str]]  # for openai orgs
    configurable_clientside_auth_params: CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS  # for allowing api base switching on finetuned models
    ## DROP PARAMS ##
    drop_params: Optional[bool]
    ## UNIFIED PROJECT/REGION ##
    region_name: Optional[str]
    ## VERTEX AI ##
    vertex_project: Optional[str]
    vertex_location: Optional[str]
    ## AWS BEDROCK / SAGEMAKER ##
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]
    aws_region_name: Optional[str]
    ## IBM WATSONX ##
    watsonx_region_name: Optional[str]
    ## CUSTOM PRICING ##
    input_cost_per_token: Optional[float]
    output_cost_per_token: Optional[float]
    input_cost_per_second: Optional[float]
    output_cost_per_second: Optional[float]
    num_retries: Optional[int]
    ## MOCK RESPONSES ##
    mock_response: Optional[Union[str, ModelResponse, Exception]]

    # routing params
    # use this for tag-based routing
    tags: Optional[List[str]]

    # deployment budgets
    max_budget: Optional[float]
    budget_duration: Optional[str]


class DeploymentTypedDict(TypedDict, total=False):
    model_name: Required[str]
    litellm_params: Required[LiteLLMParamsTypedDict]
    model_info: dict


SPECIAL_MODEL_INFO_PARAMS = [
    "input_cost_per_token",
    "output_cost_per_token",
    "input_cost_per_character",
    "output_cost_per_character",
]


class Deployment(BaseModel):
    model_name: str
    litellm_params: LiteLLM_Params
    model_info: ModelInfo

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    def __init__(
        self,
        model_name: str,
        litellm_params: LiteLLM_Params,
        model_info: Optional[Union[ModelInfo, dict]] = None,
        **params,
    ):
        if model_info is None:
            model_info = ModelInfo()
        elif isinstance(model_info, dict):
            model_info = ModelInfo(**model_info)

        for (
            key
        ) in (
            SPECIAL_MODEL_INFO_PARAMS
        ):  # ensures custom pricing info is consistently in 'model_info'
            field = getattr(litellm_params, key, None)
            if field is not None:
                setattr(model_info, key, field)

        super().__init__(
            model_info=model_info,
            model_name=model_name,
            litellm_params=litellm_params,
            **params,
        )

    def to_json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)  # noqa
        except Exception as e:
            # if using pydantic v1
            return self.dict(**kwargs)

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
    no_deployments_with_tag_routing = (
        "Not allowed to access model due to tags configuration"
    )
    no_deployments_with_provider_budget_routing = (
        "No deployments available - crossed budget"
    )


class AllowedFailsPolicy(BaseModel):
    """
    Use this to set a custom number of allowed fails/minute before cooling down a deployment
    If `AuthenticationErrorAllowedFails = 1000`, then 1000 AuthenticationError will be allowed before cooling down a deployment

    Mapping of Exception type to allowed_fails for each exception
    https://docs.litellm.ai/docs/exception_mapping
    """

    BadRequestErrorAllowedFails: Optional[int] = None
    AuthenticationErrorAllowedFails: Optional[int] = None
    TimeoutErrorAllowedFails: Optional[int] = None
    RateLimitErrorAllowedFails: Optional[int] = None
    ContentPolicyViolationErrorAllowedFails: Optional[int] = None
    InternalServerErrorAllowedFails: Optional[int] = None


class RetryPolicy(BaseModel):
    """
    Use this to set a custom number of retries per exception type
    If RateLimitErrorRetries = 3, then 3 retries will be made for RateLimitError
    Mapping of Exception type to number of retries
    https://docs.litellm.ai/docs/exception_mapping
    """

    BadRequestErrorRetries: Optional[int] = None
    AuthenticationErrorRetries: Optional[int] = None
    TimeoutErrorRetries: Optional[int] = None
    RateLimitErrorRetries: Optional[int] = None
    ContentPolicyViolationErrorRetries: Optional[int] = None
    InternalServerErrorRetries: Optional[int] = None


class AlertingConfig(BaseModel):
    """
    Use this configure alerting for the router. Receive alerts on the following events
    - LLM API Exceptions
    - LLM Responses Too Slow
    - LLM Requests Hanging

    Args:
        webhook_url: str            - webhook url for alerting, slack provides a webhook url to send alerts to
        alerting_threshold: Optional[float] = None - threshold for slow / hanging llm responses (in seconds)
    """

    webhook_url: str
    alerting_threshold: Optional[float] = 300


class ModelGroupInfo(BaseModel):
    model_group: str
    providers: List[str]
    max_input_tokens: Optional[float] = None
    max_output_tokens: Optional[float] = None
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    input_cost_per_pixel: Optional[float] = None
    mode: Optional[
        Union[
            str,
            Literal[
                "chat",
                "embedding",
                "completion",
                "image_generation",
                "audio_transcription",
                "rerank",
                "moderations",
            ],
        ]
    ] = Field(default="chat")
    tpm: Optional[int] = None
    rpm: Optional[int] = None
    supports_parallel_function_calling: bool = Field(default=False)
    supports_vision: bool = Field(default=False)
    supports_web_search: bool = Field(default=False)
    supports_url_context: bool = Field(default=False)
    supports_reasoning: bool = Field(default=False)
    supports_function_calling: bool = Field(default=False)
    supported_openai_params: Optional[List[str]] = Field(default=[])
    configurable_clientside_auth_params: CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS = None

    def __init__(self, **data):
        for field_name, field_type in get_type_hints(self.__class__).items():
            if field_type == bool and data.get(field_name) is None:
                data[field_name] = False
        super().__init__(**data)


class AssistantsTypedDict(TypedDict):
    custom_llm_provider: Literal["azure", "openai"]
    litellm_params: LiteLLMParamsTypedDict


class FineTuningConfig(BaseModel):
    custom_llm_provider: Literal["azure", "openai"]


class CustomRoutingStrategyBase:
    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        """
        Asynchronously retrieves the available deployment based on the given parameters.

        Args:
            model (str): The name of the model.
            messages (Optional[List[Dict[str, str]]], optional): The list of messages for a given request. Defaults to None.
            input (Optional[Union[str, List]], optional): The input for a given embedding request. Defaults to None.
            specific_deployment (Optional[bool], optional): Whether to retrieve a specific deployment. Defaults to False.
            request_kwargs (Optional[Dict], optional): Additional request keyword arguments. Defaults to None.

        Returns:
            Returns an element from litellm.router.model_list

        """
        pass

    def get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        """
        Synchronously retrieves the available deployment based on the given parameters.

        Args:
            model (str): The name of the model.
            messages (Optional[List[Dict[str, str]]], optional): The list of messages for a given request. Defaults to None.
            input (Optional[Union[str, List]], optional): The input for a given embedding request. Defaults to None.
            specific_deployment (Optional[bool], optional): Whether to retrieve a specific deployment. Defaults to False.
            request_kwargs (Optional[Dict], optional): Additional request keyword arguments. Defaults to None.

        Returns:
            Returns an element from litellm.router.model_list

        """
        pass


class RouterGeneralSettings(BaseModel):
    async_only_mode: bool = Field(
        default=False
    )  # this will only initialize async clients. Good for memory utils
    pass_through_all_models: bool = Field(
        default=False
    )  # if passed a model not llm_router model list, pass through the request to litellm.acompletion/embedding


class RouterRateLimitErrorBasic(ValueError):
    """
    Raise a basic error inside helper functions.
    """

    def __init__(
        self,
        model: str,
    ):
        self.model = model
        _message = f"{RouterErrors.no_deployments_available.value}."
        super().__init__(_message)


class RouterRateLimitError(ValueError):
    def __init__(
        self,
        model: str,
        cooldown_time: float,
        enable_pre_call_checks: bool,
        cooldown_list: List,
    ):
        self.model = model
        self.cooldown_time = cooldown_time
        self.enable_pre_call_checks = enable_pre_call_checks
        self.cooldown_list = cooldown_list
        _message = f"{RouterErrors.no_deployments_available.value}, Try again in {cooldown_time} seconds. Passed model={model}. pre-call-checks={enable_pre_call_checks}, cooldown_list={cooldown_list}"
        super().__init__(_message)


class RouterModelGroupAliasItem(TypedDict):
    model: str
    hidden: bool  # if 'True', don't return on `.get_model_list`


VALID_LITELLM_ENVIRONMENTS = [
    "development",
    "staging",
    "production",
]


class RoutingStrategy(enum.Enum):
    LEAST_BUSY = "least-busy"
    LATENCY_BASED = "latency-based-routing"
    COST_BASED = "cost-based-routing"
    USAGE_BASED_ROUTING_V2 = "usage-based-routing-v2"
    USAGE_BASED_ROUTING = "usage-based-routing"
    PROVIDER_BUDGET_LIMITING = "provider-budget-routing"


class RouterCacheEnum(enum.Enum):
    TPM = "global_router:{id}:{model}:tpm:{current_minute}"
    RPM = "global_router:{id}:{model}:rpm:{current_minute}"


class GenericBudgetWindowDetails(BaseModel):
    """Details about a provider's budget window"""

    budget_start: float
    spend_key: str
    start_time_key: str
    ttl_seconds: int


OptionalPreCallChecks = List[
    Literal[
        "prompt_caching",
        "router_budget_limiting",
        "responses_api_deployment_check",
        "forward_client_headers_by_model_group",
    ]
]


class LiteLLM_RouterFileObject(TypedDict, total=False):
    """
    Tracking the litellm params hash, used for mapping the file id to the right model
    """

    litellm_params_sensitive_credential_hash: str
    file_object: OpenAIFileObject


@dataclass
class MockRouterTestingParams:
    mock_testing_fallbacks: Optional[bool] = None
    mock_testing_context_fallbacks: Optional[bool] = None
    mock_testing_content_policy_fallbacks: Optional[bool] = None

    @classmethod
    def from_kwargs(cls, kwargs: dict) -> "MockRouterTestingParams":
        from litellm.secret_managers.main import str_to_bool

        def extract_bool_param(name: str) -> Optional[bool]:
            value = kwargs.pop(name, None)
            return str_to_bool(value) if isinstance(value, str) else value

        return cls(
            mock_testing_fallbacks=extract_bool_param("mock_testing_fallbacks"),
            mock_testing_context_fallbacks=extract_bool_param(
                "mock_testing_context_fallbacks"
            ),
            mock_testing_content_policy_fallbacks=extract_bool_param(
                "mock_testing_content_policy_fallbacks"
            ),
        )

class ModelGroupSettings(BaseModel):
    forward_client_headers_to_llm_api: Optional[List[str]] = None

class PreRoutingHookResponse(BaseModel):
    """
    Response object from the pre-routing hook.

    Allows the Pre-Routing Hook to return a modified model and messages.

    Add fields that you expect to be modified by the pre-routing hook.
    """
    model: str
    messages: Optional[List[Dict[str, str]]]