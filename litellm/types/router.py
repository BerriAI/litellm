"""
litellm.Router Types - includes RouterConfig, UpdateRouterConfig, ModelInfo etc
"""

import datetime
import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Generic, Literal, TypeVar, get_type_hints

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Protocol, ReadOnly, Required, TypedDict, runtime_checkable

from litellm._uuid import uuid

from .completion import CompletionRequest
from .embedding import EmbeddingRequest
from .llms.openai import OpenAIFileObject
from .search import SearchProvider
from .utils import (
    CustomPricingLiteLLMParams,
    MirroredPricingParams,
    ModelResponse,
    StandardLoggingRoutingDecision,
)


class ConfigurableClientsideParamsCustomAuth(TypedDict):
    api_base: str


CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS = list[str | ConfigurableClientsideParamsCustomAuth] | None


class ModelConfig(BaseModel):
    model_name: str
    litellm_params: CompletionRequest | EmbeddingRequest
    tpm: int
    rpm: int

    model_config = ConfigDict(protected_namespaces=())


class RoutingGroup(BaseModel):
    """
    A group of models that share a routing strategy.
    """

    group_name: str
    models: list[str]
    routing_strategy: str
    routing_strategy_args: dict | None = None

    model_config = ConfigDict(protected_namespaces=())


class RouterConfig(BaseModel):
    model_list: list[ModelConfig]

    redis_url: str | None = None
    redis_host: str | None = None
    redis_port: int | None = None
    redis_password: str | None = None

    cache_responses: bool | None = False
    cache_kwargs: dict | None = {}
    caching_groups: list[tuple[str, list[str]]] | None = None
    client_ttl: int | None = 3600
    num_retries: int | None = 0
    timeout: float | None = None
    default_litellm_params: dict[str, str] | None = {}
    set_verbose: bool | None = False
    fallbacks: list | None = []
    allowed_fails: int | None = None
    context_window_fallbacks: list | None = []
    model_group_alias: dict[str, list[str]] | None = {}
    retry_after: int | None = 0
    routing_strategy: Literal[
        "simple-shuffle",
        "least-busy",
        "usage-based-routing",
        "latency-based-routing",
    ] = "simple-shuffle"
    routing_groups: list[RoutingGroup] | None = None

    model_config = ConfigDict(protected_namespaces=())


class RetryPolicy(BaseModel):
    """
    Use this to set a custom number of retries per exception type
    If RateLimitErrorRetries = 3, then 3 retries will be made for RateLimitError
    Mapping of Exception type to number of retries
    https://docs.litellm.ai/docs/exception_mapping
    """

    BadRequestErrorRetries: int | None = None
    AuthenticationErrorRetries: int | None = None
    TimeoutErrorRetries: int | None = None
    RateLimitErrorRetries: int | None = None
    ContentPolicyViolationErrorRetries: int | None = None
    InternalServerErrorRetries: int | None = None


class UpdateRouterConfig(BaseModel):
    """
    Set of params that you can modify via `router.update_settings()`.
    """

    routing_strategy_args: dict | None = None
    routing_strategy: str | None = None
    routing_groups: list[RoutingGroup] | None = None
    retry_policy: RetryPolicy | None = None
    model_group_retry_policy: dict[str, RetryPolicy] | None = None
    model_group_affinity_config: dict[str, list[str]] | None = None
    allowed_fails: int | None = None
    cooldown_time: float | None = None
    num_retries: int | None = None
    timeout: float | None = None
    max_retries: int | None = None
    retry_after: float | None = None
    fallbacks: list[dict] | None = None
    context_window_fallbacks: list[dict] | None = None
    model_group_alias: dict[str, str | dict] | None = {}
    enable_tag_filtering: bool | None = None
    tag_routing_prefix: str | None = None

    model_config = ConfigDict(protected_namespaces=())


def _as_utc(value: datetime.datetime | None) -> datetime.datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.timezone.utc)
    return value.astimezone(datetime.timezone.utc)


class ModelInfo(MirroredPricingParams):
    id: str | None  # Allow id to be optional on input, but it will always be present as a str in the model instance
    db_model: bool = False  # used for proxy - to separate models which are stored in the db vs. config.
    updated_at: datetime.datetime | None = None
    updated_by: str | None = None

    created_at: datetime.datetime | None = None
    created_by: str | None = None

    base_model: str | None = None  # specify if the base model is azure/gpt-3.5-turbo etc for accurate cost tracking
    tier: Literal["free", "paid"] | None = None

    """
    Team Model Specific Fields
    """
    # the team id that this model belongs to
    team_id: str | None = None

    # the model_name that can be used by the team when making LLM calls
    team_public_model_name: str | None = None

    # admin-toggled pause flag; mirrors LiteLLM_ProxyModelTable.blocked
    blocked: bool | None = None

    # Bounds live on the model rather than litellm.constants: names there reach
    # litellm/__init__ through several modules' star re-exports, and a Final rebound that
    # way trips the basedpyright gate.
    MAX_PTU_COUNT: ClassVar[int] = 1_000_000
    MAX_COST_PER_PTU_PER_HOUR: ClassVar[float] = 1_000_000.0

    ptu_count: int | None = None
    cost_per_ptu_per_hour: float | None = None
    ptu_effective_from: datetime.datetime | None = None
    ptu_effective_to: datetime.datetime | None = None

    # when tag-based routing's "!" or "&" constraints eliminate every deployment
    # in this model group, fall back to the default-tagged pool instead of
    # raising no_deployments_with_tag_routing. Defaults to False (raise), so
    # existing "!" negation behavior is unchanged unless explicitly opted in.
    allow_fail_open: bool | None = None

    # per-model-group override for router_settings.enable_tag_filtering; unset
    # defers to the router-wide default. Checked against any deployment in the
    # group, so set it consistently across every deployment sharing this
    # model_name. A request-level enable_tag_filtering=True (from key/team
    # settings) still wins over this, exactly as it already does over the
    # router-wide default.
    enable_tag_filtering: bool | None = None

    def __init__(self, id: str | int | None = None, **params) -> None:
        if id is None:
            id = str(uuid.uuid4())  # Generate a UUID if id is None or not provided
        elif isinstance(id, int):
            id = str(id)
        super().__init__(id=id, **params)

    @model_validator(mode="after")
    def _validate_ptu_bounds(self) -> "ModelInfo":
        if self.ptu_count is not None and not 0 < self.ptu_count <= self.MAX_PTU_COUNT:
            raise ValueError(f"ptu_count must be a positive integer no greater than {self.MAX_PTU_COUNT}")
        if (
            self.cost_per_ptu_per_hour is not None
            and not 0 <= self.cost_per_ptu_per_hour <= self.MAX_COST_PER_PTU_PER_HOUR
        ):
            raise ValueError(
                f"cost_per_ptu_per_hour must be a finite number between 0 and {self.MAX_COST_PER_PTU_PER_HOUR}"
            )
        start: Final = _as_utc(self.ptu_effective_from)
        end: Final = _as_utc(self.ptu_effective_to)
        if start is not None and end is not None and end <= start:
            raise ValueError("ptu_effective_to must be after ptu_effective_from")
        return self

    model_config = ConfigDict(extra="allow")

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class CredentialLiteLLMParams(BaseModel):
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    ## AZURE OAUTH ##
    # Without this field, ``get_deployment_credentials_with_provider``
    # round-trips ``litellm_params`` through a strict Pydantic dump and
    # silently drops the OAuth token before the files/batch/passthrough
    # callers see it, breaking Azure deployments configured with
    # ``azure_ad_token`` instead of a static ``api_key`` (#30235).
    azure_ad_token: str | None = None
    ## VERTEX AI ##
    vertex_project: str | None = None
    vertex_location: str | None = None
    vertex_credentials: str | dict | None = None
    ## UNIFIED PROJECT/REGION ##
    region_name: str | None = None

    ## OBJECT STORAGE (files / batches) ##
    gcs_bucket_name: str | None = None

    ## AWS BEDROCK / SAGEMAKER ##
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    aws_region_name: str | None = None
    aws_session_name: str | None = None
    aws_profile_name: str | None = None
    aws_role_name: str | None = None
    aws_web_identity_token: str | None = None
    aws_sts_endpoint: str | None = None
    aws_external_id: str | None = None
    aws_bedrock_runtime_endpoint: str | None = None
    aws_bedrock_project_id: str | None = None
    s3_bucket_name: str | None = None
    s3_region_name: str | None = None
    s3_encryption_key_id: str | None = None
    aws_batch_role_arn: str | None = None
    s3_output_bucket_name: str | None = None
    bedrock_tags: list | None = None
    ## IBM WATSONX ##
    watsonx_region_name: str | None = None


_RESERVED_INIT_KEYS: Final = frozenset({"self", "params", "__class__"})


class GenericLiteLLMParams(CredentialLiteLLMParams, CustomPricingLiteLLMParams):
    """
    LiteLLM Params without 'model' arg (used across completion / assistants api)
    """

    custom_llm_provider: str | None = None
    tpm: int | None = None
    rpm: int | None = None
    itpm: int | None = None
    otpm: int | None = None
    timeout: float | str | httpx.Timeout | None = None  # if str, pass in as os.environ/
    stream_timeout: float | str | None = None  # timeout when making stream=True calls, if str, pass in as os.environ/
    max_retries: int | None = None
    organization: str | None = None  # for openai orgs
    configurable_clientside_auth_params: CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS = None
    litellm_credential_name: str | None = None

    ## LOGGING PARAMS ##
    litellm_trace_id: str | None = None

    max_file_size_mb: float | None = None

    # Proxy-wide default rate limits applied to any API key using this deployment
    # when the key does not have a model-specific tpm/rpm limit configured.
    default_api_key_tpm_limit: int | None = None
    default_api_key_rpm_limit: int | None = None

    # Deployment budgets
    max_budget: float | None = None
    budget_duration: str | None = None
    keepalive_seconds: float | None = None
    # keepalive_seconds is operator-only by default: a client's request-level
    # value is ignored unless the deployment opts in here. Prevents a client
    # from unilaterally enabling heartbeats (and the LB-idle-timeout evasion
    # that comes with them) for a deployment that never configured them.
    allow_client_keepalive_override: bool | None = False
    use_in_pass_through: bool | None = False
    use_litellm_proxy: bool | None = False
    use_chat_completions_api: bool | None = None
    use_xai_oauth: bool | None = Field(
        default=False,
        description="Use stored xAI OAuth credentials when no xAI API key is configured.",
    )
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    merge_reasoning_content_in_choices: bool | None = False
    model_info: dict | None = None
    mock_response: str | ModelResponse | Exception | Any | None = None

    # tag-based routing
    tags: list[str] | None = None
    # regex patterns matched against request headers for tag routing
    tag_regex: list[str] | None = None

    # auto-router params
    auto_router_config_path: str | None = None
    auto_router_config: str | None = None
    auto_router_default_model: str | None = None
    auto_router_embedding_model: str | None = None
    auto_router_max_input_chars: int | None = None

    # complexity-router params
    complexity_router_config: dict | None = None
    complexity_router_default_model: str | None = None

    # adaptive-router params
    adaptive_router_default_model: str | None = None
    adaptive_router_config: dict | None = None
    # quality-router params
    quality_router_config: dict | None = None
    quality_router_default_model: str | None = None

    # Vector Store Params
    vector_store_id: str | None = None
    milvus_text_field: str | None = None
    milvus_db_name: str | None = None
    milvus_partition_names: list[str] | None = None
    valkey_host: str | None = None
    valkey_port: int | None = None
    valkey_password: str | None = None
    valkey_ssl: bool | None = None
    valkey_text_field: str | None = None
    valkey_embedding_field: str | None = None

    @model_validator(mode="before")
    @classmethod
    def preprocess_input_data(cls, data: Any) -> Any:
        """
        Pre-process input data before validation:
        1. Filter out reserved Python keywords ('self', 'params', '__class__') to prevent
           'got multiple values for argument' errors when user data contains these keys.
        2. Convert max_retries from string to int if needed.
        """
        if isinstance(data, dict):
            filtered: Final = {k: v for k, v in data.items() if k not in _RESERVED_INIT_KEYS}
            if "max_retries" in filtered and isinstance(filtered["max_retries"], str):
                filtered["max_retries"] = int(filtered["max_retries"])
            return filtered
        return data

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class LiteLLM_Params(GenericLiteLLMParams):
    """
    LiteLLM Params with 'model' requirement - used for completions
    """

    model: str
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class updateLiteLLMParams(GenericLiteLLMParams):
    # This class is used to update the LiteLLM_Params
    # only differece is model is optional
    model: str | None = None


class updateDeployment(BaseModel):
    model_name: str | None = None
    litellm_params: updateLiteLLMParams | None = None
    model_info: ModelInfo | None = None
    blocked: bool | None = None

    model_config = ConfigDict(protected_namespaces=())


class LiteLLMParamsTypedDict(TypedDict, total=False):
    model: str
    custom_llm_provider: str | None
    tpm: int | None
    rpm: int | None
    itpm: int | None
    otpm: int | None
    order: int | None
    weight: int | None
    max_parallel_requests: int | None
    api_key: str | None
    api_base: str | None
    api_version: str | None
    timeout: float | str | httpx.Timeout | None
    stream_timeout: float | str | None
    max_retries: int | None
    organization: list | str | None  # for openai orgs
    configurable_clientside_auth_params: (
        CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS  # for allowing api base switching on finetuned models
    )
    ## DROP PARAMS ##
    drop_params: bool | None
    ## RESPONSES API → CHAT COMPLETIONS BRIDGE ##
    use_chat_completions_api: bool | None
    ## PASS-THROUGH ENDPOINTS ##
    use_in_pass_through: bool | None
    litellm_credential_name: str | None
    ## UNIFIED PROJECT/REGION ##
    region_name: str | None
    ## VERTEX AI ##
    vertex_project: str | None
    vertex_location: str | None
    ## AWS BEDROCK / SAGEMAKER ##
    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    aws_region_name: str | None
    aws_bedrock_project_id: str | None
    ## AWS S3 VECTORS ##
    vector_bucket_name: str | None
    index_name: str | None
    embedding_model: str | None
    ## IBM WATSONX ##
    watsonx_region_name: str | None
    ## CUSTOM PRICING ##
    input_cost_per_token: float | None
    output_cost_per_token: float | None
    input_cost_per_second: float | None
    output_cost_per_second: float | None
    output_cost_per_second_480p: ReadOnly[float | None]
    output_cost_per_second_1080p: float | None
    output_cost_per_second_4k: ReadOnly[float | None]
    num_retries: int | None
    ## MOCK RESPONSES ##
    mock_response: str | ModelResponse | Exception | None

    # routing params
    # use this for tag-based routing
    tags: list[str] | None
    # regex patterns matched against request headers (e.g. "^User-Agent:\\s*claude-code\\/")
    tag_regex: list[str] | None

    # deployment budgets
    max_budget: float | None
    budget_duration: str | None
    keepalive_seconds: float | None
    allow_client_keepalive_override: bool | None

    # per-deployment cooldown override
    cooldown_time: float | None


class DeploymentTypedDict(TypedDict, total=False):
    model_name: Required[str]
    litellm_params: Required[LiteLLMParamsTypedDict]
    model_info: dict


SPECIAL_MODEL_INFO_PARAMS = tuple(MirroredPricingParams.model_fields)


class Deployment(BaseModel):
    model_name: str
    litellm_params: LiteLLM_Params
    model_info: ModelInfo

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    def __init__(
        self,
        model_name: str,
        litellm_params: LiteLLM_Params,
        model_info: ModelInfo | dict | None = None,
        **params,
    ) -> None:
        if model_info is None:
            model_info = ModelInfo()
        elif isinstance(model_info, dict):
            model_info = ModelInfo(**model_info)

        for key in SPECIAL_MODEL_INFO_PARAMS:  # ensures custom pricing info is consistently in 'model_info'
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
            return self.model_dump(**kwargs)
        except Exception:
            # if using pydantic v1
            return self.dict(**kwargs)

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class RouterErrors(enum.Enum):
    """
    Enum for router specific errors with common codes
    """

    user_defined_ratelimit_error = "Deployment over user-defined ratelimit."
    no_deployments_available = "No deployments available for selected model"
    no_deployments_with_tag_routing = "Not allowed to access model due to tags configuration"
    no_deployments_with_provider_budget_routing = "No deployments available - crossed budget"
    no_healthy_deployments = "There are no healthy deployments for this model"
    only_strategy_marker_deployments = (
        "Every deployment for it is a strategy router marker (auto_router/...), which is not a callable "
        "model, and no pre-routing strategy selected a deployment for this request"
    )


class AllowedFailsPolicy(BaseModel):
    """
    Use this to set a custom number of allowed fails/minute before cooling down a deployment
    If `AuthenticationErrorAllowedFails = 1000`, then 1000 AuthenticationError will be allowed before cooling down a deployment

    Mapping of Exception type to allowed_fails for each exception
    https://docs.litellm.ai/docs/exception_mapping
    """

    BadRequestErrorAllowedFails: int | None = None
    AuthenticationErrorAllowedFails: int | None = None
    TimeoutErrorAllowedFails: int | None = None
    RateLimitErrorAllowedFails: int | None = None
    ContentPolicyViolationErrorAllowedFails: int | None = None
    InternalServerErrorAllowedFails: int | None = None
    ServiceUnavailableErrorAllowedFails: int | None = None
    BadGatewayErrorAllowedFails: int | None = None
    NotFoundErrorAllowedFails: int | None = None


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
    alerting_threshold: float | None = 300


class ModelGroupInfo(BaseModel):
    model_group: str
    providers: list[str]
    max_input_tokens: float | None = None
    max_output_tokens: float | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    input_cost_per_pixel: float | None = None
    mode: (
        str
        | Literal["chat", "embedding", "completion", "image_generation", "audio_transcription", "rerank", "moderations"]
        | None
    ) = Field(default="chat")
    tpm: int | None = None
    rpm: int | None = None
    itpm: int | None = None
    otpm: int | None = None
    supports_parallel_function_calling: bool = Field(default=False)
    supports_vision: bool = Field(default=False)
    supports_web_search: bool = Field(default=False)
    supports_url_context: bool = Field(default=False)
    supports_reasoning: bool = Field(default=False)
    supports_function_calling: bool = Field(default=False)
    supported_reasoning_efforts: tuple[str, ...] | None = Field(default=None)
    supported_openai_params: list[str] | None = Field(default=[])
    configurable_clientside_auth_params: CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS = None

    def __init__(self, **data) -> None:
        for field_name, field_type in get_type_hints(self.__class__).items():
            if field_type is bool and data.get(field_name) is None:
                data[field_name] = False
        super().__init__(**data)


class AssistantsTypedDict(TypedDict):
    custom_llm_provider: Literal["azure", "openai"]
    litellm_params: LiteLLMParamsTypedDict


class SearchToolLiteLLMParams(TypedDict, total=False):
    """
    LiteLLM params for search tools.
    Search tools don't require a 'model' field like regular deployments.
    """

    search_provider: Required[SearchProvider]
    api_key: str | None
    api_base: str | None
    timeout: float | str | httpx.Timeout | None
    max_retries: int | None


class SearchToolInfoTypedDict(TypedDict, total=False):
    """Optional metadata about a search tool."""

    description: str


class SearchToolTypedDict(TypedDict, total=False):
    """
    Configuration for a search tool in the router.

    Example:
        {
            "search_tool_name": "litellm-search",
            "litellm_params": {
                "search_provider": "perplexity",
                "api_key": "os.environ/PERPLEXITYAI_API_KEY"
            }
        }
    """

    search_tool_name: Required[str]
    litellm_params: Required[SearchToolLiteLLMParams]
    search_tool_info: SearchToolInfoTypedDict


class GuardrailLiteLLMParams(TypedDict, total=False):
    """
    LiteLLM params for guardrails.
    """

    guardrail: Required[str]
    mode: Required[str]
    api_key: str | None
    api_base: str | None
    weight: int | None  # For load balancing


class GuardrailTypedDict(TypedDict, total=False):
    """
    Configuration for a guardrail in the router.
    """

    guardrail_name: Required[str]
    litellm_params: Required[GuardrailLiteLLMParams]
    callback: Any  # The CustomGuardrail instance
    id: str | None  # Unique identifier for the guardrail deployment


class FineTuningConfig(BaseModel):
    custom_llm_provider: Literal["azure", "openai"]


class CustomRoutingStrategyBase:
    async def async_get_available_deployment(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
        specific_deployment: bool | None = False,
        request_kwargs: dict | None = None,
    ) -> None:
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

    def get_available_deployment(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
        specific_deployment: bool | None = False,
        request_kwargs: dict | None = None,
    ) -> None:
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


class RouterGeneralSettings(BaseModel):
    async_only_mode: bool = Field(default=False)  # this will only initialize async clients. Good for memory utils
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
    ) -> None:
        self.model = model
        _message: Final = f"{RouterErrors.no_deployments_available.value}."
        super().__init__(_message)


class RouterRateLimitError(ValueError):
    def __init__(
        self,
        model: str,
        cooldown_time: float,
        enable_pre_call_checks: bool,
        cooldown_list: list,
    ) -> None:
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
    ITPM = "global_router:{id}:{model}:itpm:{current_minute}"
    OTPM = "global_router:{id}:{model}:otpm:{current_minute}"


class GenericBudgetWindowDetails(BaseModel):
    """Details about a provider's budget window"""

    budget_start: float
    spend_key: str
    start_time_key: str
    ttl_seconds: int


OptionalPreCallChecks = list[
    Literal[
        "prompt_caching",
        "router_budget_limiting",
        "responses_api_deployment_check",
        "deployment_affinity",
        "session_affinity",
        "forward_client_headers_by_model_group",
        "enforce_model_rate_limits",
        "encrypted_content_affinity",
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
    mock_testing_fallbacks: bool | None = None
    mock_testing_context_fallbacks: bool | None = None
    mock_testing_content_policy_fallbacks: bool | None = None

    @classmethod
    def from_kwargs(cls, kwargs: dict) -> "MockRouterTestingParams":
        from litellm.secret_managers.main import str_to_bool

        def extract_bool_param(name: str) -> bool | None:
            value: Final = kwargs.pop(name, None)
            return str_to_bool(value) if isinstance(value, str) else value

        return cls(
            mock_testing_fallbacks=extract_bool_param("mock_testing_fallbacks"),
            mock_testing_context_fallbacks=extract_bool_param("mock_testing_context_fallbacks"),
            mock_testing_content_policy_fallbacks=extract_bool_param("mock_testing_content_policy_fallbacks"),
        )


class ModelGroupSettings(BaseModel):
    forward_client_headers_to_llm_api: list[str] | None = None


class PreRoutingHookResponse(BaseModel):
    """
    Response object from the pre-routing hook.

    Allows the Pre-Routing Hook to return a modified model and messages.

    Add fields that you expect to be modified by the pre-routing hook.
    """

    model: str
    messages: list[dict[str, Any]] | None
    routing_decision: StandardLoggingRoutingDecision | None = None
    session_affinity_ttl_seconds: int | None = None
    litellm_params: Mapping[str, object] | None = None


_PreRoutingStrategyT_co = TypeVar("_PreRoutingStrategyT_co", covariant=True)


@dataclass(frozen=True, slots=True)
class TaggedPreRoutingStrategy(Generic[_PreRoutingStrategyT_co]):
    """A pre-routing strategy paired with the deployment `tags` it was registered under."""

    tags: tuple[str, ...]
    strategy: _PreRoutingStrategyT_co


@dataclass(frozen=True, slots=True)
class ConsumedRequestTagsStamp:
    """The model group a tagged router rewrote to, plus the request tags spent selecting it."""

    model_group: str
    tags: tuple[str, ...]


@runtime_checkable
class PreRoutingStrategy(Protocol):
    """Structural interface shared by the auto / complexity / adaptive / quality routers."""

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: dict[str, Any],
        messages: list[dict[str, Any]] | None = None,
        input: "str | list[Any] | None" = None,
        specific_deployment: bool | None = False,
    ) -> "PreRoutingHookResponse | None": ...


class RoutingContext(BaseModel):
    """
    Passed through a Router's `plugins` pipeline before the routing decision is made.

    Each plugin reads and mutates this object; the next plugin sees the previous
    plugin's changes. `candidate_models` narrows as the pipeline runs -- Router
    only selects a deployment whose `litellm_params.model` survives the pipeline.

    `raw_messages` and `structured_messages` mirror the pattern
    `CustomGuardrail.apply_guardrail` uses: the message shape differs by API
    surface (chat completions, Anthropic /v1/messages, Responses API `input`,
    ...), so plugins that need a stable, provider-agnostic shape should read
    `structured_messages` (normalized to OpenAI chat-completions format);
    plugins that need the exact original payload can read `raw_messages`.
    """

    raw_messages: list[dict[str, Any]]
    structured_messages: list[dict[str, Any]]
    candidate_models: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
    signals: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class RoutingPlugin(Protocol):
    """Interface a custom routing plugin must implement to run in `Router(plugins=[...])`."""

    async def run(self, context: RoutingContext) -> RoutingContext: ...


@runtime_checkable
class ClassifierPlugin(Protocol):
    """Interface a custom classifier must implement to run as the complexity router's classifier_type='custom'.

    `classify` returns the name of the tier the request belongs to (a built-in tier value or label,
    or a tier_definitions name), or None to decline and let classifier_fallback decide.

    The context's `candidate_models` is an informational snapshot of every tier's models, unlike
    the narrowing surface RoutingPlugin filters: the returned tier decides the pool, so mutating
    the list is a no-op.
    """

    async def classify(self, context: RoutingContext) -> str | None: ...


class RequestType(str, enum.Enum):
    """Fixed v0 taxonomy. User-extensible types come in v1."""

    CODE_GENERATION = "code_generation"
    CODE_UNDERSTANDING = "code_understanding"
    TECHNICAL_DESIGN = "technical_design"
    ANALYTICAL_REASONING = "analytical_reasoning"
    WRITING = "writing"
    FACTUAL_LOOKUP = "factual_lookup"
    GENERAL = "general"


class AdaptiveRouterWeights(BaseModel):
    quality: float = Field(default=0.7, ge=0.0, le=1.0)
    cost: float = Field(default=0.3, ge=0.0, le=1.0)

    @field_validator("cost")
    @classmethod
    def _weights_sum_to_one(cls, v, info):
        q: Final = info.data.get("quality", 0.7)
        if abs(q + v - 1.0) > 0.001:
            raise ValueError(f"weights must sum to 1.0, got quality={q} + cost={v} = {q + v}")
        return v


class AdaptiveRouterConfig(BaseModel):
    available_models: list[str]
    weights: AdaptiveRouterWeights = Field(default_factory=AdaptiveRouterWeights)


class AdaptiveRouterPreferences(BaseModel):
    """model_info.adaptive_router_preferences — declared by each model."""

    model_config = ConfigDict(use_enum_values=False)

    quality_tier: int = Field(ge=1, le=3)
    strengths: list[RequestType] = Field(default_factory=list)
