"""LiteLLM-owned request kwargs declared as typed fields; types/utils.py splices these with the callback and pricing
models and KWARG_ARTIFACTS into all_litellm_params."""

from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypeAlias

from pydantic import BeforeValidator, Field

if TYPE_CHECKING:
    import httpx
    from aiohttp import ClientSession
    from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
    from litellm.router_strategy.complexity_router.context_compaction import CompactionState
    from litellm.router_utils.fallback_event_handlers import AttemptedFallbackTargets
    from litellm.types.caching import DynamicCacheControl
    from litellm.types.llms.openai import ChatCompletionAssistantMessage, ChatCompletionUserMessage
    from litellm.types.proxy.litellm_pre_call_utils import SecretFields
    from litellm.types.router import ConfigurableClientsideParamsCustomAuth, DeploymentTypedDict, RetryPolicy
    from litellm.types.router_weights import RouterWeights
    from litellm.types.utils import ModelResponse, ModelResponseStream, ProviderSpecificHeader

    ProviderClient: TypeAlias = (
        OpenAI
        | AsyncOpenAI
        | AzureOpenAI
        | AsyncAzureOpenAI
        | HTTPHandler
        | AsyncHTTPHandler
        | httpx.Client
        | httpx.AsyncClient
    )
    MockResponse: TypeAlias = (
        str | Exception | Mapping[str, object] | Sequence[float] | ModelResponse | ModelResponseStream
    )

RetryStrategy: TypeAlias = Literal["constant_retry", "exponential_backoff_retry"]
AgenticSurface: TypeAlias = Literal["chat_completions", "responses"]
RoutingStrategyName: TypeAlias = Literal[
    "simple-shuffle",
    "least-busy",
    "usage-based-routing",
    "latency-based-routing",
    "cost-based-routing",
    "usage-based-routing-v2",
    "lar1",
]

TRUSTED_CALLBACK_VARS_FIELD: Final = "litellm_trusted_callback_vars"
ADDRESSED_RESPONSE_ID_FIELD: Final = "_litellm_addressed_response_id"

WIRE_NAME: Final = "wire_name"


def wire(name: str) -> Mapping[str, str]:
    return MappingProxyType({WIRE_NAME: name})


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderConnection:
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    region_name: str | None = None
    headers: Mapping[str, str] | None = None
    provider_specific_header: "ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None" = None
    client: "ProviderClient | None" = None
    shared_session: "ClientSession | None" = None
    ssl_verify: bool | str | None = None
    request_timeout: float | None = None
    force_timeout: float | None = None
    stream_timeout: float | str | None = None
    max_retries: int | None = None
    tenant_id: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    azure_username: str | None = None
    azure_password: str | None = None
    azure_scope: str | None = None
    azure_ad_token_provider: Callable[[], str] | None = None
    litellm_credential_name: str | None = None
    configurable_clientside_auth_params: "Sequence[str | ConfigurableClientsideParamsCustomAuth] | None" = None
    use_xai_oauth: bool | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BedrockBatchConnection:
    # Bedrock rejects these names in request bodies, so register them as LiteLLM-owned
    aws_batch_role_arn: str | None = None
    s3_bucket_name: str | None = None
    s3_region_name: str | None = None
    s3_endpoint_url: str | None = None
    s3_output_bucket_name: str | None = None
    s3_bucket_owner: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_encryption_key_id: str | None = None
    bedrock_tags: Sequence[Mapping[str, str]] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ConnectionSettings:
    provider: ProviderConnection
    bedrock_batch: BedrockBatchConnection


@dataclass(frozen=True, slots=True, kw_only=True)
class DispatchOptions:
    custom_llm_provider: str | None = None
    azure: bool | None = None
    use_litellm_proxy: bool | None = None
    use_chat_completions_api: bool | None = None
    use_in_pass_through: bool | None = None
    allowed_openai_params: Sequence[str] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutingOptions:
    fallbacks: Sequence[str | Mapping[str, object]] | None = None
    context_window_fallback_dict: Mapping[str, str] | None = None
    num_retries: int | None = None
    retry_policy: "RetryPolicy | Mapping[str, object] | None" = None
    retry_strategy: RetryStrategy | None = None
    routing_strategy: RoutingStrategyName | None = None
    cooldown_time: float | None = None
    allowed_model_region: str | None = None
    enable_tag_filtering: bool | None = None
    fastest_response: bool | None = None
    provider_affinity_header: str | None = None
    search_tool_name: str | None = None
    model_list: "Sequence[DeploymentTypedDict] | None" = None


@dataclass(frozen=True, slots=True, kw_only=True)
class DeploymentOptions:
    model_info: Mapping[str, object] | None = None
    rpm: int | None = None
    tpm: int | None = None
    itpm: int | None = None
    otpm: int | None = None
    default_api_key_rpm_limit: int | None = None
    default_api_key_tpm_limit: int | None = None
    max_parallel_requests: int | None = None
    weight: int | None = None
    order: int | None = None
    tag_regex: Sequence[str] | None = None
    max_file_size_mb: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SpecializedRouterOptions:
    auto_router_config_path: str | None = None
    auto_router_config: str | None = None
    auto_router_default_model: str | None = None
    auto_router_embedding_model: str | None = None
    auto_router_max_input_chars: int | None = None
    auto_router_routing_compression: str | None = None
    auto_router_model_compression: str | None = None
    complexity_router_config: Mapping[str, object] | None = None
    complexity_router_default_model: str | None = None
    adaptive_router_config: Mapping[str, object] | None = None
    adaptive_router_default_model: str | None = None
    quality_router_config: Mapping[str, object] | None = None
    quality_router_default_model: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CachingOptions:
    caching: bool | None = None
    cache: "DynamicCacheControl | None" = None
    ttl: float | None = None
    enable_prompt_caching: bool | None = None
    caching_groups: Sequence[Sequence[str]] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CostOptions:
    cost_per_query: float | None = None
    base_model: str | None = None
    max_budget: float | None = None
    budget_duration: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservabilityOptions:
    id: str | None = None
    metadata: MutableMapping[str, object] | None = None  # mutable-ok: the router and logging write keys into it
    litellm_metadata: MutableMapping[str, object] | None = None  # mutable-ok: the proxy writes keys into it
    tags: Sequence[str] | None = None
    litellm_trace_id: str | None = None
    litellm_session_id: str | None = None
    litellm_request_debug: bool | None = None
    logger_fn: Callable[[Mapping[str, object]], None] | None = None
    verbose: bool | None = None
    no_log: bool | None = field(default=None, metadata=wire("no-log"))


@dataclass(frozen=True, slots=True, kw_only=True)
class AgenticLoopOptions:
    max_agentic_loops: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class GuardrailOptions:
    guardrails: Sequence[str] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptOptions:
    prompt_id: str | None = None
    prompt_variables: Mapping[str, object] | None = None
    prompt_version: str | None = None
    prompt_environment: str | None = None
    prompt_label: str | None = None
    litellm_system_prompt: str | None = None
    custom_prompt_dict: Mapping[str, object] | None = None
    roles: Mapping[str, object] | None = None
    final_prompt_value: str | None = None
    bos_token: str | None = None
    eos_token: str | None = None
    hf_model_name: str | None = None
    supports_system_message: bool | None = None
    ensure_alternating_roles: bool | None = None
    user_continue_message: "ChatCompletionUserMessage | None" = None
    assistant_continue_message: "ChatCompletionAssistantMessage | None" = None
    disable_add_transform_inline_image_block: bool | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseOptions:
    merge_reasoning_content_in_choices: bool | None = None
    enable_json_schema_validation: bool | None = None
    complete_response: bool | None = None
    keepalive_seconds: float | None = None
    allow_client_keepalive_override: bool | None = None


def _int_from_decimal_string(value: object) -> object:
    return int(value) if isinstance(value, str) and value.isascii() and value.isdecimal() else value


@dataclass(frozen=True, slots=True, kw_only=True)
class LiteLLMControlParams:
    stream_chunk_size: Annotated[int, BeforeValidator(_int_from_decimal_string), Field(strict=True, gt=0)] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class MockOptions:
    mock_response: "MockResponse | None" = None
    mock_timeout: bool | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class LiteLLMOptions:
    dispatch: DispatchOptions
    routing: RoutingOptions
    deployment: DeploymentOptions
    specialized_routers: SpecializedRouterOptions
    caching: CachingOptions
    cost: CostOptions
    observability: ObservabilityOptions
    agentic_loop: AgenticLoopOptions
    guardrails: GuardrailOptions
    prompt: PromptOptions
    response: ResponseOptions
    control: LiteLLMControlParams
    mock: MockOptions


@dataclass(frozen=True, slots=True, kw_only=True)
class CallState:
    litellm_call_id: str | None = None
    completion_call_id: str | None = None
    model_alias_map: Mapping[str, str] | None = None
    data_residency: str | None = None
    litellm_logging_obj: "Logging | None" = None
    preset_cache_key: str | None = None
    cache_key: str | None = None
    stream_response: "Mapping[str, ModelResponse] | None" = None
    context_compaction_state: "CompactionState | None" = field(default=None, metadata=wire("_context_compaction_state"))


@dataclass(frozen=True, slots=True, kw_only=True)
class AgenticLoopState:
    depth: int | None = field(default=None, metadata=wire("_agentic_loop_depth"))
    fingerprints: Sequence[str] | None = field(default=None, metadata=wire("_agentic_loop_fingerprints"))
    api_surface: Literal["chat_completions", "responses"] | None = field(
        default=None, metadata=wire("_agentic_loop_api_surface")
    )
    code_interpreter_active: bool | None = field(default=None, metadata=wire("_code_interpreter_interception_active"))
    code_interpreter_sandbox_key: str | None = field(
        default=None, metadata=wire("_code_interpreter_interception_sandbox_key")
    )
    code_interpreter_session_scoped: bool | None = field(
        default=None, metadata=wire("_code_interpreter_interception_session_scoped")
    )
    code_interpreter_converted_stream: bool | None = field(
        default=None, metadata=wire("_code_interpreter_interception_converted_stream")
    )
    websearch_emit_native_blocks: bool | None = field(
        default=None, metadata=wire("_websearch_interception_emit_native_blocks")
    )
    websearch_converted_stream: bool | None = field(
        default=None, metadata=wire("_websearch_interception_converted_stream")
    )
    headroom_converted_stream: bool | None = field(
        default=None, metadata=wire("_headroom_interception_converted_stream")
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RouterState:
    weights: "RouterWeights | None" = field(default=None, metadata=wire("_router_weights"))
    fallback_depth: int | None = None
    max_fallbacks: int | None = None
    attempted_targets: "AttemptedFallbackTargets | None" = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ProxyRequestState:
    proxy_server_request: Mapping[str, object] | None = None
    secret_fields: "SecretFields | None" = None
    trusted_callback_vars: Mapping[str, str] | None = field(default=None, metadata=wire(TRUSTED_CALLBACK_VARS_FIELD))
    addressed_response_id: str | None = field(default=None, metadata=wire(ADDRESSED_RESPONSE_ID_FIELD))
    strip_stream_usage: bool | None = field(default=None, metadata=wire("_litellm_strip_stream_usage"))
    client_side_timeout: bool | None = None
    model_file_id_mapping: Mapping[str, Mapping[str, str]] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class EntrypointState:
    acompletion: bool | None = None
    aembedding: bool | None = None
    aimg_generation: bool | None = None
    atext_completion: bool | None = None
    text_completion: bool | None = None
    allm_passthrough_route: bool | None = None
    async_call: bool | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class InternalState:
    call: CallState
    agentic_loop: AgenticLoopState
    router: RouterState
    proxy: ProxyRequestState
    entrypoint: EntrypointState


KWARG_ARTIFACTS: Final[tuple[str, ...]] = ("self", "use_client", "model_config", "rust")

LITELLM_OWNED_ROOTS: Final = (ConnectionSettings, LiteLLMOptions, InternalState)


def wire_names(owner: type) -> tuple[str, ...]:
    return tuple(owned.metadata.get(WIRE_NAME, owned.name) for owned in fields(owner))


def owned_wire_names(root: type) -> tuple[str, ...]:
    def names() -> Iterator[str]:
        for leaf in fields(root):
            if not is_dataclass(leaf.type):
                raise TypeError(f"{root.__name__}.{leaf.name} is not a dataclass leaf")
            yield from wire_names(leaf.type)  # pyright: ignore[reportArgumentType]  # Field.type admits str

    return tuple(names())


OWNED_KWARG_NAMES: Final = tuple(name for root in LITELLM_OWNED_ROOTS for name in owned_wire_names(root))
AGENTIC_LOOP_KWARG_NAMES: Final = (*wire_names(AgenticLoopState), *wire_names(AgenticLoopOptions))
BEDROCK_BATCH_KWARG_NAMES: Final = wire_names(BedrockBatchConnection)
