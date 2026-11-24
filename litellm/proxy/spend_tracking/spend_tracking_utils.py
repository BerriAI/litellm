import asyncio
import ipaddress
import os
import re
import secrets
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from datetime import datetime as dt
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, cast, runtime_checkable

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    BILLED_RETRY_ATTEMPT_METADATA_KEY,
    CALLER_SESSION_ID_METADATA_KEY,
    CLI_SESSION_KEY_PREFIX,
    EMPTY_MAPPING,
    INTERNAL_CALL_ORIGIN_METADATA_KEY,
    KUBERNETES_POD_ROUTING_KEY,
    LITELLM_PROXY_MASTER_KEY_ALIAS,
    LITELLM_TRUNCATED_PAYLOAD_FIELD,
    LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE,
    LITTELM_CLI_SERVICE_ACCOUNT_NAME,
    LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
    MAX_STRING_LENGTH_COMPACTION_SUMMARY_IN_DB,
    REDACTED_BY_LITELM_STRING,
    ROUTE_CALL_TYPE_METADATA_KEY,
)
from litellm.constants import (
    MAX_STRING_LENGTH_PROMPT_IN_DB as DEFAULT_MAX_STRING_LENGTH_PROMPT_IN_DB,
)
from litellm.integrations.otel.model.metadata import caller_session_id
from litellm.integrations.otel.model.trace_controls import TraceControls, caller_trace_controls
from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
from litellm.litellm_core_utils.batch_utils import (
    BATCH_COST_REQUEST_ID_SUFFIX,
    charged_poll_cost,
    charged_video_id,
    pays_only_its_guardrail,
)
from litellm.litellm_core_utils.core_helpers import (
    as_str_mapping,
    get_litellm_metadata_from_kwargs,
)
from litellm.litellm_core_utils.get_llm_provider_logic import declared_authenticating_provider
from litellm.litellm_core_utils.internal_call_metadata import (
    BACKGROUND_RESPONSE_COST_REQUEST_ID_SUFFIX,
    inherits_parent_session,
    settles_background_response,
)
from litellm.litellm_core_utils.litellm_logging import (
    coerce_model_access_groups,
    is_valid_sha256_hash,
    request_model_access_groups_from_litellm_params,
)
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import message_field
from litellm.litellm_core_utils.redact_messages import numeric_compression_stats
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps, strip_null_bytes
from litellm.litellm_core_utils.secret_redaction import redact_tokens
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy._types import (
    SpendLogsMetadata,
    SpendLogsMetadataFields,
    SpendLogsPayload,
    SpendLogsRouterMetadata,
)
from litellm.proxy.route_llm_request import GENERATE_CONTENT_SPEND_ENDPOINTS, ProxyModelNotFoundError
from litellm.proxy.spend_tracking.rust_spend_log_rows import spendlogs_payload_rust, uses_spend_logs_ledger
from litellm.proxy.utils import PrismaClient, hash_token, query_raw_within_statement_timeout
from litellm.responses.litellm_completion_transformation.compaction_items import (
    compaction_summary,
    with_compaction_summary,
)
from litellm.secret_managers.main import str_to_bool
from litellm.types.router import DeploymentTypedDict, LiteLLM_Params, SilentExperimentFlag
from litellm.types.utils import (
    PROMPT_CARRYING_GUARDRAIL_FIELDS,
    AzureSpillover,
    CallTypes,
    ClaudeSubscriptionUsage,
    CostBreakdown,
    LlmProviders,
    StandardLoggingGuardrailInformation,
    StandardLoggingKubernetesPodRouting,
    StandardLoggingMCPToolCall,
    StandardLoggingModelInformation,
    StandardLoggingPayload,
    StandardLoggingPayloadErrorInformation,
    StandardLoggingVectorStoreRequest,
    VectorStoreSearchResponse,
)
from litellm.types.vector_stores import VectorStoreResultContent, VectorStoreSearchResult
from litellm.types.videos.utils import VIDEO_COST_REQUEST_ID_SUFFIX

if TYPE_CHECKING:
    from litellm.router import Router


_TENANT_PROMPT_STORAGE_FIELDS: Final = ("user_api_key_metadata", "user_api_key_team_metadata")
_SPEND_LOGS_PRIVACY_METADATA_ADAPTER: Final[TypeAdapter[Mapping[str, object] | None]] = TypeAdapter(
    Mapping[str, object] | None
)


def _get_max_string_length_prompt_in_db() -> int:
    """
    Resolve prompt truncation threshold at runtime so values loaded later via
    proxy config environment_variables are honored.
    """
    max_length_str: Final = os.getenv("MAX_STRING_LENGTH_PROMPT_IN_DB")
    if max_length_str is None:
        return DEFAULT_MAX_STRING_LENGTH_PROMPT_IN_DB
    try:
        return int(max_length_str)
    except (TypeError, ValueError):
        return DEFAULT_MAX_STRING_LENGTH_PROMPT_IN_DB


def is_master_key(api_key: str | None, _master_key: str | None) -> bool:
    """
    Raw-only constant-time master-key comparison. The hashed form is never
    considered equivalent — only the raw master-key string matches.
    """
    if _master_key is None or api_key is None:
        return False
    return secrets.compare_digest(api_key, _master_key)


_is_master_key: Final = is_master_key


_HASHED_JWT_RE = re.compile(r"hashed-jwt-[a-fA-F0-9]{64}")
_NON_SECRET_KEY_ALIASES: Final = frozenset(
    {
        LITELLM_PROXY_MASTER_KEY_ALIAS,
        LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
        LITTELM_CLI_SERVICE_ACCOUNT_NAME,
    }
)


def _is_cli_session_alias(value: str, key_alias: object) -> bool:
    return value.startswith(f"{CLI_SESSION_KEY_PREFIX}-") and value == key_alias


def _is_non_secret_key_value(value: str, *, key_alias: object = None) -> bool:
    return (
        value in _NON_SECRET_KEY_ALIASES
        or is_valid_sha256_hash(value)
        or _HASHED_JWT_RE.fullmatch(value) is not None
        or _is_cli_session_alias(value, key_alias)
    )


def redact_logged_api_key(value: str | None, *, already_redacted: bool = False, key_alias: object = None) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    stripped: Final = re.sub(r"(?i)^bearer ", "", value)
    if not stripped:
        return None
    if already_redacted and _is_non_secret_key_value(stripped, key_alias=key_alias):
        return stripped
    return hash_token(stripped)


_STAMPED_METADATA_KEYS: Final = frozenset(
    (
        "router_metadata",
        "azure_spillover",
        "autorouter_savings",
        "autorouter_savings_estimate",
        "autorouter_baseline_observation",
        "used_client_oauth_token",
        "claude_subscription_usage",
        "upstream_provider",
        KUBERNETES_POD_ROUTING_KEY,
        "litellm_roi_estimator",
        "is_silent_experiment",
        BILLED_RETRY_ATTEMPT_METADATA_KEY,
        ROUTE_CALL_TYPE_METADATA_KEY,
        "user_api_key_user_email",
        "user_api_key_customer_org_id",
        "user_api_key_app_id",
        "user_api_key_app_group_id",
        "user_api_key_app_alias",
        "user_api_key_customer_org_alias",
        "user_api_key_customer_team_id",
        "user_api_key_customer_team_alias",
        "user_api_key_customer_member_id",
        "user_api_key_customer_member_alias",
    )
)


class _UpstreamProviderMetadata(TypedDict, total=False):
    upstream_provider: ReadOnly[str]


class _RouteCallTypeMetadata(TypedDict, total=False):
    route_call_type: ReadOnly[str]


def _adapter_route_call_type(  # pyright: ignore[reportUnusedFunction]  # shared with legacy_spend_log_rows and rust_spend_log_rows
    call_type: object, metadata: Mapping[str, object]
) -> str | None:
    route: Final = metadata.get("user_api_key_request_route")
    if (
        call_type != CallTypes.acompletion.value
        or not isinstance(route, str)
        or metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY)
    ):
        return None
    return next(
        (
            served.value
            for served in get_call_types_for_route(route) or ()
            if served.value in GENERATE_CONTENT_SPEND_ENDPOINTS
        ),
        None,
    )


def _stored_user_email(metadata: Mapping[str, object]) -> str | None:
    email: Final = metadata.get("user_api_key_user_email")
    return email if isinstance(email, str) and email.strip() else None


_CUSTOMER_METADATA_ADAPTER: Final[TypeAdapter[Mapping[object, object]]] = TypeAdapter(Mapping[object, object])


class _CustomerIdentityMetadata(TypedDict, total=False):
    user_api_key_app_id: ReadOnly[str]
    user_api_key_app_group_id: ReadOnly[str]
    user_api_key_app_alias: ReadOnly[str]
    user_api_key_customer_org_id: ReadOnly[str]
    user_api_key_customer_org_alias: ReadOnly[str]
    user_api_key_customer_team_id: ReadOnly[str]
    user_api_key_customer_team_alias: ReadOnly[str]
    user_api_key_customer_member_id: ReadOnly[str]
    user_api_key_customer_member_alias: ReadOnly[str]


_CUSTOMER_IDENTITY_ADAPTER: Final[TypeAdapter[_CustomerIdentityMetadata]] = TypeAdapter(_CustomerIdentityMetadata)


def _stored_customer_identity(metadata: Mapping[object, object]) -> _CustomerIdentityMetadata:
    values: Final = {
        key: value
        for key in _CustomerIdentityMetadata.__annotations__
        if isinstance(value := metadata.get(key), str) and value.strip()
    }
    return _CUSTOMER_IDENTITY_ADAPTER.validate_python(values)


def _get_spend_logs_metadata(  # pyright: ignore[reportUnusedFunction]  # shared with legacy_spend_log_rows and rust_spend_log_rows
    metadata: dict | None,
    applied_guardrails: list[str] | None = None,
    batch_models: list[str] | None = None,
    batch_successful_requests: int | None = None,
    batch_failed_requests: int | None = None,
    mcp_tool_call_metadata: StandardLoggingMCPToolCall | None = None,
    vector_store_request_metadata: list[StandardLoggingVectorStoreRequest] | None = None,
    guardrail_information: list[StandardLoggingGuardrailInformation] | None = None,
    usage_object: dict | None = None,
    model_map_information: StandardLoggingModelInformation | None = None,
    cold_storage_object_key: str | None = None,
    litellm_overhead_time_ms: float | None = None,
    upstream_provider: str | None = None,
    cost_breakdown: CostBreakdown | None = None,
    litellm_call_id: str | None = None,
    autorouter_savings: float | None = None,
    autorouter_savings_estimate: Mapping[str, JsonValue] | None = None,
    autorouter_baseline_observation: str | None = None,
    router_metadata: SpendLogsRouterMetadata | None = None,
    azure_spillover: AzureSpillover | None = None,
    used_client_oauth_token: bool | None = None,
    subscription_usage: ClaudeSubscriptionUsage | None = None,
    kubernetes_pod_routing: StandardLoggingKubernetesPodRouting | None = None,
    is_silent_experiment: bool = False,
    route_call_type: str | None = None,
) -> SpendLogsMetadata:
    upstream_provider_metadata: Final[_UpstreamProviderMetadata] = (
        {"upstream_provider": upstream_provider} if upstream_provider is not None else {}
    )
    route_call_type_metadata: Final[_RouteCallTypeMetadata] = (
        {"route_call_type": route_call_type} if route_call_type is not None else {}
    )
    if metadata is None:
        return SpendLogsMetadata(
            user_api_key=None,
            user_api_key_alias=None,
            user_api_key_team_id=None,
            user_api_key_project_id=None,
            user_api_key_project_alias=None,
            user_api_key_org_id=None,
            user_api_key_user_id=None,
            user_api_key_user_email=None,
            user_api_key_team_alias=None,
            spend_logs_metadata=None,
            requester_ip_address=None,
            user_agent=None,
            additional_usage_values=None,
            applied_guardrails=None,
            status="success",
            error_information=None,
            proxy_server_request=None,
            batch_models=None,
            batch_successful_requests=None,
            batch_failed_requests=None,
            mcp_tool_call_metadata=None,
            vector_store_request_metadata=None,
            model_map_information=None,
            usage_object=None,
            guardrail_information=None,
            internal_call_origin=None,
            litellm_roi_estimator=False,
            billed_retry_attempt=False,
            eval_information=None,
            cold_storage_object_key=cold_storage_object_key,
            litellm_overhead_time_ms=None,
            **upstream_provider_metadata,
            attempted_retries=None,
            max_retries=None,
            attempted_fallbacks=None,
            original_model_group=None,
            cost_breakdown=None,
            compression_savings=None,
            autorouter_savings=autorouter_savings,
            autorouter_savings_estimate=autorouter_savings_estimate,
            autorouter_baseline_observation=autorouter_baseline_observation,
            litellm_gateway_injected_cache=None,
            litellm_call_id=litellm_call_id,
            router_metadata=router_metadata,
            azure_spillover=azure_spillover,
            used_client_oauth_token=used_client_oauth_token,
            claude_subscription_usage=subscription_usage,
            kubernetes_pod_routing=None,
        )
    verbose_proxy_logger.debug("getting payload for SpendLogs, available keys in metadata: %s", [*metadata])

    retry_key: Final = BILLED_RETRY_ATTEMPT_METADATA_KEY
    billed_retry_attempt: Final = retry_key in metadata and metadata[retry_key] is True
    # Filter the metadata dictionary to include only the specified keys
    customer_metadata: Final = _CUSTOMER_METADATA_ADAPTER.validate_python(metadata)
    clean_metadata: Final = SpendLogsMetadata(
        **MappingProxyType(
            {key: metadata.get(key) for key in SpendLogsMetadata.__annotations__ if key not in _STAMPED_METADATA_KEYS}
        ),
        **_stored_customer_identity(customer_metadata),
        **upstream_provider_metadata,
        **route_call_type_metadata,
        autorouter_savings=autorouter_savings,
        autorouter_savings_estimate=autorouter_savings_estimate,
        autorouter_baseline_observation=autorouter_baseline_observation,
        router_metadata=router_metadata,
        azure_spillover=azure_spillover,
        used_client_oauth_token=used_client_oauth_token,
        claude_subscription_usage=subscription_usage,
        kubernetes_pod_routing=kubernetes_pod_routing,
        litellm_roi_estimator=metadata.get("litellm_roi_estimator") is True,
        **(SilentExperimentFlag(is_silent_experiment=True) if is_silent_experiment else SilentExperimentFlag()),
        billed_retry_attempt=billed_retry_attempt,
        user_api_key_user_email=_stored_user_email(metadata),
    )
    _raw_key: Final = clean_metadata.get("user_api_key")
    _trusted_hash: Final = metadata.get("user_api_key_hash")
    _key_alias: Final = metadata.get("user_api_key_alias")
    _already_redacted: Final = (
        isinstance(_trusted_hash, str)
        and _is_non_secret_key_value(_trusted_hash, key_alias=_key_alias)
        and _trusted_hash == _raw_key
    )
    clean_metadata["user_api_key"] = redact_logged_api_key(
        _raw_key, already_redacted=_already_redacted, key_alias=_key_alias
    )
    clean_metadata["applied_guardrails"] = applied_guardrails
    clean_metadata["batch_models"] = batch_models
    clean_metadata["batch_successful_requests"] = batch_successful_requests
    clean_metadata["batch_failed_requests"] = batch_failed_requests
    clean_metadata["mcp_tool_call_metadata"] = mcp_tool_call_metadata
    clean_metadata["vector_store_request_metadata"] = _get_vector_store_request_for_spend_logs_payload(
        vector_store_request_metadata, metadata=_SPEND_LOGS_PRIVACY_METADATA_ADAPTER.validate_python(metadata)
    )
    clean_metadata["guardrail_information"] = _sanitize_guardrail_information_for_spend_logs(
        guardrail_information, metadata=_SPEND_LOGS_PRIVACY_METADATA_ADAPTER.validate_python(metadata)
    )
    clean_metadata["usage_object"] = usage_object
    clean_metadata["model_map_information"] = model_map_information
    clean_metadata["cold_storage_object_key"] = cold_storage_object_key
    clean_metadata["litellm_overhead_time_ms"] = litellm_overhead_time_ms
    clean_metadata["cost_breakdown"] = cost_breakdown
    clean_metadata["litellm_call_id"] = litellm_call_id

    return clean_metadata


JOB_POLL_GUARDRAIL_COST_REQUEST_ID_SUFFIX: Final = "_guardrail_cost"


def _cost_row_request_id_suffix(call_type: str, metadata: Mapping[str, object], kwargs: Mapping[str, object]) -> str:
    if call_type == CallTypes.aretrieve_batch.value:
        return BATCH_COST_REQUEST_ID_SUFFIX
    if settles_background_response(call_type, metadata, kwargs):
        return BACKGROUND_RESPONSE_COST_REQUEST_ID_SUFFIX
    return ""


def get_spend_logs_id(
    call_type: str,
    response_obj: Mapping[str, object],
    kwargs: Mapping[str, object],
    metadata: Mapping[str, object] = EMPTY_MAPPING,
) -> str | None:
    standard_logging_payload = kwargs.get("standard_logging_object")
    candidate_ids: Final = (
        response_obj.get("id"),
        standard_logging_payload.get("id") if isinstance(standard_logging_payload, dict) else None,
        kwargs.get("litellm_call_id"),
    )
    resolved_id: Final = next(
        (candidate for candidate in candidate_ids if isinstance(candidate, str) and candidate), None
    )
    if resolved_id is None:
        return None
    response_cost: Final = charged_poll_cost(kwargs)
    charged_video: Final = charged_video_id(call_type, kwargs, resolved_id, response_cost)
    if charged_video is not None:
        return f"{charged_video}{VIDEO_COST_REQUEST_ID_SUFFIX}"
    call_id: Final = kwargs.get("litellm_call_id")
    if pays_only_its_guardrail(call_type, kwargs, response_cost) and isinstance(call_id, str) and call_id:
        return f"{call_id}{JOB_POLL_GUARDRAIL_COST_REQUEST_ID_SUFFIX}"
    return f"{resolved_id}{_cost_row_request_id_suffix(call_type, metadata, kwargs)}"


_MISSING_ATTRIBUTE: Final = object()


def _attribute_or_missing(source: object, name: str) -> object:
    return getattr(source, name, _MISSING_ATTRIBUTE)


@runtime_checkable
class _ModelDumpable(Protocol):
    def model_dump(self) -> object: ...


def _dumped_usage_info(usage_info: object) -> object:
    if isinstance(usage_info, _ModelDumpable):
        return usage_info.model_dump()
    instance_dict: Final = _attribute_or_missing(usage_info, "__dict__")
    if instance_dict is not _MISSING_ATTRIBUTE:
        return instance_dict
    return usage_info


def _extract_usage_for_ocr_call(  # pyright: ignore[reportUnusedFunction]  # shared with legacy_spend_log_rows and rust_spend_log_rows
    response_obj: object, response_obj_dict: Mapping[str, object]
) -> dict[str, object]:
    """
    Extract usage information for OCR/AOCR calls.

    OCR responses use usage_info (with pages_processed) instead of token-based usage.

    Args:
        response_obj: The raw response object (can be dict, BaseModel, or other)
        response_obj_dict: Dictionary representation of the response object

    Returns:
        A dict with prompt_tokens=0, completion_tokens=0, total_tokens=0,
        and pages_processed from usage_info.
    """
    usage_info = None

    # Try to extract usage_info from dict
    if isinstance(response_obj_dict, dict) and "usage_info" in response_obj_dict:
        usage_info = response_obj_dict.get("usage_info")

    # Try to extract usage_info from object attributes if not found in dict
    if not usage_info:
        attribute_usage_info: Final = _attribute_or_missing(response_obj, "usage_info")
        if attribute_usage_info is not _MISSING_ATTRIBUTE:
            usage_info = _dumped_usage_info(attribute_usage_info)

    # For OCR, we track pages instead of tokens
    if usage_info is not None:
        # Handle dict or object with attributes
        if isinstance(usage_info, dict):
            result: Final[dict[str, object]] = {
                "prompt_tokens": 0,  # OCR doesn't use traditional tokens
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            # Add all fields from usage_info, including pages_processed
            for key, value in usage_info.items():
                result[key] = value
            # Ensure pages_processed exists
            if "pages_processed" not in result:
                result["pages_processed"] = 0
            return result
        else:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "pages_processed": 0,
            }
    else:
        return {}


def get_request_model_access_groups(kwargs: Mapping[str, object] | None) -> tuple[str, ...]:
    """Model access groups that authorized this request, as stamped onto request metadata at auth time."""
    if kwargs is None:
        return ()

    standard_logging_payload: Final = kwargs.get("standard_logging_object")
    if isinstance(standard_logging_payload, Mapping):
        from_payload: Final = coerce_model_access_groups(standard_logging_payload.get("request_model_access_groups"))
        if from_payload:
            return from_payload

    litellm_params: Final = kwargs.get("litellm_params")
    if not isinstance(litellm_params, Mapping):
        return ()
    return request_model_access_groups_from_litellm_params(litellm_params)


def _string_or_empty(  # pyright: ignore[reportUnusedFunction]  # shared with legacy_spend_log_rows and rust_spend_log_rows
    value: object,
) -> str:
    return value if isinstance(value, str) else ""


def _deployment_provider(deployment: DeploymentTypedDict) -> str | None:
    litellm_params: Final[LiteLLM_Params] = LiteLLM_Params.model_validate(deployment["litellm_params"])
    return deployment_params_provider(litellm_params)


def deployment_params_provider(litellm_params: LiteLLM_Params) -> str | None:
    """The provider a deployment's ``litellm_params`` authenticate against, None when its model cannot be resolved."""
    if litellm.LiteLLMProxyChatConfig.should_use_litellm_proxy_by_default(litellm_params=litellm_params):
        return LlmProviders.LITELLM_PROXY.value
    declared: Final = declared_authenticating_provider(litellm_params.model, litellm_params.custom_llm_provider)
    if declared is not None:
        return declared
    try:
        _, provider, _, _ = litellm.get_llm_provider(
            model=litellm_params.model, custom_llm_provider=litellm_params.custom_llm_provider
        )
    except litellm.exceptions.BadRequestError:
        return None
    return provider or None


def _model_group_provider(  # pyright: ignore[reportUnusedFunction]  # shared with legacy_spend_log_rows and rust_spend_log_rows
    model_group: str, llm_router: "Router | None"
) -> str | None:
    if llm_router is None or not model_group:
        return None
    providers: Final = frozenset(
        provider
        for deployment in llm_router.registry().deployments(model_group) or ()
        if (provider := _deployment_provider(deployment)) is not None
    )
    return next(iter(providers)) if len(providers) == 1 else None


def _is_configured_model_group(  # pyright: ignore[reportUnusedFunction]  # shared with legacy_spend_log_rows and rust_spend_log_rows
    model_group: str, llm_router: "Router | None"
) -> bool:
    if llm_router is None or not model_group:
        return False
    registry: Final = llm_router.registry()
    return registry.resolve(model_group, kind="recognized") or model_group in registry.names(kind="team_public")


_TRUNCATION_MARKER: Final = re.compile(
    rf"\.\.\. \({re.escape(LITELLM_TRUNCATED_PAYLOAD_FIELD)} skipped \d+ chars\. "
    rf"{re.escape(LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE)}\) \.\.\."
)
_SCRUBBED_ERROR_TEXT_FIELDS: Final = frozenset(("error_message", "traceback"))


def _spend_log_worker_kwargs(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    params: Final = as_str_mapping(kwargs.get("litellm_params"))
    if params is None:
        return kwargs
    metadata: Final = as_str_mapping(params.get("litellm_metadata"))
    return {
        **kwargs,
        "litellm_params": {
            **params,
            **({"litellm_metadata": dict(metadata)} if metadata is not None else {}),
        },
    }


async def get_logging_payload_async(
    kwargs: Mapping[str, object] | None,
    response_obj: object,
    start_time: datetime,
    end_time: datetime,
    llm_router: "Router | None" = None,
    run_in_thread: Callable[[Callable[[], SpendLogsPayload]], Awaitable[SpendLogsPayload]] = asyncio.to_thread,
) -> SpendLogsPayload:
    from litellm.proxy.proxy_server import general_settings_view

    call_kwargs: Final[Mapping[str, object]] = kwargs if kwargs is not None else MappingProxyType({})
    offload: Final = general_settings_view().get("redact_tokens_in_spend_logs") is True
    build: Final = partial(
        get_logging_payload,
        kwargs=dict(_spend_log_worker_kwargs(call_kwargs)) if offload else dict(call_kwargs),
        response_obj=response_obj,
        start_time=start_time,
        end_time=end_time,
        llm_router=llm_router,
    )
    if offload:
        return await run_in_thread(build)
    return build()


SNAPSHOTLESS_COST_BREAKDOWN_KEY: Final = "snapshotless_cost_breakdown"
_OPTIONAL_COST_BREAKDOWN: Final[TypeAdapter[CostBreakdown | None]] = TypeAdapter(CostBreakdown | None)


def _snapshotless_cost_breakdown(  # pyright: ignore[reportUnusedFunction]  # shared with legacy_spend_log_rows and rust_spend_log_rows
    row_kwargs: Mapping[object, object],
) -> CostBreakdown | None:
    """The breakdown of a billed row that has no logging payload to carry it."""
    try:
        return _OPTIONAL_COST_BREAKDOWN.validate_python(row_kwargs.get(SNAPSHOTLESS_COST_BREAKDOWN_KEY))
    except ValidationError:
        return None


def get_logging_payload(
    kwargs: dict[str, object] | None,
    response_obj: object,
    start_time: datetime,
    end_time: datetime,
    llm_router: "Router | None" = None,
) -> SpendLogsPayload:
    if kwargs is None:
        kwargs = {}

    if uses_spend_logs_ledger():
        return spendlogs_payload_rust(kwargs, response_obj, start_time, end_time, llm_router)  # pyright: ignore[reportUnknownArgumentType]  # the logging kwargs dict is untyped upstream

    from litellm.proxy.spend_tracking import legacy_spend_log_rows

    return legacy_spend_log_rows.logging_payload(kwargs, response_obj, start_time, end_time, llm_router)


def caller_session_for_accounting(kwargs: Mapping[str, object]) -> str | None:
    params: Final = as_str_mapping(kwargs.get("litellm_params")) or MappingProxyType({})
    preserved: Final = next(
        (
            session
            for key in ("litellm_metadata", "metadata")
            if isinstance(session := (as_str_mapping(params.get(key)) or {}).get(CALLER_SESSION_ID_METADATA_KEY), str)
        ),
        None,
    )
    return _caller_session_for_spend_log(
        kwargs,
        MappingProxyType({CALLER_SESSION_ID_METADATA_KEY: preserved}) if preserved else None,
        trace=caller_trace_controls(kwargs),
    )


def _caller_session_for_spend_log(
    kwargs: Mapping[str, object], metadata: Mapping[str, object] | None, trace: TraceControls | None = None
) -> str | None:
    beside_trace_header: Final = metadata.get(CALLER_SESSION_ID_METADATA_KEY) if metadata else None
    caller_session: Final = (
        beside_trace_header
        if isinstance(beside_trace_header, str) and beside_trace_header
        else caller_session_id(kwargs, trace or TraceControls())
    )
    return caller_session if caller_session and "\x00" not in caller_session else None


def _get_session_id_for_spend_log(  # pyright: ignore[reportUnusedFunction]  # shared with legacy_spend_log_rows and rust_spend_log_rows
    kwargs: Mapping[str, object],
    metadata: Mapping[str, object] | None,
    standard_logging_payload: StandardLoggingPayload | None,
    omit_when_missing: bool,
    batch_trace_session_id: str | None = None,
) -> str | None:
    """Resolve the session id for the spend log row: `omit` honors metadata.session_id, else the parent session a proxy
    sub-call inherited, never a trace id; batch sessions outrank everything,
    then the conversation id the caller sent (also one an x-litellm-trace-id header displaced from the request's
    session), then trace ids.

    A session minted by `missing_session_id: generate` or copied from the request's trace id is not a conversation, so
    it yields to the trace id, which the Responses bridge points at the previous turn's session to keep a chain whole.
    """
    if omit_when_missing:
        session_id: Final = metadata.get("session_id") if metadata else None
        if session_id:
            return str(session_id)
        params: Final = as_str_mapping(kwargs.get("litellm_params")) or MappingProxyType({})
        return _caller_session_for_spend_log(kwargs, metadata=None) if inherits_parent_session(params) else None

    from litellm._uuid import uuid

    if batch_trace_session_id is not None:
        return batch_trace_session_id
    if litellm.request_correlation_in_logs and (caller_session := _caller_session_for_spend_log(kwargs, metadata)):
        return caller_session
    match standard_logging_payload:
        case {"trace_id": trace_id} if trace_id is not None:
            return str(trace_id)
        case _:
            litellm_trace_id: Final = kwargs.get("litellm_trace_id")
            return str(litellm_trace_id) if litellm_trace_id is not None else str(uuid.uuid4())


async def _query_raw_rows(
    prisma_client: PrismaClient,
    sql_query: str,
    *args: object,
) -> Sequence[Mapping[str, object]] | None:
    return await query_raw_within_statement_timeout(prisma_client, sql_query, *args)


async def get_spend_by_team(
    start_date: dt,
    end_date: dt,
    team_id: str | None,
    prisma_client: PrismaClient,
) -> Sequence[Mapping[str, object]]:
    sql_query: Final = """
    WITH SpendByModelApiKey AS (
        SELECT
            date_trunc('day', sl."startTime") AS group_by_day,
            COALESCE(tt.team_alias, 'Unassigned Team') AS team_name,
            sl.model,
            sl.api_key,
            SUM(sl.spend) AS model_api_spend,
            SUM(sl.total_tokens) AS model_api_tokens
        FROM
            "LiteLLM_SpendLogs" sl
        LEFT JOIN
            "LiteLLM_TeamTable" tt
        ON
            sl.team_id = tt.team_id
        WHERE
            sl."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
            AND sl."startTime" <  (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
            AND ($3::text IS NULL OR sl.team_id = $3)
        GROUP BY
            date_trunc('day', sl."startTime"),
            tt.team_alias,
            sl.model,
            sl.api_key
    )
        SELECT
            group_by_day,
            jsonb_agg(jsonb_build_object(
                'team_name', team_name,
                'total_spend', total_spend,
                'metadata', metadata
            )) AS teams
        FROM (
            SELECT
                group_by_day,
                team_name,
                SUM(model_api_spend) AS total_spend,
                jsonb_agg(jsonb_build_object(
                    'model', model,
                    'api_key', api_key,
                    'spend', model_api_spend,
                    'total_tokens', model_api_tokens
                )) AS metadata
            FROM
                SpendByModelApiKey
            GROUP BY
                group_by_day,
                team_name
        ) AS aggregated
        GROUP BY
            group_by_day
        ORDER BY
            group_by_day;
    """

    db_response: Final = await _query_raw_rows(prisma_client, sql_query, start_date, end_date, team_id)
    if db_response is None:
        return []

    return db_response


async def get_spend_by_team_and_customer(
    start_date: dt,
    end_date: dt,
    team_id: str,
    customer_id: str,
    prisma_client: PrismaClient,
) -> Sequence[Mapping[str, object]]:
    sql_query: Final = """
    WITH SpendByModelApiKey AS (
        SELECT
            date_trunc('day', sl."startTime") AS group_by_day,
            COALESCE(tt.team_alias, 'Unassigned Team') AS team_name,
            sl.end_user AS customer,
            sl.model,
            sl.api_key,
            SUM(sl.spend) AS model_api_spend,
            SUM(sl.total_tokens) AS model_api_tokens
        FROM 
            "LiteLLM_SpendLogs" sl
        LEFT JOIN 
            "LiteLLM_TeamTable" tt 
        ON 
            sl.team_id = tt.team_id
        WHERE
            sl."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
            AND sl."startTime" <  (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
            AND sl.team_id = $3
            AND sl.end_user = $4
        GROUP BY
            date_trunc('day', sl."startTime"),
            tt.team_alias,
            sl.end_user,
            sl.model,
            sl.api_key
    )
        SELECT
            group_by_day,
            jsonb_agg(jsonb_build_object(
                'team_name', team_name,
                'customer', customer,
                'total_spend', total_spend,
                'metadata', metadata
            )) AS teams_customers
        FROM (
            SELECT
                group_by_day,
                team_name,
                customer,
                SUM(model_api_spend) AS total_spend,
                jsonb_agg(jsonb_build_object(
                    'model', model,
                    'api_key', api_key,
                    'spend', model_api_spend,
                    'total_tokens', model_api_tokens
                )) AS metadata
            FROM 
                SpendByModelApiKey
            GROUP BY
                group_by_day,
                team_name,
                customer
        ) AS aggregated
        GROUP BY
            group_by_day
        ORDER BY
            group_by_day;
    """

    db_response: Final = await _query_raw_rows(prisma_client, sql_query, start_date, end_date, team_id, customer_id)
    if db_response is None:
        return []

    return db_response


_SENSITIVE_REQUEST_BODY_KEYS: Final = frozenset({"secret_fields"})
_REQUEST_BODY_CREDENTIAL_MASKER: Final = SensitiveDataMasker(extra_sensitive_patterns=frozenset({"apikey"}))
_TOOL_INPUT_BLOCK_TYPES: Final = frozenset({"tool_use", "server_tool_use", "mcp_tool_use"})
_TOOL_OUTPUT_BLOCK_TYPES: Final = frozenset({"tool_result", "mcp_tool_result", "function_call_output"})
_LATEST_TURN_STORAGE: Final = "latest_turn"
_RESPONSES_CALL_TYPES: Final = frozenset({"responses", "aresponses"})
_REQUEST_BODY_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_TEXT_BLOCK_TYPES: Final = frozenset({"text", "input_text"})
_OMITTED_BLOCK_PLACEHOLDERS: Final = MappingProxyType(
    {
        "image": "[image omitted]",
        "image_url": "[image omitted]",
        "input_image": "[image omitted]",
        "file": "[file omitted]",
        "input_file": "[file omitted]",
        "document": "[file omitted]",
        "input_audio": "[audio omitted]",
        "input_json": "[json omitted]",
    }
)


def stores_latest_turn_only() -> bool:
    from litellm.proxy.proxy_server import general_settings_view

    return general_settings_view().get("spend_logs_prompt_storage") == _LATEST_TURN_STORAGE


def stores_full_prompts_in_spend_logs(metadata: Mapping[str, object] | None = None) -> bool:
    return should_store_prompts_and_responses_in_spend_logs(metadata) and not stores_latest_turn_only()


def _block_type(block: object) -> str | None:
    block_type: Final = message_field(block, "type")
    return block_type if isinstance(block_type, str) else None


def _content_block_text(  # pyright: ignore[reportUnusedFunction]  # shared with legacy_spend_log_rows and rust_spend_log_rows
    block: object,
) -> str | None:
    if isinstance(block, str):
        return block
    block_type: Final = _block_type(block)
    if block_type in _TEXT_BLOCK_TYPES:
        text: Final = message_field(block, "text")
        return text if isinstance(text, str) else None
    return _OMITTED_BLOCK_PLACEHOLDERS.get(block_type) if block_type is not None else None


def _is_request_body_credential(key: str, value: object) -> bool:
    return isinstance(value, str) and _REQUEST_BODY_CREDENTIAL_MASKER.is_sensitive_key(key)


def _is_spend_log_content(parent: Mapping[str, object], key: str) -> bool:
    block_type: Final = parent.get("type")
    block_type_name: Final = block_type if isinstance(block_type, str) else None
    return (
        key in ("arguments", "logprobs")
        or (key == "input" and block_type_name in _TOOL_INPUT_BLOCK_TYPES)
        or (
            key in ("content", "output")
            and (block_type_name in _TOOL_OUTPUT_BLOCK_TYPES or parent.get("role") == "tool")
        )
    )


def _without_tokens(value: str) -> str:
    from litellm.proxy.proxy_server import general_settings_view

    if general_settings_view().get("redact_tokens_in_spend_logs") is not True:
        return value
    return redact_tokens(strip_null_bytes(value), REDACTED_BY_LITELM_STRING)


def _truncated_for_db(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    start_chars: Final = int(max_length * 0.35)
    end_chars: Final = int(max_length * 0.65)
    return (
        f"{value[:start_chars]}"
        f"... ({LITELLM_TRUNCATED_PAYLOAD_FIELD} skipped {len(value) - start_chars - end_chars} chars. "
        f"{LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE}) ..."
        f"{value[-min(end_chars, max_length - start_chars) :]}"
    )


def _compaction_summary_for_db(summary: str) -> str:
    return _truncated_for_db(
        _without_tokens(summary), max(_get_max_string_length_prompt_in_db(), MAX_STRING_LENGTH_COMPACTION_SUMMARY_IN_DB)
    )


def _bounded_compaction_summary(
    parent: Mapping[str, object], key: str, value: object, keep_compaction_blocks: bool
) -> str | None:
    if not keep_compaction_blocks or not isinstance(value, str) or parent.get("type") != "compaction":
        return None
    if key == "content":
        return _compaction_summary_for_db(value)
    summary: Final = compaction_summary(parent) if key == "encrypted_content" else None
    return None if summary is None else with_compaction_summary(value, _compaction_summary_for_db(summary))


def _sanitize_request_body_for_spend_logs_payload(
    request_body: Mapping[str, object],
    visited: set[int] | None = None,
    max_string_length_prompt_in_db: int | None = None,
    mask_credentials: bool = True,
    keep_compaction_blocks: bool = False,
) -> dict[str, object]:
    """
    Recursively sanitize request body to prevent logging large base64 strings or other large values.
    Truncates strings longer than MAX_STRING_LENGTH_PROMPT_IN_DB characters and handles nested dictionaries.

    At every nesting level, also strips keys listed in _SENSITIVE_REQUEST_BODY_KEYS (e.g. secret_fields,
    which holds raw HTTP headers including Authorization tokens), and replaces string values under keys
    SensitiveDataMasker classifies as credentials with REDACTED_BY_LITELM_STRING, except inside tool payloads
    and logprobs. When enabled, token-shaped substrings (e.g. a GitHub token in a tool result) are masked everywhere.
    """
    if visited is None:
        visited = set()
    if max_string_length_prompt_in_db is None:
        max_string_length_prompt_in_db = _get_max_string_length_prompt_in_db()

    # Get the object's memory address to track visited objects
    obj_id: Final = id(request_body)
    if obj_id in visited:
        return {}
    visited.add(obj_id)

    def _sanitize_value(value: object, mask_credentials: bool) -> object:
        if isinstance(value, Mapping):
            return _sanitize_request_body_for_spend_logs_payload(
                value, visited, max_string_length_prompt_in_db, mask_credentials, keep_compaction_blocks
            )
        elif isinstance(value, list):
            return [_sanitize_value(item, mask_credentials) for item in value]
        elif isinstance(value, str):
            return _truncated_for_db(_without_tokens(value), max_string_length_prompt_in_db)
        return value

    def _stored_value(key: str, value: object) -> object:
        if mask_credentials and _is_request_body_credential(key, value):
            return REDACTED_BY_LITELM_STRING
        bounded_summary: Final = _bounded_compaction_summary(request_body, key, value, keep_compaction_blocks)
        if bounded_summary is not None:
            return bounded_summary
        return _sanitize_value(value, mask_credentials and not _is_spend_log_content(request_body, key))

    return {k: _stored_value(k, v) for k, v in request_body.items() if k not in _SENSITIVE_REQUEST_BODY_KEYS}


# Quoted-key form: ``"input"`` / ``'messages'`` / ``"prompt"`` followed by
# ``:``. Covers JSON bodies and Python dict-reprs in provider error strings.
# ``prompt`` is included for ``/v1/completions``-style payloads where the user
# input lives under a top-level ``prompt`` key rather than ``messages``.
_ERROR_MESSAGE_PROMPT_LEAK_KEYS: Final = ("input", "messages", "prompt")


# Assignment-style keys: Pydantic v2 validation errors render the offending
# value as ``input_value=<repr>`` inside ``[type=..., input_value=...,
# input_type=...]``. The same prompt body that would appear under an
# ``"input"`` JSON key is echoed here as a Python repr, so we redact it
# under the same store_prompts_in_spend_logs gate.
_ERROR_MESSAGE_ASSIGN_LEAK_KEYS: Final = ("input_value",)


_SENSITIVE_KEY_START_PATTERN: Final = re.compile(
    r"(?:"
    r"['\"](?:" + "|".join(_ERROR_MESSAGE_PROMPT_LEAK_KEYS) + r")['\"]\s*:\s*"
    r"|"
    r"\b(?:" + "|".join(_ERROR_MESSAGE_ASSIGN_LEAK_KEYS) + r")\s*=\s*"
    r")"
)


def _scan_quoted_string_end(text: str, start: int, quote: str) -> int:
    """
    Given ``text[start] == quote`` (``'`` or ``"``), return the index just
    past the matching close quote, honoring backslash escapes. Returns
    ``-1`` if unterminated.
    """
    n: Final = len(text)
    i = start + 1
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == quote:
            return i + 1
        i += 1
    return -1


def _scan_balanced_value_end(text: str, start: int) -> int:
    """
    Given ``text[start]`` is ``[``, ``{``, ``'`` or ``"``, return the index
    just past the matching close, accounting for nested brackets and
    quoted strings (with escape sequences). Returns ``-1`` if the
    structure is unterminated.

    Implemented iteratively (no self-recursion): the bracket scanner
    inlines a quote-skip helper rather than re-entering itself, since
    JSON-style values cannot contain another bracket *as a first char*
    inside a quoted string — only the quote-skip case can occur.
    """
    n: Final = len(text)
    if start >= n:
        return -1
    first: Final = text[start]
    if first in ("'", '"'):
        return _scan_quoted_string_end(text, start, first)
    if first == "[":
        close = "]"
    elif first == "{":
        close = "}"
    else:
        return -1
    depth = 0
    i = start
    while i < n:
        c = text[i]
        if c in ("'", '"'):
            end = _scan_quoted_string_end(text, i, c)
            if end == -1:
                return -1
            i = end
            continue
        if c == first:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _redact_prompt_leaks_in_error_string(text: str) -> str:
    """
    Strip echoed request input from provider error strings.

    Provider validation errors (e.g. OpenAI ``RateLimitError`` carrying 178
    pydantic validation errors, each with its own ``'input': [...]`` field)
    embed the full request body in their message. When prompts must not be
    stored in spend logs, that echo is a back-door leak.

    Two leak shapes are handled:

    - Quoted-key form — ``"<key>": <value>`` where ``key`` is ``input``,
      ``messages`` or ``prompt`` (covers JSON bodies, Python dict-reprs,
      and ``/v1/completions`` payloads).
    - Assignment form — ``input_value=<value>`` from Pydantic v2 validation
      errors, which render the offending value as a Python repr inside
      ``[type=..., input_value=..., input_type=...]``.

    The value scan understands nested ``[]`` / ``{}`` and quoted strings,
    so multi-modal payloads (``'messages': [{'content': [{...}]}]``) and
    user text containing brackets (``"secret[123"``) are handled correctly.
    """
    if not text:
        return text
    redaction: Final = f'"{REDACTED_BY_LITELM_STRING}"'
    out: Final[list[str]] = []
    n: Final = len(text)
    pos = 0
    while pos < n:
        m = _SENSITIVE_KEY_START_PATTERN.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        out.append(text[pos : m.end()])
        v_start = m.end()
        if v_start >= n:
            break
        first = text[v_start]
        if first in ("[", "{", "'", '"'):
            v_end = _scan_balanced_value_end(text, v_start)
            if v_end == -1:
                # Unterminated value — redact through the rest of the string
                # so a malformed leak can't slip past.
                out.append(redaction)
                pos = n
                break
            out.append(redaction)
            pos = v_end
        else:
            # Unquoted scalar (number, null, bare identifier) — not a leak
            # carrier, leave intact and resume after the key match.
            pos = v_start
    return "".join(out)


def _sanitize_guardrail_information_for_spend_logs(
    guardrail_information: list[StandardLoggingGuardrailInformation] | None,
    metadata: Mapping[str, object] | None = None,
) -> list[StandardLoggingGuardrailInformation] | None:
    """
    When ``store_prompts_in_spend_logs`` is False, redact prompt-carrying fields
    (``guardrail_request``, ``guardrail_response``, ``match_details``,
    ``classification``) before they land in ``LiteLLM_SpendLogs.metadata``.

    Guardrail hooks may echo the LLM request payload back into
    ``guardrail_response``, and two first-party hooks
    (``block_code_execution``, ``litellm_content_filter``) inline user-prompt
    substrings into ``match_details`` / ``classification`` too, so the flag
    must cover all four fields. Every other typed field on the entry (name,
    provider, mode, status, timings, action, violation_categories, risk_score,
    masked_entity_count, ...) is preserved so guardrail dashboards keep
    working.

    ``guardrail_information`` is typed ``Optional[List[...]]`` but at least
    one writer (``xecguard``) assigns a bare dict, so normalize to a list
    here to match OTEL's defensive read pattern; otherwise iteration would
    yield the dict's keys and crash the whole spend-log write.
    """
    if guardrail_information is None or stores_full_prompts_in_spend_logs(metadata):
        return guardrail_information
    entries: Final = [guardrail_information] if isinstance(guardrail_information, dict) else guardrail_information
    return [_redact_prompt_fields_in_guardrail_entry(entry) for entry in entries if isinstance(entry, dict)]


def _redact_prompt_fields_in_guardrail_entry(
    entry: StandardLoggingGuardrailInformation,
) -> StandardLoggingGuardrailInformation:
    """
    Replace prompt-carrying fields with the redaction marker. Purely numeric
    compression stats inside ``guardrail_response`` (e.g. Headroom's
    ``tokens_saved``) cannot carry prompt content, so they are preserved as a
    stats-only dict; spend aggregation reads them via
    ``extract_compression_saved_tokens``.
    """
    preserved_stats: Final = numeric_compression_stats(entry)
    redacted: Final[StandardLoggingGuardrailInformation] = {
        **entry,
        **{key: REDACTED_BY_LITELM_STRING for key in PROMPT_CARRYING_GUARDRAIL_FIELDS if key in entry},
    }
    if preserved_stats is None:
        return redacted
    return {**redacted, "guardrail_response": preserved_stats}


def sanitize_error_information_for_spend_logs(
    error_information: StandardLoggingPayloadErrorInformation | None,
    original_exception: BaseException | None = None,
    metadata: Mapping[str, object] | None = None,
) -> StandardLoggingPayloadErrorInformation | None:
    """
    Sanitize ``error_information`` before it lands in ``LiteLLM_SpendLogs.metadata``.

    Provider errors are stored verbatim via ``str(original_exception)``; those
    strings can echo the full request body, producing multi-megabyte spend-log
    rows.

    - Always: cap ``error_message`` and ``traceback`` with the existing
      ``MAX_STRING_LENGTH_PROMPT_IN_DB`` DB-storage safeguard.
    - When ``store_prompts_in_spend_logs`` is False: additionally redact
      ``'input'`` / ``'messages'`` / ``'prompt'`` values *and* Pydantic v2
      ``input_value=...`` assignments inside both ``error_message`` and
      ``traceback`` so prompts cannot leak through either field.

    Scoped to the spend-log path — OTEL/Datadog/etc. callbacks still receive
    the untruncated error per ``LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE``.
    """
    if error_information is None:
        return None

    persisted: Final = (
        {**error_information, "error_message": original_exception.spend_log_error_message}
        if isinstance(original_exception, ProxyModelNotFoundError)
        else error_information
    )
    sanitized = cast(dict, {**persisted})

    if not stores_full_prompts_in_spend_logs(metadata):
        for field in ("error_message", "traceback"):
            value = sanitized.get(field)
            if isinstance(value, str):
                sanitized[field] = _redact_prompt_leaks_in_error_string(value)

    sanitized = _sanitize_request_body_for_spend_logs_payload(sanitized)
    return cast(
        StandardLoggingPayloadErrorInformation,
        _error_information_without_ip_addresses(_OBJECT_MAPPING.validate_python(sanitized))
        if requester_ip_address_logging_disabled()
        else sanitized,
    )


_sanitize_error_information_for_spend_logs: Final = sanitize_error_information_for_spend_logs


_REQUEST_METADATA_KEYS: Final = ("metadata", "litellm_metadata")
_OBJECT_MAPPING: Final = TypeAdapter(Mapping[object, object])
_OBJECT_LIST: Final = TypeAdapter(list[object])
_IP_ADDRESS_CHARACTERS: Final = "0123456789abcdefABCDEF.:"
_MAX_IP_ADDRESS_LENGTH: Final = len("ffff:ffff:ffff:ffff:ffff:ffff:255.255.255.255")
_LONG_HEX_GROUP: Final = re.compile(r"[0-9A-Fa-f]{5,}")
_TRUNCATION_MARKER_GROUP: Final = re.compile(f"({_TRUNCATION_MARKER.pattern})")
_IPV4_ADDRESS: Final = re.compile(r"(?<![0-9A-Za-z])(?:\d{1,3}\.){3}\d{1,3}(?![0-9A-Za-z])")
_IPV6_ADDRESS_CANDIDATE: Final = re.compile(
    r"(?:(?<![0-9A-Za-z:])|(?<=\w:))(?:[0-9A-Fa-f]{0,4}:){2,9}[0-9A-Fa-f]{0,4}(?![0-9A-Za-z:])"
)


def _masked_ipv6_address(candidate: re.Match[str]) -> str:
    """Masks the candidate when it, or it without a leading label or a trailing port group, is an IPv6 address."""
    text: Final = candidate.group(0)
    unlabelled: Final = text.partition(":")[2]
    for address in (text, text.rpartition(":")[0], unlabelled, unlabelled.rpartition(":")[0]):
        try:
            ipaddress.IPv6Address(address)
        except ValueError:
            continue
        return REDACTED_BY_LITELM_STRING
    return text


def _without_ip_addresses(text: str) -> str:
    return _IPV6_ADDRESS_CANDIDATE.sub(_masked_ipv6_address, _IPV4_ADDRESS.sub(REDACTED_BY_LITELM_STRING, text))


def _is_ip_address_piece(piece: str) -> bool:
    return any(char.isdigit() or char == ":" for char in piece)


def _without_ip_piece_before_cut(text: str) -> str:
    run: Final = text[len(text.rstrip(_IP_ADDRESS_CHARACTERS)) :]
    long_groups: Final = list(_LONG_HEX_GROUP.finditer(run))
    piece: Final = run[long_groups[-1].end() if long_groups else 0 :][-_MAX_IP_ADDRESS_LENGTH:].lstrip(".:")
    return text[: len(text) - len(piece)] + REDACTED_BY_LITELM_STRING if _is_ip_address_piece(piece) else text


def _without_ip_piece_after_cut(text: str) -> str:
    run: Final = text[: len(text) - len(text.lstrip(_IP_ADDRESS_CHARACTERS))]
    long_group: Final = _LONG_HEX_GROUP.search(run)
    piece: Final = run[: long_group.start() if long_group else len(run)][:_MAX_IP_ADDRESS_LENGTH].rstrip(".:")
    return REDACTED_BY_LITELM_STRING + text[len(piece) :] if _is_ip_address_piece(piece) else text


def _without_ip_pieces_beside_cuts(text: str, *, after_cut: bool, before_cut: bool) -> str:
    after: Final = _without_ip_piece_after_cut(text) if after_cut else text
    return _without_ip_piece_before_cut(after) if before_cut else after


def _masked_ip_text(text: str) -> str:
    """The text with every IP address masked, plus the run of short hex, digit, dot and colon groups beside each
    truncation marker when that run holds a digit or a colon."""
    parts: Final = _TRUNCATION_MARKER_GROUP.split(_without_ip_addresses(text))
    last: Final = len(parts) - 1
    return "".join(
        part if index % 2 else _without_ip_pieces_beside_cuts(part, after_cut=index > 0, before_cut=index < last)
        for index, part in enumerate(parts)
    )


def _error_information_without_ip_addresses(error_information: Mapping[object, object]) -> Mapping[object, object]:
    """The error information with every IP address in its error message and traceback masked."""
    return {
        key: _masked_ip_text(value) if key in _SCRUBBED_ERROR_TEXT_FIELDS and isinstance(value, str) else value
        for key, value in error_information.items()
    }


def _get_vector_store_request_for_spend_logs_payload(
    vector_store_request_metadata: list[StandardLoggingVectorStoreRequest] | None,
    metadata: Mapping[str, object] | None = None,
) -> list[StandardLoggingVectorStoreRequest] | None:
    """
    If user does not want to store prompts and responses, then remove the content from the vector store request metadata
    """
    if stores_full_prompts_in_spend_logs(metadata):
        return vector_store_request_metadata

    if vector_store_request_metadata is None:
        return None
    return [_redacted_vector_store_request(request) for request in vector_store_request_metadata]


def _redacted_vector_store_content(content: VectorStoreResultContent) -> VectorStoreResultContent:
    return {**content, "text": REDACTED_BY_LITELM_STRING} if "text" in content else content


def _redacted_vector_store_result(result: VectorStoreSearchResult) -> VectorStoreSearchResult:
    return {**result, "content": [_redacted_vector_store_content(content) for content in result.get("content") or ()]}


def _redacted_vector_store_request(request: StandardLoggingVectorStoreRequest) -> StandardLoggingVectorStoreRequest:
    """A redacted copy, so the shared logging payload other callbacks read keeps its search results."""
    response: Final = request.get("vector_store_search_response") or VectorStoreSearchResponse()
    return {
        **request,
        "vector_store_search_response": {
            **response,
            "data": [_redacted_vector_store_result(result) for result in response.get("data") or ()],
        },
    }


def _disables_prompt_storage(tenant_metadata: object) -> bool:
    settings: Final = as_str_mapping(tenant_metadata)
    value: Final = settings.get("store_prompts_in_spend_logs") if settings is not None else None
    return value is False or (isinstance(value, str) and str_to_bool(value) is False)


def should_store_prompts_for_call(kwargs: Mapping[str, object]) -> bool:
    return should_store_prompts_and_responses_in_spend_logs(
        _SPEND_LOGS_PRIVACY_METADATA_ADAPTER.validate_python(get_litellm_metadata_from_kwargs(dict(kwargs)))
    )


def should_store_prompts_and_responses_in_spend_logs(metadata: Mapping[str, object] | None = None) -> bool:
    from litellm.proxy.proxy_server import general_settings
    from litellm.secret_managers.main import get_secret_bool

    if metadata is not None and any(
        _disables_prompt_storage(metadata.get(field)) for field in _TENANT_PROMPT_STORAGE_FIELDS
    ):
        return False

    # Check general_settings (from DB or proxy_config.yaml)
    store_prompts_value: Final = general_settings.get("store_prompts_in_spend_logs")

    # Normalize case: handle True/true/TRUE, False/false/FALSE, None/null
    if store_prompts_value is True:
        return True
    elif isinstance(store_prompts_value, str):
        # Case-insensitive string comparison
        if store_prompts_value.lower() == "true":
            return True

    # Also check environment variable
    return get_secret_bool("STORE_PROMPTS_IN_SPEND_LOGS") is True


_SPEND_LOGS_METADATA_FIELDS_ADAPTER: Final[TypeAdapter[SpendLogsMetadataFields | None]] = TypeAdapter(
    SpendLogsMetadataFields | None
)
_SPEND_LOGS_METADATA_ADAPTER: Final = TypeAdapter(dict[str, JsonValue])


def configured_spend_logs_metadata_fields() -> SpendLogsMetadataFields | None:
    from litellm.proxy.proxy_server import general_settings_view

    try:
        return _SPEND_LOGS_METADATA_FIELDS_ADAPTER.validate_python(
            general_settings_view().get("spend_logs_metadata_fields")
        )
    except ValidationError as e:
        verbose_proxy_logger.error("Ignoring invalid general_settings.spend_logs_metadata_fields: %s", e)
        return None


_OPTIONAL_BOOL_ADAPTER: Final[TypeAdapter[bool | None]] = TypeAdapter(bool | None)


def requester_ip_address_logging_disabled() -> bool:
    """True when `general_settings.disable_requester_ip_address_logging` holds a value the settings API accepts as true."""
    from litellm.proxy.proxy_server import general_settings_view

    try:
        return (
            _OPTIONAL_BOOL_ADAPTER.validate_python(general_settings_view().get("disable_requester_ip_address_logging"))
            is True
        )
    except ValidationError:
        return False


def spend_log_row_with_retained_metadata(
    row: Mapping[str, object], fields: SpendLogsMetadataFields | None
) -> Mapping[str, object]:
    metadata_json: Final = row.get("metadata")
    if fields is None or not isinstance(metadata_json, str):
        return row
    metadata: Final = _SPEND_LOGS_METADATA_ADAPTER.validate_json(metadata_json)
    retained: Final = {name: value for name, value in metadata.items() if fields.keeps(name)}
    return {**row, "metadata": safe_dumps(retained, non_finite_as_null=True)}
