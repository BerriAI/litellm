import copy
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal, Optional

import litellm
from litellm import get_secret
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy._types import CommonProxyErrors, LiteLLMPromptInjectionParams
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.proxy.types_utils.utils import get_instance_fn
from litellm.types.utils import (
    StandardLoggingGuardrailInformation,
    StandardLoggingPayload,
)

_CALLBACK_VAR_MASKER = SensitiveDataMasker()
# Compound names that are credential-bearing but don't contain any of the
# default sensitive segments (so SensitiveDataMasker won't flag them).
_EXTRA_SENSITIVE_CALLBACK_KEYS = {"gcs_path_service_account"}
# Sentinel prefix on encrypted callback_var values. Lets us detect
# already-encrypted input cheaply (no decrypt-attempt round trip) and
# avoid double-encrypting if `LITELLM_SALT_KEY` is rotated between writes.
_CALLBACK_VAR_ENCRYPTED_PREFIX = "litellm_enc::"

blue_color_code = "\033[94m"
reset_color_code = "\033[0m"

TRUSTED_PILLAR_RESPONSE_HEADERS_METADATA_KEY = "_pillar_response_headers_trusted"

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging


def initialize_callbacks_on_proxy(  # noqa: PLR0915
    value: Any,
    premium_user: bool,
    config_file_path: str,
    litellm_settings: dict,
    callback_specific_params: dict = {},
):
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.logging_callback_manager import (
        LoggingCallbackManager,
    )
    from litellm.proxy.proxy_server import prisma_client

    verbose_proxy_logger.debug(
        f"{blue_color_code}initializing callbacks={value} on proxy{reset_color_code}"
    )
    if isinstance(value, list):
        imported_list: List[Any] = []
        for callback in value:  # ["presidio", <my-custom-callback>]
            if isinstance(callback, str) and callback == "compression_interception":
                from litellm.integrations.compression_interception.handler import (
                    CompressionInterceptionLogger,
                )

                compression_interception_obj = (
                    CompressionInterceptionLogger.initialize_from_proxy_config(
                        litellm_settings=litellm_settings,
                        callback_specific_params=callback_specific_params,
                    )
                )
                imported_list.append(compression_interception_obj)
                continue

            # check if callback is a custom logger compatible callback
            if isinstance(callback, str):
                callback = LoggingCallbackManager._add_custom_callback_generic_api_str(
                    callback
                )
            if (
                isinstance(callback, str)
                and callback in litellm._known_custom_logger_compatible_callbacks
            ):
                imported_list.append(callback)
            elif isinstance(callback, str) and callback == "presidio":
                from litellm.proxy.guardrails.guardrail_hooks.presidio import (
                    _OPTIONAL_PresidioPIIMasking,
                )

                presidio_logging_only: Optional[bool] = litellm_settings.get(
                    "presidio_logging_only", None
                )
                if presidio_logging_only is not None:
                    presidio_logging_only = bool(
                        presidio_logging_only
                    )  # validate boolean given

                _presidio_params = {}
                if "presidio" in callback_specific_params and isinstance(
                    callback_specific_params["presidio"], dict
                ):
                    _presidio_params = callback_specific_params["presidio"]

                params: Dict[str, Any] = {
                    "logging_only": presidio_logging_only,
                    **_presidio_params,
                }
                pii_masking_object = _OPTIONAL_PresidioPIIMasking(**params)
                imported_list.append(pii_masking_object)
            elif isinstance(callback, str) and callback == "llamaguard_moderations":
                try:
                    from litellm_enterprise.enterprise_callbacks.llama_guard import (
                        _ENTERPRISE_LlamaGuard,
                    )
                except ImportError:
                    raise Exception(
                        "MissingTrying to use Llama Guard"
                        + CommonProxyErrors.missing_enterprise_package.value
                    )

                if premium_user is not True:
                    raise Exception(
                        "Trying to use Llama Guard"
                        + CommonProxyErrors.not_premium_user.value
                    )

                llama_guard_object = _ENTERPRISE_LlamaGuard()
                imported_list.append(llama_guard_object)
            elif isinstance(callback, str) and callback == "hide_secrets":
                try:
                    from litellm_enterprise.enterprise_callbacks.secret_detection import (
                        _ENTERPRISE_SecretDetection,
                    )
                except ImportError:
                    raise Exception(
                        "Trying to use Secret Detection"
                        + CommonProxyErrors.missing_enterprise_package.value
                    )

                if premium_user is not True:
                    raise Exception(
                        "Trying to use secret hiding"
                        + CommonProxyErrors.not_premium_user.value
                    )

                _secret_detection_object = _ENTERPRISE_SecretDetection()
                imported_list.append(_secret_detection_object)
            elif isinstance(callback, str) and callback == "openai_moderations":
                try:
                    from enterprise.enterprise_hooks.openai_moderation import (
                        _ENTERPRISE_OpenAI_Moderation,
                    )
                except ImportError:
                    raise Exception(
                        "Trying to use OpenAI Moderations Check,"
                        + CommonProxyErrors.missing_enterprise_package_docker.value
                    )

                if premium_user is not True:
                    raise Exception(
                        "Trying to use OpenAI Moderations Check"
                        + CommonProxyErrors.not_premium_user.value
                    )

                openai_moderations_object = _ENTERPRISE_OpenAI_Moderation()
                imported_list.append(openai_moderations_object)
            elif isinstance(callback, str) and callback == "lakera_prompt_injection":
                from litellm.proxy.guardrails.guardrail_hooks.lakera_ai import (
                    lakeraAI_Moderation,
                )

                init_params = {}
                if "lakera_prompt_injection" in callback_specific_params:
                    init_params = callback_specific_params["lakera_prompt_injection"]
                lakera_moderations_object = lakeraAI_Moderation(**init_params)
                imported_list.append(lakera_moderations_object)
            elif isinstance(callback, str) and callback == "aporia_prompt_injection":
                from litellm.proxy.guardrails.guardrail_hooks.aporia_ai.aporia_ai import (
                    AporiaGuardrail,
                )

                aporia_guardrail_object = AporiaGuardrail()
                imported_list.append(aporia_guardrail_object)
            elif isinstance(callback, str) and callback == "google_text_moderation":
                try:
                    from enterprise.enterprise_hooks.google_text_moderation import (
                        _ENTERPRISE_GoogleTextModeration,
                    )
                except ImportError:
                    raise Exception(
                        "Trying to use Google Text Moderation,"
                        + CommonProxyErrors.missing_enterprise_package_docker.value
                    )

                if premium_user is not True:
                    raise Exception(
                        "Trying to use Google Text Moderation"
                        + CommonProxyErrors.not_premium_user.value
                    )

                google_text_moderation_obj = _ENTERPRISE_GoogleTextModeration()
                imported_list.append(google_text_moderation_obj)
            elif isinstance(callback, str) and callback == "llmguard_moderations":
                try:
                    from litellm_enterprise.enterprise_callbacks.llm_guard import (
                        _ENTERPRISE_LLMGuard,
                    )
                except ImportError:
                    raise Exception(
                        "Trying to use Llm Guard"
                        + CommonProxyErrors.missing_enterprise_package.value
                    )

                if premium_user is not True:
                    raise Exception(
                        "Trying to use Llm Guard"
                        + CommonProxyErrors.not_premium_user.value
                    )

                llm_guard_moderation_obj = _ENTERPRISE_LLMGuard()
                imported_list.append(llm_guard_moderation_obj)
            elif isinstance(callback, str) and callback == "blocked_user_check":
                try:
                    from enterprise.enterprise_hooks.blocked_user_list import (
                        _ENTERPRISE_BlockedUserList,
                    )
                except ImportError:
                    raise Exception(
                        "Trying to use Blocked User List"
                        + CommonProxyErrors.missing_enterprise_package_docker.value
                    )

                if premium_user is not True:
                    raise Exception(
                        "Trying to use ENTERPRISE BlockedUser"
                        + CommonProxyErrors.not_premium_user.value
                    )

                blocked_user_list = _ENTERPRISE_BlockedUserList(
                    prisma_client=prisma_client
                )
                imported_list.append(blocked_user_list)
            elif isinstance(callback, str) and callback == "banned_keywords":
                try:
                    from enterprise.enterprise_hooks.banned_keywords import (
                        _ENTERPRISE_BannedKeywords,
                    )
                except ImportError:
                    raise Exception(
                        "Trying to use Banned Keywords"
                        + CommonProxyErrors.missing_enterprise_package_docker.value
                    )

                if premium_user is not True:
                    raise Exception(
                        "Trying to use ENTERPRISE BannedKeyword"
                        + CommonProxyErrors.not_premium_user.value
                    )

                banned_keywords_obj = _ENTERPRISE_BannedKeywords()
                imported_list.append(banned_keywords_obj)
            elif isinstance(callback, str) and callback == "detect_prompt_injection":
                from litellm.proxy.hooks.prompt_injection_detection import (
                    _OPTIONAL_PromptInjectionDetection,
                )

                prompt_injection_params = None
                if "prompt_injection_params" in litellm_settings:
                    prompt_injection_params_in_config = litellm_settings[
                        "prompt_injection_params"
                    ]
                    prompt_injection_params = LiteLLMPromptInjectionParams(
                        **prompt_injection_params_in_config
                    )

                prompt_injection_detection_obj = _OPTIONAL_PromptInjectionDetection(
                    prompt_injection_params=prompt_injection_params,
                )
                imported_list.append(prompt_injection_detection_obj)
            elif isinstance(callback, str) and callback == "batch_redis_requests":
                from litellm.proxy.hooks.batch_redis_get import (
                    _PROXY_BatchRedisRequests,
                )

                batch_redis_obj = _PROXY_BatchRedisRequests()
                imported_list.append(batch_redis_obj)
            elif isinstance(callback, str) and callback == "azure_content_safety":
                from litellm.proxy.hooks.azure_content_safety import (
                    _PROXY_AzureContentSafety,
                )

                azure_content_safety_params = litellm_settings[
                    "azure_content_safety_params"
                ]
                for k, v in azure_content_safety_params.items():
                    if (
                        v is not None
                        and isinstance(v, str)
                        and v.startswith("os.environ/")
                    ):
                        azure_content_safety_params[k] = get_secret(v)

                azure_content_safety_obj = _PROXY_AzureContentSafety(
                    **azure_content_safety_params,
                )
                imported_list.append(azure_content_safety_obj)
            elif isinstance(callback, str) and callback == "websearch_interception":
                from litellm.integrations.websearch_interception.handler import (
                    WebSearchInterceptionLogger,
                )

                websearch_interception_obj = (
                    WebSearchInterceptionLogger.initialize_from_proxy_config(
                        litellm_settings=litellm_settings,
                        callback_specific_params=callback_specific_params,
                    )
                )
                imported_list.append(websearch_interception_obj)
            elif isinstance(callback, str) and callback == "datadog_cost_management":
                from litellm.integrations.datadog.datadog_cost_management import (
                    DatadogCostManagementLogger,
                )

                init_params = {}
                if (
                    "datadog_cost_management" in callback_specific_params
                    and isinstance(
                        callback_specific_params["datadog_cost_management"], dict
                    )
                ):
                    init_params = callback_specific_params["datadog_cost_management"]
                datadog_cost_management_obj = DatadogCostManagementLogger(**init_params)
                imported_list.append(datadog_cost_management_obj)
            elif isinstance(callback, CustomLogger):
                imported_list.append(callback)
            else:
                verbose_proxy_logger.debug(
                    f"{blue_color_code} attempting to import custom calback={callback} {reset_color_code}"
                )
                imported_list.append(
                    get_instance_fn(
                        value=callback,
                        config_file_path=config_file_path,
                    )
                )
        if isinstance(litellm.callbacks, list):
            litellm.callbacks.extend(imported_list)
        else:
            litellm.callbacks = imported_list  # type: ignore

        if "prometheus" in value:
            from litellm.integrations.prometheus import PrometheusLogger

            PrometheusLogger._mount_metrics_endpoint()
    else:
        litellm.callbacks = [
            get_instance_fn(
                value=value,
                config_file_path=config_file_path,
            )
        ]
    verbose_proxy_logger.debug(
        f"{blue_color_code} Initialized Callbacks - {litellm.callbacks} {reset_color_code}"
    )


def get_model_group_from_litellm_kwargs(kwargs: dict) -> Optional[str]:
    _litellm_params = kwargs.get("litellm_params", None) or {}
    _metadata = (
        _litellm_params.get(get_metadata_variable_name_from_kwargs(kwargs)) or {}
    )
    _model_group = _metadata.get("model_group", None)
    if _model_group is not None:
        return _model_group

    return None


def get_model_group_from_request_data(data: dict) -> Optional[str]:
    _metadata = data.get("metadata", None) or {}
    _model_group = _metadata.get("model_group", None)
    if _model_group is not None:
        return _model_group

    return None


def get_remaining_tokens_and_requests_from_request_data(data: Dict) -> Dict[str, str]:
    """
    Helper function to return x-litellm-key-remaining-tokens-{model_group} and x-litellm-key-remaining-requests-{model_group}

    Returns {} when api_key + model rpm/tpm limit is not set

    """
    headers = {}
    _metadata = data.get("metadata", None) or {}
    model_group = get_model_group_from_request_data(data)

    # The h11 package considers "/" or ":" invalid and raise a LocalProtocolError
    h11_model_group_name = (
        model_group.replace("/", "-").replace(":", "-") if model_group else None
    )

    # Remaining Requests
    remaining_requests_variable_name = f"litellm-key-remaining-requests-{model_group}"
    remaining_requests = _metadata.get(remaining_requests_variable_name, None)
    if remaining_requests:
        headers[f"x-litellm-key-remaining-requests-{h11_model_group_name}"] = (
            remaining_requests
        )

    # Remaining Tokens
    remaining_tokens_variable_name = f"litellm-key-remaining-tokens-{model_group}"
    remaining_tokens = _metadata.get(remaining_tokens_variable_name, None)
    if remaining_tokens:
        headers[f"x-litellm-key-remaining-tokens-{h11_model_group_name}"] = (
            remaining_tokens
        )

    return headers


def get_logging_caching_headers(request_data: Dict) -> Optional[Dict]:
    _metadata: Dict = {}
    metadata_bucket = request_data.get("metadata")
    litellm_metadata_bucket = request_data.get("litellm_metadata")
    if isinstance(metadata_bucket, dict):
        _metadata.update(metadata_bucket)
    if isinstance(litellm_metadata_bucket, dict):
        # Batch/file routes store proxy tracking in litellm_metadata while
        # user-facing metadata stays in metadata; merge both for headers.
        _metadata.update(litellm_metadata_bucket)
    headers = {}
    if "applied_guardrails" in _metadata:
        headers["x-litellm-applied-guardrails"] = ",".join(
            _metadata["applied_guardrails"]
        )

    if "applied_policies" in _metadata:
        headers["x-litellm-applied-policies"] = ",".join(_metadata["applied_policies"])

    if "policy_sources" in _metadata:
        sources = _metadata["policy_sources"]
        if isinstance(sources, dict) and sources:
            # Use ';' as delimiter — matched_via reasons may contain commas
            headers["x-litellm-policy-sources"] = "; ".join(
                f"{name}={reason}" for name, reason in sources.items()
            )

    if "semantic-similarity" in _metadata:
        headers["x-litellm-semantic-similarity"] = str(_metadata["semantic-similarity"])

    is_trusted_pillar_metadata = (
        _metadata.get(TRUSTED_PILLAR_RESPONSE_HEADERS_METADATA_KEY) is True
    )
    pillar_headers = _metadata.get("pillar_response_headers")
    if is_trusted_pillar_metadata and isinstance(pillar_headers, dict):
        headers.update(
            {
                key: str(value)
                for key, value in pillar_headers.items()
                if isinstance(key, str) and key.lower().startswith("x-pillar-")
            }
        )
    elif is_trusted_pillar_metadata and "pillar_flagged" in _metadata:
        headers["x-pillar-flagged"] = str(_metadata["pillar_flagged"]).lower()

    return headers


def get_metadata_variable_name_from_kwargs(
    kwargs: dict,
) -> Literal["metadata", "litellm_metadata"]:
    """
    Helper to return what the "metadata" field should be called in the request data

    - New endpoints return `litellm_metadata`
    - Old endpoints return `metadata`

    Context:
    - LiteLLM used `metadata` as an internal field for storing metadata
    - OpenAI then started using this field for their metadata
    - LiteLLM is now moving to using `litellm_metadata` for our metadata
    """
    return "litellm_metadata" if "litellm_metadata" in kwargs else "metadata"


LITELLM_PROXY_INTERNAL_METADATA_KEYS = frozenset(
    {
        "applied_policies",
        "applied_guardrails",
        "policy_sources",
        "guardrails",
        "guardrail_config",
        "_guardrail_pipelines",
        "_pipeline_managed_guardrails",
        "disable_global_guardrails",
        "disable_global_guardrail",
        "opted_out_global_guardrails",
        "pillar_response_headers",
        "_pillar_response_headers_trusted",
        "pillar_flagged",
        "pillar_scanners",
        "pillar_evidence",
        "pillar_evidence_truncated",
        "pillar_session_id_response",
        "standard_logging_object",
        "proxy_server_request",
        "secret_fields",
    }
)


def _get_or_create_proxy_metadata_bucket(
    request_data: Dict,
) -> tuple[Literal["metadata", "litellm_metadata"], dict]:
    """
    Return the proxy-internal metadata bucket for this request.

    Batch/file routes store proxy state in ``litellm_metadata`` so the OpenAI
    ``metadata`` field can remain provider-safe (string values only).
    """
    metadata_key = get_metadata_variable_name_from_kwargs(request_data)
    metadata_bucket = request_data.get(metadata_key)
    if not isinstance(metadata_bucket, dict):
        metadata_bucket = {}
        request_data[metadata_key] = metadata_bucket
    return metadata_key, metadata_bucket


def sanitize_openai_provider_metadata(
    metadata: Optional[Dict[str, Any]],
) -> Optional[Dict[str, str]]:
    """
    Keep only provider-safe OpenAI metadata entries (string keys -> string values).

    Strips LiteLLM proxy-internal tracking fields that must not be forwarded to
    OpenAI batch/file APIs.
    """
    if not metadata:
        return metadata
    sanitized: Dict[str, str] = {}
    for key, value in metadata.items():
        if key in LITELLM_PROXY_INTERNAL_METADATA_KEYS:
            continue
        if isinstance(value, str):
            sanitized[key] = value
        else:
            verbose_proxy_logger.debug(
                "sanitize_openai_provider_metadata: dropping key %r with non-string value of type %s",
                key,
                type(value).__name__,
            )
    return sanitized or None


def add_guardrail_to_applied_guardrails_header(
    request_data: Dict, guardrail_name: Optional[str]
):
    if guardrail_name is None:
        return
    _, _metadata = _get_or_create_proxy_metadata_bucket(request_data)
    if "applied_guardrails" in _metadata:
        if guardrail_name not in _metadata["applied_guardrails"]:
            _metadata["applied_guardrails"].append(guardrail_name)
    else:
        _metadata["applied_guardrails"] = [guardrail_name]


def add_policy_to_applied_policies_header(
    request_data: Dict, policy_name: Optional[str]
):
    """
    Add a policy name to the applied_policies list in request metadata.

    This is used to track which policies were applied to a request,
    similar to how applied_guardrails tracks guardrails.
    """
    if policy_name is None:
        return
    _, _metadata = _get_or_create_proxy_metadata_bucket(request_data)
    if "applied_policies" in _metadata:
        if policy_name not in _metadata["applied_policies"]:
            _metadata["applied_policies"].append(policy_name)
    else:
        _metadata["applied_policies"] = [policy_name]


def add_policy_sources_to_metadata(request_data: Dict, policy_sources: Dict[str, str]):
    """
    Store policy match reasons in metadata for x-litellm-policy-sources header.

    Args:
        request_data: The request data dict
        policy_sources: Map of policy_name -> matched_via reason
    """
    if not policy_sources:
        return
    _, _metadata = _get_or_create_proxy_metadata_bucket(request_data)
    existing = _metadata.get("policy_sources", {})
    if not isinstance(existing, dict):
        existing = {}
    existing.update(policy_sources)
    _metadata["policy_sources"] = existing


def add_guardrail_response_to_standard_logging_object(
    litellm_logging_obj: Optional["LiteLLMLogging"],
    guardrail_response: StandardLoggingGuardrailInformation,
):
    if litellm_logging_obj is None:
        return
    standard_logging_object: Optional[StandardLoggingPayload] = (
        litellm_logging_obj.model_call_details.get("standard_logging_object")
    )
    if standard_logging_object is None:
        return
    guardrail_information = standard_logging_object.get("guardrail_information", [])
    if guardrail_information is None:
        guardrail_information = []
    guardrail_information.append(guardrail_response)
    standard_logging_object["guardrail_information"] = guardrail_information

    return standard_logging_object


def process_callback(
    _callback: str, callback_type: str, environment_variables: dict
) -> dict:
    """Process a single callback and return its data with environment variables"""
    env_vars = CustomLogger.get_callback_env_vars(_callback)

    env_vars_dict: dict[str, str | None] = {}
    for _var in env_vars:
        env_variable = environment_variables.get(_var, None)
        if env_variable is None:
            env_vars_dict[_var] = None
        else:
            env_vars_dict[_var] = env_variable

    return {"name": _callback, "variables": env_vars_dict, "type": callback_type}


def normalize_callback_names(callbacks: Iterable[Any]) -> List[Any]:
    if callbacks is None:
        return []
    return [c.lower() if isinstance(c, str) else c for c in callbacks]


def encrypt_callback_vars(metadata: Any) -> Any:
    """Return a deep copy of metadata with callback_vars values encrypted at rest.

    Idempotent: a value that already decrypts cleanly is left unchanged so
    round-trips through edit forms don't double-encrypt.
    """
    return _transform_callback_vars(metadata, _encrypt_if_plaintext)


def decrypt_callback_vars(metadata: Any) -> Any:
    """Return a deep copy of metadata with callback_vars values decrypted.

    Legacy plaintext rows pass through unchanged (decrypt failure → original).
    """
    return _transform_callback_vars(metadata, _decrypt_or_passthrough)


def _transform_callback_vars(
    metadata: Any, transform: Callable[[str, Any], Any]
) -> Any:
    if not isinstance(metadata, dict):
        return metadata
    out = copy.deepcopy(metadata)
    logging_entries = out.get("logging")
    if isinstance(logging_entries, list):
        for entry in logging_entries:
            if isinstance(entry, dict) and isinstance(entry.get("callback_vars"), dict):
                entry["callback_vars"] = {
                    k: transform(k, v) for k, v in entry["callback_vars"].items()
                }
    callback_settings = out.get("callback_settings")
    if isinstance(callback_settings, dict) and isinstance(
        callback_settings.get("callback_vars"), dict
    ):
        callback_settings["callback_vars"] = {
            k: transform(k, v) for k, v in callback_settings["callback_vars"].items()
        }
    return out


def _is_sensitive_callback_var(key: str) -> bool:
    """Match codebase precedent: only credential-bearing fields get encrypted;
    routing/identifier fields (host, base_url, project, region) stay plain."""
    if key in _EXTRA_SENSITIVE_CALLBACK_KEYS:
        return True
    return _CALLBACK_VAR_MASKER.is_sensitive_key(key)


def _encrypt_if_plaintext(key: str, value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if not _is_sensitive_callback_var(key):
        return value
    if value.startswith(_CALLBACK_VAR_ENCRYPTED_PREFIX):
        # Already encrypted — round-tripping ciphertext (e.g. UI Edit Settings
        # save without changing the field) must not double-encrypt. Cheap
        # prefix check is robust under salt-key rotation; a decrypt-based
        # idempotency check would mis-classify K1-encrypted blobs as
        # plaintext under K2 and wrap them a second time.
        return value
    try:
        return _CALLBACK_VAR_ENCRYPTED_PREFIX + encrypt_value_helper(value)
    except Exception:
        # No salt key / master key configured — leave the value as-is rather
        # than crash the write. Dev environments without LITELLM_SALT_KEY hit
        # this path; production always has a master key so encryption proceeds.
        return value


def _decrypt_or_passthrough(key: str, value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if not value.startswith(_CALLBACK_VAR_ENCRYPTED_PREFIX):
        # Legacy plaintext rows or non-credential fields — return as-is.
        return value
    inner = value[len(_CALLBACK_VAR_ENCRYPTED_PREFIX) :]
    decrypted = decrypt_value_helper(
        value=inner, key=key, exception_type="debug", return_original_value=False
    )
    return decrypted if decrypted is not None else value
