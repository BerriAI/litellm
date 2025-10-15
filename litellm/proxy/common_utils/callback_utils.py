from typing import Any, Dict, List, Literal, Optional

import litellm
from litellm import get_secret
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, LiteLLMPromptInjectionParams
from litellm.proxy.types_utils.utils import get_instance_fn

blue_color_code = "\033[94m"
reset_color_code = "\033[0m"


def initialize_callbacks_on_proxy(  # noqa: PLR0915
    value: Any,
    premium_user: bool,
    config_file_path: str,
    litellm_settings: dict,
    callback_specific_params: dict = {},
):
    from litellm.proxy.proxy_server import prisma_client

    verbose_proxy_logger.debug(
        f"{blue_color_code}initializing callbacks={value} on proxy{reset_color_code}"
    )
    if isinstance(value, list):
        imported_list: List[Any] = []
        for callback in value:  # ["presidio", <my-custom-callback>]
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
            try:
                from litellm_enterprise.integrations.prometheus import PrometheusLogger
            except Exception:
                PrometheusLogger = None

            if PrometheusLogger:
                PrometheusLogger._mount_metrics_endpoint(premium_user)
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
    _metadata = _litellm_params.get(get_metadata_variable_name_from_kwargs(kwargs)) or {}
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
    _metadata = request_data.get("metadata", None) or {}
    headers = {}
    if "applied_guardrails" in _metadata:
        headers["x-litellm-applied-guardrails"] = ",".join(
            _metadata["applied_guardrails"]
        )

    if "semantic-similarity" in _metadata:
        headers["x-litellm-semantic-similarity"] = str(_metadata["semantic-similarity"])

    return headers


def add_guardrail_to_applied_guardrails_header(
    request_data: Dict, guardrail_name: Optional[str]
):
    if guardrail_name is None:
        return
    _metadata = request_data.get("metadata", None) or {}
    if "applied_guardrails" in _metadata:
        _metadata["applied_guardrails"].append(guardrail_name)
    else:
        _metadata["applied_guardrails"] = [guardrail_name]


def get_metadata_variable_name_from_kwargs(
        kwargs: dict
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
