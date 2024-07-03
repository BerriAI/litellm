from typing import Any, List, Optional, get_args

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, LiteLLMPromptInjectionParams
from litellm.proxy.utils import get_instance_fn

blue_color_code = "\033[94m"
reset_color_code = "\033[0m"


def initialize_callbacks_on_proxy(
    value: Any,
    premium_user: bool,
    config_file_path: str,
    litellm_settings: dict,
):
    from litellm.proxy.proxy_server import prisma_client

    verbose_proxy_logger.debug(
        f"{blue_color_code}initializing callbacks={value} on proxy{reset_color_code}"
    )
    if isinstance(value, list):
        imported_list: List[Any] = []
        known_compatible_callbacks = list(
            get_args(litellm._custom_logger_compatible_callbacks_literal)
        )

        for callback in value:  # ["presidio", <my-custom-callback>]
            if isinstance(callback, str) and callback in known_compatible_callbacks:
                imported_list.append(callback)
            elif isinstance(callback, str) and callback == "otel":
                from litellm.integrations.opentelemetry import OpenTelemetry

                open_telemetry_logger = OpenTelemetry()

                imported_list.append(open_telemetry_logger)
            elif isinstance(callback, str) and callback == "presidio":
                from litellm.proxy.hooks.presidio_pii_masking import (
                    _OPTIONAL_PresidioPIIMasking,
                )

                pii_masking_object = _OPTIONAL_PresidioPIIMasking()
                imported_list.append(pii_masking_object)
            elif isinstance(callback, str) and callback == "llamaguard_moderations":
                from enterprise.enterprise_hooks.llama_guard import (
                    _ENTERPRISE_LlamaGuard,
                )

                if premium_user != True:
                    raise Exception(
                        "Trying to use Llama Guard"
                        + CommonProxyErrors.not_premium_user.value
                    )

                llama_guard_object = _ENTERPRISE_LlamaGuard()
                imported_list.append(llama_guard_object)
            elif isinstance(callback, str) and callback == "hide_secrets":
                from enterprise.enterprise_hooks.secret_detection import (
                    _ENTERPRISE_SecretDetection,
                )

                if premium_user != True:
                    raise Exception(
                        "Trying to use secret hiding"
                        + CommonProxyErrors.not_premium_user.value
                    )

                _secret_detection_object = _ENTERPRISE_SecretDetection()
                imported_list.append(_secret_detection_object)
            elif isinstance(callback, str) and callback == "openai_moderations":
                from enterprise.enterprise_hooks.openai_moderation import (
                    _ENTERPRISE_OpenAI_Moderation,
                )

                if premium_user != True:
                    raise Exception(
                        "Trying to use OpenAI Moderations Check"
                        + CommonProxyErrors.not_premium_user.value
                    )

                openai_moderations_object = _ENTERPRISE_OpenAI_Moderation()
                imported_list.append(openai_moderations_object)
            elif isinstance(callback, str) and callback == "lakera_prompt_injection":
                from enterprise.enterprise_hooks.lakera_ai import (
                    _ENTERPRISE_lakeraAI_Moderation,
                )

                if premium_user != True:
                    raise Exception(
                        "Trying to use LakeraAI Prompt Injection"
                        + CommonProxyErrors.not_premium_user.value
                    )

                lakera_moderations_object = _ENTERPRISE_lakeraAI_Moderation()
                imported_list.append(lakera_moderations_object)
            elif isinstance(callback, str) and callback == "google_text_moderation":
                from enterprise.enterprise_hooks.google_text_moderation import (
                    _ENTERPRISE_GoogleTextModeration,
                )

                if premium_user != True:
                    raise Exception(
                        "Trying to use Google Text Moderation"
                        + CommonProxyErrors.not_premium_user.value
                    )

                google_text_moderation_obj = _ENTERPRISE_GoogleTextModeration()
                imported_list.append(google_text_moderation_obj)
            elif isinstance(callback, str) and callback == "llmguard_moderations":
                from enterprise.enterprise_hooks.llm_guard import _ENTERPRISE_LLMGuard

                if premium_user != True:
                    raise Exception(
                        "Trying to use Llm Guard"
                        + CommonProxyErrors.not_premium_user.value
                    )

                llm_guard_moderation_obj = _ENTERPRISE_LLMGuard()
                imported_list.append(llm_guard_moderation_obj)
            elif isinstance(callback, str) and callback == "blocked_user_check":
                from enterprise.enterprise_hooks.blocked_user_list import (
                    _ENTERPRISE_BlockedUserList,
                )

                if premium_user != True:
                    raise Exception(
                        "Trying to use ENTERPRISE BlockedUser"
                        + CommonProxyErrors.not_premium_user.value
                    )

                blocked_user_list = _ENTERPRISE_BlockedUserList(
                    prisma_client=prisma_client
                )
                imported_list.append(blocked_user_list)
            elif isinstance(callback, str) and callback == "banned_keywords":
                from enterprise.enterprise_hooks.banned_keywords import (
                    _ENTERPRISE_BannedKeywords,
                )

                if premium_user != True:
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
                        azure_content_safety_params[k] = litellm.get_secret(v)

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
