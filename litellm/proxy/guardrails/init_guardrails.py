import importlib
import traceback
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, RootModel

import litellm
from litellm import get_secret
from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.callback_utils import initialize_callbacks_on_proxy

# v2 implementation
from litellm.types.guardrails import (
    Guardrail,
    GuardrailEventHooks,
    GuardrailItem,
    GuardrailItemSpec,
    LakeraCategoryThresholds,
    LitellmParams,
)

all_guardrails: List[GuardrailItem] = []


def initialize_guardrails(
    guardrails_config: List[Dict[str, GuardrailItemSpec]],
    premium_user: bool,
    config_file_path: str,
    litellm_settings: dict,
) -> Dict[str, GuardrailItem]:
    try:
        verbose_proxy_logger.debug(f"validating  guardrails passed {guardrails_config}")
        global all_guardrails
        for item in guardrails_config:
            """
            one item looks like this:

            {'prompt_injection': {'callbacks': ['lakera_prompt_injection', 'prompt_injection_api_2'], 'default_on': True, 'enabled_roles': ['user']}}
            """
            for k, v in item.items():
                guardrail_item = GuardrailItem(**v, guardrail_name=k)
                all_guardrails.append(guardrail_item)
                litellm.guardrail_name_config_map[k] = guardrail_item

        # set appropriate callbacks if they are default on
        default_on_callbacks = set()
        callback_specific_params = {}
        for guardrail in all_guardrails:
            verbose_proxy_logger.debug(guardrail.guardrail_name)
            verbose_proxy_logger.debug(guardrail.default_on)

            callback_specific_params.update(guardrail.callback_args)

            if guardrail.default_on is True:
                # add these to litellm callbacks if they don't exist
                for callback in guardrail.callbacks:
                    if callback not in litellm.callbacks:
                        default_on_callbacks.add(callback)

                    if guardrail.logging_only is True:
                        if callback == "presidio":
                            callback_specific_params["presidio"] = {"logging_only": True}  # type: ignore

        default_on_callbacks_list = list(default_on_callbacks)
        if len(default_on_callbacks_list) > 0:
            initialize_callbacks_on_proxy(
                value=default_on_callbacks_list,
                premium_user=premium_user,
                config_file_path=config_file_path,
                litellm_settings=litellm_settings,
                callback_specific_params=callback_specific_params,
            )

        return litellm.guardrail_name_config_map
    except Exception as e:
        verbose_proxy_logger.exception(
            "error initializing guardrails {}".format(str(e))
        )
        raise e


"""
Map guardrail_name: <pre_call>, <post_call>, during_call

"""


def init_guardrails_v2(
    all_guardrails: dict,
    config_file_path: str,
):
    # Convert the loaded data to the TypedDict structure
    guardrail_list = []

    # Parse each guardrail and replace environment variables
    for guardrail in all_guardrails:

        # Init litellm params for guardrail
        litellm_params_data = guardrail["litellm_params"]
        verbose_proxy_logger.debug("litellm_params= %s", litellm_params_data)

        _litellm_params_kwargs = {
            k: litellm_params_data[k] if k in litellm_params_data else None
            for k in LitellmParams.__annotations__.keys()
        }

        litellm_params = LitellmParams(**_litellm_params_kwargs)  # type: ignore

        if (
            "category_thresholds" in litellm_params_data
            and litellm_params_data["category_thresholds"]
        ):
            lakera_category_thresholds = LakeraCategoryThresholds(
                **litellm_params_data["category_thresholds"]
            )
            litellm_params["category_thresholds"] = lakera_category_thresholds

        if litellm_params["api_key"]:
            if litellm_params["api_key"].startswith("os.environ/"):
                litellm_params["api_key"] = str(get_secret(litellm_params["api_key"]))  # type: ignore

        if litellm_params["api_base"]:
            if litellm_params["api_base"].startswith("os.environ/"):
                litellm_params["api_base"] = str(get_secret(litellm_params["api_base"]))  # type: ignore

        # Init guardrail CustomLoggerClass
        if litellm_params["guardrail"] == "aporia":
            from litellm.proxy.guardrails.guardrail_hooks.aporia_ai import (
                AporiaGuardrail,
            )

            _aporia_callback = AporiaGuardrail(
                api_base=litellm_params["api_base"],
                api_key=litellm_params["api_key"],
                guardrail_name=guardrail["guardrail_name"],
                event_hook=litellm_params["mode"],
            )
            litellm.callbacks.append(_aporia_callback)  # type: ignore
        if litellm_params["guardrail"] == "bedrock":
            from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
                BedrockGuardrail,
            )

            _bedrock_callback = BedrockGuardrail(
                guardrail_name=guardrail["guardrail_name"],
                event_hook=litellm_params["mode"],
                guardrailIdentifier=litellm_params["guardrailIdentifier"],
                guardrailVersion=litellm_params["guardrailVersion"],
            )
            litellm.callbacks.append(_bedrock_callback)  # type: ignore
        elif litellm_params["guardrail"] == "lakera":
            from litellm.proxy.guardrails.guardrail_hooks.lakera_ai import (
                lakeraAI_Moderation,
            )

            _lakera_callback = lakeraAI_Moderation(
                api_base=litellm_params["api_base"],
                api_key=litellm_params["api_key"],
                guardrail_name=guardrail["guardrail_name"],
                event_hook=litellm_params["mode"],
                category_thresholds=litellm_params.get("category_thresholds"),
            )
            litellm.callbacks.append(_lakera_callback)  # type: ignore
        elif litellm_params["guardrail"] == "presidio":
            from litellm.proxy.guardrails.guardrail_hooks.presidio import (
                _OPTIONAL_PresidioPIIMasking,
            )

            _presidio_callback = _OPTIONAL_PresidioPIIMasking(
                guardrail_name=guardrail["guardrail_name"],
                event_hook=litellm_params["mode"],
                output_parse_pii=litellm_params["output_parse_pii"],
                presidio_ad_hoc_recognizers=litellm_params[
                    "presidio_ad_hoc_recognizers"
                ],
                mock_redacted_text=litellm_params.get("mock_redacted_text") or None,
            )

            if litellm_params["output_parse_pii"] is True:
                _success_callback = _OPTIONAL_PresidioPIIMasking(
                    output_parse_pii=True,
                    guardrail_name=guardrail["guardrail_name"],
                    event_hook=GuardrailEventHooks.post_call.value,
                    presidio_ad_hoc_recognizers=litellm_params[
                        "presidio_ad_hoc_recognizers"
                    ],
                )

                litellm.callbacks.append(_success_callback)  # type: ignore

            litellm.callbacks.append(_presidio_callback)  # type: ignore
        elif litellm_params["guardrail"] == "hide-secrets":
            from enterprise.enterprise_hooks.secret_detection import (
                _ENTERPRISE_SecretDetection,
            )

            _secret_detection_object = _ENTERPRISE_SecretDetection(
                detect_secrets_config=litellm_params.get("detect_secrets_config"),
                event_hook=litellm_params["mode"],
                guardrail_name=guardrail["guardrail_name"],
            )

            litellm.callbacks.append(_secret_detection_object)  # type: ignore
        elif (
            isinstance(litellm_params["guardrail"], str)
            and "." in litellm_params["guardrail"]
        ):
            import os

            from litellm.proxy.utils import get_instance_fn

            # Custom guardrail
            _guardrail = litellm_params["guardrail"]
            _file_name, _class_name = _guardrail.split(".")
            verbose_proxy_logger.debug(
                "Initializing custom guardrail: %s, file_name: %s, class_name: %s",
                _guardrail,
                _file_name,
                _class_name,
            )

            directory = os.path.dirname(config_file_path)
            module_file_path = os.path.join(directory, _file_name)
            module_file_path += ".py"

            spec = importlib.util.spec_from_file_location(_class_name, module_file_path)  # type: ignore
            if spec is None:
                raise ImportError(
                    f"Could not find a module specification for {module_file_path}"
                )

            module = importlib.util.module_from_spec(spec)  # type: ignore
            spec.loader.exec_module(module)  # type: ignore
            _guardrail_class = getattr(module, _class_name)

            _guardrail_callback = _guardrail_class(
                guardrail_name=guardrail["guardrail_name"],
                event_hook=litellm_params["mode"],
            )
            litellm.callbacks.append(_guardrail_callback)  # type: ignore

        parsed_guardrail = Guardrail(
            guardrail_name=guardrail["guardrail_name"],
            litellm_params=litellm_params,
        )

        guardrail_list.append(parsed_guardrail)
        guardrail["guardrail_name"]
    # pretty print guardrail_list in green
    print(f"\nGuardrail List:{guardrail_list}\n")  # noqa
