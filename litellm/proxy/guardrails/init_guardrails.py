import importlib
import os
from typing import Dict, List, Optional

import litellm
from litellm import get_secret
from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.callback_utils import initialize_callbacks_on_proxy

# v2 implementation
from litellm.types.guardrails import (
    Guardrail,
    GuardrailItem,
    GuardrailItemSpec,
    LakeraCategoryThresholds,
    LitellmParams,
)

from .guardrail_registry import guardrail_registry

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
    all_guardrails: List[Dict],
    config_file_path: Optional[str] = None,
):
    guardrail_list = []

    for guardrail in all_guardrails:
        litellm_params_data = guardrail["litellm_params"]
        verbose_proxy_logger.debug("litellm_params= %s", litellm_params_data)

        _litellm_params_kwargs = {
            k: litellm_params_data.get(k) for k in LitellmParams.__annotations__.keys()
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

        if litellm_params["api_key"] and litellm_params["api_key"].startswith(
            "os.environ/"
        ):
            litellm_params["api_key"] = str(get_secret(litellm_params["api_key"]))  # type: ignore

        if litellm_params["api_base"] and litellm_params["api_base"].startswith(
            "os.environ/"
        ):
            litellm_params["api_base"] = str(get_secret(litellm_params["api_base"]))  # type: ignore

        guardrail_type = litellm_params["guardrail"]

        initializer = guardrail_registry.get(guardrail_type)

        if initializer:
            initializer(litellm_params, guardrail)
        elif isinstance(guardrail_type, str) and "." in guardrail_type:
            if not config_file_path:
                raise Exception(
                    "GuardrailsAIException - Please pass the config_file_path to initialize_guardrails_v2"
                )

            _file_name, _class_name = guardrail_type.split(".")
            verbose_proxy_logger.debug(
                "Initializing custom guardrail: %s, file_name: %s, class_name: %s",
                guardrail_type,
                _file_name,
                _class_name,
            )

            directory = os.path.dirname(config_file_path)
            module_file_path = os.path.join(directory, _file_name) + ".py"

            spec = importlib.util.spec_from_file_location(_class_name, module_file_path)  # type: ignore
            if not spec:
                raise ImportError(
                    f"Could not find a module specification for {module_file_path}"
                )

            module = importlib.util.module_from_spec(spec)  # type: ignore
            spec.loader.exec_module(module)  # type: ignore
            _guardrail_class = getattr(module, _class_name)

            _guardrail_callback = _guardrail_class(
                guardrail_name=guardrail["guardrail_name"],
                event_hook=litellm_params["mode"],
                default_on=litellm_params["default_on"],
            )
            litellm.logging_callback_manager.add_litellm_callback(_guardrail_callback)  # type: ignore
        else:
            raise ValueError(f"Unsupported guardrail: {guardrail_type}")

        parsed_guardrail = Guardrail(
            guardrail_name=guardrail["guardrail_name"],
            litellm_params=litellm_params,
        )

        guardrail_list.append(parsed_guardrail)

    verbose_proxy_logger.info(f"\nGuardrail List:{guardrail_list}\n")
