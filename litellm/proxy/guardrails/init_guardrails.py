import traceback
from typing import Dict, List

from pydantic import BaseModel, RootModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.init_callbacks import initialize_callbacks_on_proxy
from litellm.types.guardrails import GuardrailItem

all_guardrails: List[GuardrailItem] = []

guardrail_name_config_map: Dict[str, GuardrailItem] = {}


def initialize_guardrails(
    guardrails_config: list,
    premium_user: bool,
    config_file_path: str,
    litellm_settings: dict,
):
    try:
        verbose_proxy_logger.debug(f"validating  guardrails passed {guardrails_config}")
        global all_guardrails
        for item in guardrails_config:
            """
            one item looks like this:

            {'prompt_injection': {'callbacks': ['lakera_prompt_injection', 'prompt_injection_api_2'], 'default_on': True}}
            """

            for k, v in item.items():
                guardrail_item = GuardrailItem(**v, guardrail_name=k)
                all_guardrails.append(guardrail_item)
                guardrail_name_config_map[k] = guardrail_item

        # set appropriate callbacks if they are default on
        default_on_callbacks = set()
        for guardrail in all_guardrails:
            verbose_proxy_logger.debug(guardrail.guardrail_name)
            verbose_proxy_logger.debug(guardrail.default_on)

            if guardrail.default_on is True:
                # add these to litellm callbacks if they don't exist
                for callback in guardrail.callbacks:
                    if callback not in litellm.callbacks:
                        default_on_callbacks.add(callback)

        default_on_callbacks_list = list(default_on_callbacks)
        if len(default_on_callbacks_list) > 0:
            initialize_callbacks_on_proxy(
                value=default_on_callbacks_list,
                premium_user=premium_user,
                config_file_path=config_file_path,
                litellm_settings=litellm_settings,
            )

    except Exception as e:
        verbose_proxy_logger.error(f"error initializing guardrails {str(e)}")
        traceback.print_exc()
        raise e
