import os
import sys
from typing import Dict

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy.proxy_server import LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.types.guardrails import *

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path


def can_modify_guardrails(team_obj: Optional[LiteLLM_TeamTable]) -> bool:
    if team_obj is None:
        return True

    team_metadata = team_obj.metadata or {}

    if team_metadata.get("guardrails", None) is not None and isinstance(team_metadata.get("guardrails"), Dict):
        if team_metadata.get("guardrails", {}).get("modify_guardrails", None) is False:
            return False

    return True


def _callbacks_for_request_guardrails(request_guardrails: Dict[str, bool], should_run: bool) -> "frozenset[str]":
    """
    Collects the callback names of every guardrail the caller explicitly toggled
    to `should_run` in a per-request override
    """
    return frozenset(
        callback
        for _guardrail_name, _should_run in request_guardrails.items()
        if (_should_run is not False) is should_run and _guardrail_name in litellm.guardrail_name_config_map
        for callback in litellm.guardrail_name_config_map[_guardrail_name].callbacks
    )


def _guardrail_callback_default_on(guardrail_name: str) -> bool:
    """
    Returns True if any configured guardrail owning this callback is default_on
    """
    return any(
        guardrail_item.default_on
        for guardrail_item in litellm.guardrail_name_config_map.values()
        if guardrail_name in guardrail_item.callbacks
    )


async def should_proceed_based_on_metadata(data: dict, guardrail_name: str) -> bool:
    """
    checks if this guardrail should be applied to this call

    A per-request `metadata.guardrails` override only affects the guardrails the
    caller actually named. Any guardrail the caller did not mention falls back to
    its configured default, so `default_on` guardrails keep running even when the
    request names a different guardrail (or names none at all)
    """
    metadata = data.get("metadata")
    if not isinstance(metadata, dict) or "guardrails" not in metadata:
        return True

    # expect users to pass
    # guardrails: { prompt_injection: true, rail_2: false }
    request_guardrails = metadata["guardrails"]
    verbose_proxy_logger.debug(
        "Guardrails %s passed in request - checking which to apply",
        request_guardrails,
    )

    # v1 implementation of this only understands the dict form
    if not isinstance(request_guardrails, dict):
        return True

    enabled_callback_names = _callbacks_for_request_guardrails(request_guardrails, should_run=True)
    disabled_callback_names = _callbacks_for_request_guardrails(request_guardrails, should_run=False)
    verbose_proxy_logger.debug("enabled_callback_names %s", enabled_callback_names)

    if guardrail_name in enabled_callback_names:
        return True

    if guardrail_name in disabled_callback_names:
        return False

    return _guardrail_callback_default_on(guardrail_name)


async def should_proceed_based_on_api_key(user_api_key_dict: UserAPIKeyAuth, guardrail_name: str) -> bool:
    """
    checks if this guardrail should be applied to this call
    """
    if user_api_key_dict.permissions is not None:
        # { prompt_injection: true, rail_2: false }
        verbose_proxy_logger.debug(
            "Guardrails valid for API Key= %s - checking which to apply",
            user_api_key_dict.permissions,
        )

        if not isinstance(user_api_key_dict.permissions, dict):
            verbose_proxy_logger.error(
                "API Key permissions must be a dict - %s running guardrail %s",
                user_api_key_dict,
                guardrail_name,
            )
            return True

        for _guardrail_name, should_run in user_api_key_dict.permissions.items():
            if should_run is False:
                verbose_proxy_logger.debug(
                    "Guardrail %s skipped because request set to False",
                    _guardrail_name,
                )
                continue

            # lookup the guardrail in guardrail_name_config_map
            guardrail_item: GuardrailItem = litellm.guardrail_name_config_map[_guardrail_name]

            guardrail_callbacks = guardrail_item.callbacks
            if guardrail_name in guardrail_callbacks:
                return True

        # Do not proceeed if - "metadata": { "guardrails": { "lakera_prompt_injection": false } }
        return False
    return True
