import json
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Final, Literal

from fastapi import HTTPException
from pydantic import StrictBool, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.utils import premium_user_check
from litellm.repositories.table_repositories import GuardrailsRepository
from litellm.types.guardrails import *

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


def can_modify_guardrails(team_obj: LiteLLM_TeamTable | None) -> bool:
    if team_obj is None:
        return True

    team_metadata: Final = team_obj.metadata or {}

    if team_metadata.get("guardrails", None) is not None and isinstance(team_metadata.get("guardrails"), dict):
        if team_metadata.get("guardrails", {}).get("modify_guardrails", None) is False:
            return False

    return True


async def should_proceed_based_on_metadata(data: dict, guardrail_name: str) -> bool:
    """
    checks if this guardrail should be applied to this call
    """
    if "metadata" in data and isinstance(data["metadata"], dict):
        if "guardrails" in data["metadata"]:
            # expect users to pass
            # guardrails: { prompt_injection: true, rail_2: false }
            request_guardrails: Final = data["metadata"]["guardrails"]
            verbose_proxy_logger.debug(
                "Guardrails %s passed in request - checking which to apply",
                request_guardrails,
            )

            requested_callback_names: Final = []

            # v1 implementation of this
            if isinstance(request_guardrails, dict):
                # get guardrail configs from `init_guardrails.py`
                # for all requested guardrails -> get their associated callbacks
                for _guardrail_name, should_run in request_guardrails.items():
                    if should_run is False:
                        verbose_proxy_logger.debug(
                            "Guardrail %s skipped because request set to False",
                            _guardrail_name,
                        )
                        continue

                    # lookup the guardrail in guardrail_name_config_map
                    guardrail_item: GuardrailItem = litellm.guardrail_name_config_map[_guardrail_name]

                    guardrail_callbacks = guardrail_item.callbacks
                    requested_callback_names.extend(guardrail_callbacks)

                verbose_proxy_logger.debug("requested_callback_names %s", requested_callback_names)
                if guardrail_name in requested_callback_names:
                    return True

                # Do no proceeed if - "metadata": { "guardrails": { "lakera_prompt_injection": false } }
                return False

    return True


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


_GUARDRAIL_ENTRIES: Final = TypeAdapter(list[str | dict[str, object]])
_MODIFY_GUARDRAILS_PERMISSION: Final = TypeAdapter(dict[Literal["modify_guardrails"], StrictBool])


def _guardrail_entry_names(entries: object) -> frozenset[str] | None:
    try:
        parsed: Final = _GUARDRAIL_ENTRIES.validate_python(entries)
    except ValidationError:
        return None
    return frozenset(chain.from_iterable((entry,) if isinstance(entry, str) else entry.keys() for entry in parsed))


def _is_modify_guardrails_permission(entries: object) -> bool:
    # a team's documented {"modify_guardrails": false}, read by auth checks, is the one stored dict that has an effect
    try:
        return len(_MODIFY_GUARDRAILS_PERMISSION.validate_python(entries, strict=True)) == 1
    except ValidationError:
        return False


def _malformed_guardrails_message(entries: object) -> str:
    shown: Final = json.dumps(entries, default=str)
    return (
        f"Invalid guardrail detected - {shown if len(shown) <= 100 else shown[:100] + '...'}. "
        + "Please provide a valid list of guardrail names."
    )


async def registered_guardrail_names(names: frozenset[str], prisma_client: "PrismaClient | None") -> frozenset[str]:
    loaded_names: Final = frozenset(
        callback.guardrail_name
        for callback in litellm.logging_callback_manager.get_custom_loggers_for_type(callback_type=CustomGuardrail)
        if isinstance(callback, CustomGuardrail)
    )
    loaded: Final = names & loaded_names
    missing: Final = names - loaded
    if not missing or prisma_client is None:
        return loaded
    return loaded | await GuardrailsRepository(prisma_client, use_writer=True).find_active_guardrail_names(missing)


async def validate_guardrail_names(
    guardrails: Sequence[str] | None,
    metadata: Mapping[str, object] | None,
    existing_metadata: Mapping[str, object] | None,
    prisma_client: "PrismaClient | None",
    *,
    license_first: bool,
) -> None:
    if license_first and guardrails:
        # key and team writes license-gate top-level guardrails later on, keep that 403 ahead of a name 400
        premium_user_check("guardrails")
    requested_entries: Final = (
        guardrails if guardrails is not None else (metadata.get("guardrails") if metadata is not None else None)
    )
    if requested_entries is None or _is_modify_guardrails_permission(requested_entries):
        return
    requested: Final = _guardrail_entry_names(requested_entries)
    if requested is None:
        raise HTTPException(status_code=400, detail={"error": _malformed_guardrails_message(requested_entries)})
    stored_entries: Final = existing_metadata.get("guardrails") if existing_metadata is not None else None
    new_names: Final = requested - (_guardrail_entry_names(stored_entries) or frozenset())
    if not new_names:
        return
    unknown: Final = new_names - await registered_guardrail_names(new_names, prisma_client)
    if not unknown:
        return
    raise HTTPException(
        status_code=400,
        detail={
            "error": f"Unknown guardrail(s): {', '.join(sorted(unknown))}. "
            + "Create the guardrail first, or check the name against /v2/guardrails/list."
        },
    )
