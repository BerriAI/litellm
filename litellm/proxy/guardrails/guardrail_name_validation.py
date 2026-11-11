import json
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Final, Literal

from fastapi import HTTPException
from pydantic import StrictBool, TypeAdapter, ValidationError

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.utils import premium_user_check
from litellm.repositories.table_repositories import GuardrailsRepository

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

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
