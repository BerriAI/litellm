from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.auth.auth_checks import delete_cache_key_object
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.verification_token_repository import VerificationTokenRepository

if TYPE_CHECKING:
    from prisma.models import LiteLLM_VerificationToken as PrismaVerificationToken

SCIM_BLOCKED_METADATA_KEY: Final = "scim_blocked"


def _key_was_scim_blocked(metadata: object) -> bool:
    mapping: Final = as_str_mapping(metadata)
    return mapping is not None and mapping.get(SCIM_BLOCKED_METADATA_KEY) is True


async def set_user_keys_blocked(user_id: str, blocked: bool) -> int:
    """
    Block or unblock virtual keys owned by a user and invalidate them in the
    in-memory/redis caches so the change takes effect immediately.

    Each key this blocks is tagged with ``metadata.scim_blocked = True``. On
    reactivation only keys carrying that marker are unblocked, so a key an admin
    blocked manually for unrelated reasons is left alone.

    Returns the number of keys whose state was flipped.
    """
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No database connected"})
    tokens: Final[TableActions[PrismaVerificationToken]] = VerificationTokenRepository(prisma_client).table

    # `blocked` is a nullable column with no default, so existing rows typically hold NULL;
    # SQL equality on NULL would otherwise silently skip them.
    candidates: Final = await tokens.find_many(
        where=(
            {"user_id": user_id, "OR": [{"blocked": False}, {"blocked": None}]}
            if blocked
            else {"user_id": user_id, "blocked": True}
        ),
    )
    affected_keys: Final = candidates if blocked else [key for key in candidates if _key_was_scim_blocked(key.metadata)]

    for key_row in affected_keys:
        current_metadata: Mapping[str, object] = as_str_mapping(key_row.metadata) or {}
        new_metadata: Mapping[str, object] = (
            {**current_metadata, SCIM_BLOCKED_METADATA_KEY: True}
            if blocked
            else {k: v for k, v in current_metadata.items() if k != SCIM_BLOCKED_METADATA_KEY}
        )
        await tokens.update(
            where={"token": key_row.token},
            data={"blocked": blocked, "metadata": safe_dumps(new_metadata)},
        )

    for key_row in affected_keys:
        await delete_cache_key_object(
            hashed_token=key_row.token,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    if affected_keys:
        verbose_proxy_logger.info(
            "%s %d virtual key(s) for user_id=%s",
            "Blocked" if blocked else "Unblocked",
            len(affected_keys),
            user_id,
        )
    return len(affected_keys)
