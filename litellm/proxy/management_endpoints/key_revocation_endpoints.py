import re
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from typing import Annotated, Final, TypeAlias

from fastapi import APIRouter, Depends, Request, status
from pydantic import ConfigDict, Field

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid4
from litellm.constants import KEY_REVOKE_MAX_KEYS_PER_REQUEST
from litellm.models.verification_token import KeyRevocationStore, LiteLLM_VerificationToken
from litellm.proxy._types import LiteLLM_AuditLogs, LitellmTableNames, ProxyErrorTypes, ProxyException
from litellm.proxy.auth.auth_checks import delete_cache_key_objects
from litellm.proxy.auth.key_revoke_throttle import KeyRevokeThrottle
from litellm.proxy.deps.repositories import VerificationTokenRepositoryDep
from litellm.proxy.management_helpers.audit_logs import create_audit_logs_for_updates, is_audit_logging_enabled
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.proxy.auth.user_api_key_auth import hash_token

router: Final = APIRouter()

PUBLIC_KEY_REVOCATION_ACTOR: Final = "litellm_public_key_revocation"
_RAW_KEY: Final = re.compile(r"sk-[A-Za-z0-9_-]+")
_MAX_RAW_KEY_LENGTH: Final = 100

KeyRevocationAudit: TypeAlias = Callable[[Sequence[LiteLLM_AuditLogs]], Awaitable[None]]


class KeyRevokeRequest(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    keys: str | Annotated[tuple[str, ...], Field(min_length=1, max_length=KEY_REVOKE_MAX_KEYS_PER_REQUEST)] = Field(
        description=f"One virtual key (sk-...) or a list of up to {KEY_REVOKE_MAX_KEYS_PER_REQUEST} keys to block."
    )


class KeyRevokeResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)


async def get_key_revoke_throttle(request: Request) -> KeyRevokeThrottle:
    from litellm.proxy.proxy_server import general_settings_view, redis_usage_cache

    return KeyRevokeThrottle.from_request(request, general_settings_view(), redis_usage_cache)


async def get_key_revocation_audit() -> KeyRevocationAudit:
    return create_audit_logs_for_updates


KeyRevokeThrottleDep: TypeAlias = Annotated[KeyRevokeThrottle, Depends(get_key_revoke_throttle)]
KeyRevocationAuditDep: TypeAlias = Annotated[KeyRevocationAudit, Depends(get_key_revocation_audit)]


def _is_raw_key(entry: str) -> bool:
    return len(entry) <= _MAX_RAW_KEY_LENGTH and _RAW_KEY.fullmatch(entry) is not None


def revocable_hashes(keys: str | Sequence[str]) -> tuple[str, ...]:
    entries: Final = (keys,) if isinstance(keys, str) else keys
    return tuple(hash_token(entry) for entry in entries if _is_raw_key(entry))


def _audit_row(row: LiteLLM_VerificationToken, hashed_token: str) -> LiteLLM_AuditLogs:
    return LiteLLM_AuditLogs(
        id=str(uuid4()),
        updated_at=datetime.now(timezone.utc),
        changed_by=PUBLIC_KEY_REVOCATION_ACTOR,
        changed_by_api_key=None,
        table_name=LitellmTableNames.KEY_TABLE_NAME,
        object_id=hashed_token,
        action="blocked",
        updated_values="{}",
        before_value=row.model_dump_json(),
    )


async def _block_live_keys(
    tokens: KeyRevocationStore, hashed_tokens: Sequence[str], audit: KeyRevocationAudit, source: str
) -> None:
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    newly_blocked: Final = tuple(
        (row.token, row) for row in await tokens.find_by_tokens(hashed_tokens) if row.token and row.blocked is not True
    )
    if not newly_blocked:
        return
    blocked_tokens: Final = tuple(token for token, _ in newly_blocked)
    await tokens.block_tokens(blocked_tokens)
    await delete_cache_key_objects(
        hashed_tokens=blocked_tokens, user_api_key_cache=user_api_key_cache, proxy_logging_obj=proxy_logging_obj
    )
    if is_audit_logging_enabled():
        await audit(tuple(_audit_row(row, token) for token, row in newly_blocked))
    verbose_proxy_logger.warning(
        "Blocked %d virtual key(s) reported through POST /key/revoke from source %s; hashed tokens: %s",
        len(blocked_tokens),
        source,
        ", ".join(blocked_tokens),
    )


def _too_many_requests(retry_after: int) -> ProxyException:
    return ProxyException(
        message="Too many revocation requests from this address. Retry later.",
        type="rate_limit_error",
        param=None,
        code=status.HTTP_429_TOO_MANY_REQUESTS,
        headers={"Retry-After": str(retry_after)},
    )


def _revocation_failed() -> ProxyException:
    return ProxyException(
        message="Could not process the revocation request. Retry it.",
        type=ProxyErrorTypes.internal_server_error,
        param=None,
        code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


@router.post(
    "/key/revoke",
    tags=["key management"],
    status_code=status.HTTP_202_ACCEPTED,
    response_model=KeyRevokeResponse,
)
async def revoke_keys(
    data: KeyRevokeRequest,
    throttle: KeyRevokeThrottleDep,
    tokens: VerificationTokenRepositoryDep,
    audit: KeyRevocationAuditDep,
) -> KeyRevokeResponse:
    """
    Block virtual keys that were published by accident, without an admin.

    Anyone holding a key can report it, so this route takes no authentication. Every request that passes
    validation gets 202 with an empty body, whether or not any submitted key exists. Each source address can
    send 60 requests an hour.
    """
    refused: Final = await throttle.check()
    if refused is not None:
        raise _too_many_requests(refused.retry_after)
    hashed_tokens: Final = revocable_hashes(data.keys)
    if tokens is None or not hashed_tokens:
        return KeyRevokeResponse()
    try:
        await _block_live_keys(tokens, hashed_tokens, audit, throttle.source)
    except Exception as err:  # noqa: BLE001  # every storage or cache failure gets the same retryable answer
        verbose_proxy_logger.exception("POST /key/revoke could not block the submitted keys: %s", err)
        raise _revocation_failed() from err
    return KeyRevokeResponse()
