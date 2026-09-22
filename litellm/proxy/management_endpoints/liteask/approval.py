"""Sealed, short-lived proposals with a shared, at-most-once dispatch claim."""

import hmac
import secrets
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper

APPROVAL_TTL: Final = 300
_PREFIX: Final = "liteask-approval-v1:"
_ISSUED_MARKER: Final = "issued"
_ISSUE_SCRIPT: Final = """
if redis.call('SET', KEYS[1], ARGV[1], 'NX', 'EX', ARGV[2]) then
    return 1
end
return 0
"""
_CLAIM_SCRIPT: Final = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


class SealedApproval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    purpose: Literal["liteask-mutation-v1"] = "liteask-mutation-v1"
    user_id: str = Field(min_length=1)
    credential: str = Field(min_length=1)
    conversation_id: str
    tool: str
    arguments: Mapping[str, JsonValue]
    nonce: str
    expires_at: int


@dataclass(frozen=True, slots=True)
class ApprovalError:
    status_code: int
    message: str


class ScriptRunner(Protocol):
    def __call__(self, keys: Sequence[str], args: Sequence[str | bytes | int | float]) -> Awaitable[object]: ...


@runtime_checkable
class NonceStore(Protocol):
    def async_register_script(self, script: str) -> ScriptRunner: ...


async def seal_approval(
    *,
    user_id: str,
    credential: str,
    conversation_id: str,
    tool: str,
    arguments: Mapping[str, JsonValue],
    now: int,
    store: NonceStore | None,
) -> tuple[str, int] | ApprovalError:
    if store is None:
        return ApprovalError(503, "Changes require the gateway's shared Redis connection.")
    payload: Final = SealedApproval(
        user_id=user_id,
        credential=credential,
        conversation_id=conversation_id,
        tool=tool,
        arguments=arguments,
        nonce=secrets.token_urlsafe(24),
        expires_at=now + APPROVAL_TTL,
    )
    try:
        encrypted: Final = encrypt_value_helper(payload.model_dump_json())
        issue: Final = store.async_register_script(_ISSUE_SCRIPT)
        issued: Final = await issue(("liteask:approval:" + payload.nonce,), (_ISSUED_MARKER, APPROVAL_TTL))
    except Exception:  # noqa: BLE001  # an unacknowledged issuance must not return an executable proposal
        return ApprovalError(503, "Could not prepare this change. Try again later.")
    if type(issued) is not int or issued != 1:
        return ApprovalError(503, "Could not prepare this change. Try again later.")
    return _PREFIX + encrypted, payload.expires_at


def open_approval(
    token: str, *, user_id: str, credential: str, conversation_id: str, now: int
) -> SealedApproval | ApprovalError:
    invalid: Final = ApprovalError(400, "This approval is invalid or expired. Ask LiteAsk to prepare the change again.")
    if not token.startswith(_PREFIX):
        return invalid
    decrypted: Final = decrypt_value_helper(token[len(_PREFIX) :], "liteask_approval", exception_type="debug")
    if decrypted is None:
        return invalid
    try:
        payload: Final = SealedApproval.model_validate_json(decrypted)
    except ValidationError:
        return invalid
    if (
        payload.expires_at <= now
        or payload.expires_at > now + APPROVAL_TTL
        or not hmac.compare_digest(payload.user_id.encode(), user_id.encode())
        or not hmac.compare_digest(payload.credential.encode(), credential.encode())
        or payload.conversation_id != conversation_id
    ):
        return invalid
    return payload


async def claim_approval(payload: SealedApproval, *, store: NonceStore | None, now: int) -> ApprovalError | None:
    if payload.expires_at <= now:
        return ApprovalError(400, "This approval expired. Ask LiteAsk to prepare the change again.")
    if store is None:
        return ApprovalError(503, "Changes require the gateway's shared Redis connection.")
    try:
        claim: Final = store.async_register_script(_CLAIM_SCRIPT)
        claimed: Final = await claim(("liteask:approval:" + payload.nonce,), (_ISSUED_MARKER,))
    except Exception:  # noqa: BLE001  # every shared-store fault must fail closed, without falling back to local state
        return ApprovalError(503, "Could not verify this approval. No change was started.")
    if type(claimed) is not int or claimed != 1:
        return ApprovalError(409, "This approval is no longer available. Prepare the change again.")
    return None
