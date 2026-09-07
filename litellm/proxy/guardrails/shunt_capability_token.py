"""
Mints and opens the short-lived credential a shunt-generated command carries to authenticate
its own call back into `/v1/bulk_read` / `/v1/code_write`.

Identifies the caller by a reference to their existing key rather than the key itself, sealed
with the proxy's own encryption helper and given a short expiry, carried in an `Authorization`
header rather than a URL query string (which routinely ends up in access logs).

Deliberately not single-use: an agent retries a timed-out or interrupted Bash command, and a
single-use guard would turn that ordinary retry into a permanent 401. The token's only defense
is its short TTL; the worst a replay within that window can do is spend the caller's own budget
on a request they already made.
"""

import time
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper

_TOKEN_PREFIX: Final = "shunt_cap_v1:"
TOKEN_TTL_SECONDS: Final = 120


class ShuntCapabilityGrant(BaseModel):
    """What the sealed token attests: the caller to bill this call to, and until when it's valid.

    Exactly one of `key_hash` (the same hash `UserAPIKeyAuth.api_key` already stores for a
    DB-backed key) or `master_key` (the real master key, for the one caller with no such row)
    is set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    key_hash: str | None = None
    master_key: str | None = None
    exp: int = Field(gt=0)


def mint_shunt_capability_token(*, key_hash: str | None, master_key: str | None, now: float | None = None) -> str:
    """Seal a grant for the caller identified by exactly one of `key_hash` or `master_key`."""
    if (key_hash is None) == (master_key is None):
        raise ValueError("mint_shunt_capability_token requires exactly one of key_hash or master_key")
    grant: Final = ShuntCapabilityGrant(
        key_hash=key_hash, master_key=master_key, exp=int((now if now is not None else time.time()) + TOKEN_TTL_SECONDS)
    )
    return _TOKEN_PREFIX + encrypt_value_helper(grant.model_dump_json(exclude_none=True))


def open_shunt_capability_token(token: str, *, now: float | None = None) -> ShuntCapabilityGrant | None:
    """The grant a token carries, or None if it is malformed, unparseable, or expired.

    Total: every failure mode (wrong prefix, decrypt failure, schema mismatch, past expiry)
    returns None rather than raising, so the caller has one branch to handle -- reject the
    request -- instead of distinguishing why the token didn't validate.
    """
    if not token.startswith(_TOKEN_PREFIX):
        return None
    decrypted: Final = decrypt_value_helper(
        token[len(_TOKEN_PREFIX) :], "shunt_capability_token", return_original_value=False
    )
    if not isinstance(decrypted, str):
        return None
    try:
        grant: Final = ShuntCapabilityGrant.model_validate_json(decrypted)
    except ValidationError:
        return None
    if (grant.key_hash is None) == (grant.master_key is None):
        return None
    if grant.exp < (now if now is not None else time.time()):
        return None
    return grant
