"""
Self-hosted worker pairing tokens (LIT-2891 / Screen 3).

Flow:
1. UI calls `POST /v2/agent-workers/pair-token` → server generates a 32-byte
   urlsafe token, stores ONLY its sha256 in
   `LiteLLM_AgentWorkerPairingToken`, returns the raw token to the caller
   exactly once. TTL = 15 min.
2. The user runs the install one-liner with `--token <raw>`. The worker calls
   `POST /v2/agent-workers/register`, which calls `consume_pair_token` — that
   re-hashes the token, atomically marks the row `used_at=now()`, and returns
   the team_id. Single-use is enforced by checking `used_at IS NULL`.
3. The worker is then issued a long-lived JWT (also stored as a sha256 hash
   on the `LiteLLM_AgentWorker` row).

Raw tokens and JWTs are NEVER persisted — only their sha256 digests. This
matches the existing virtual-key hashed-token pattern.
"""

import hashlib
import secrets
import shlex
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from litellm.constants import CLOUD_AGENT_PAIR_TOKEN_TTL_MINUTES


@dataclass
class IssuedPairToken:
    raw_token: str  # returned to the caller ONCE
    token_hash: str  # what we persist
    expires_at: datetime  # UTC


def hash_pair_token(raw_token: str) -> str:
    """sha256 hex digest. Same algorithm used at issue time and consume time."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def issue_pair_token(
    *,
    ttl_minutes: int = CLOUD_AGENT_PAIR_TOKEN_TTL_MINUTES,
) -> IssuedPairToken:
    """Generate a fresh pair token. Caller persists `token_hash` + `expires_at`.

    The raw token is 32 bytes of urandom, urlsafe-base64-encoded — that's ~43
    chars of entropy, which is plenty for a single-use 15-min token.
    """
    raw_token = secrets.token_urlsafe(32)
    token_hash = hash_pair_token(raw_token)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
    return IssuedPairToken(
        raw_token=raw_token, token_hash=token_hash, expires_at=expires_at
    )


def hash_worker_jwt(raw_jwt: str) -> str:
    """sha256 of the worker JWT — what we persist on `LiteLLM_AgentWorker`."""
    return hashlib.sha256(raw_jwt.encode("utf-8")).hexdigest()


def is_expired(
    expires_at: Optional[datetime], *, now: Optional[datetime] = None
) -> bool:
    """True iff `expires_at` is in the past. Naive datetimes are treated as UTC."""
    if expires_at is None:
        return True
    current = now or datetime.now(timezone.utc)
    if expires_at.tzinfo is None:
        # Match the implicit-UTC convention of the existing proxy DB writes.
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return expires_at <= current


def build_install_command(
    *,
    proxy_url: str,
    raw_token: str,
    install_script_url: str = "https://litellm.ai/install-worker",
) -> str:
    """Render the one-liner the UI shows in the Add Machine modal.

    Kept as a helper (not f-string at the callsite) so tests can lock the
    exact format and so we can swap the install host without touching
    endpoint code.

    All operator-controlled values (`proxy_url`, `raw_token`, and the
    install script URL) are run through `shlex.quote` before interpolation
    so that spaces, quotes, or other shell metacharacters in any of them
    can't break out of the install command. The proxy URL is otherwise
    not validated here — the caller is responsible for verifying the
    host (see `worker_endpoints._resolve_proxy_url`).
    """
    quoted_url = shlex.quote(install_script_url)
    quoted_proxy = shlex.quote(proxy_url)
    quoted_token = shlex.quote(raw_token)
    return (
        f"curl -fsS {quoted_url} | sh -s -- "
        f"--proxy {quoted_proxy} --token {quoted_token}"
    )
