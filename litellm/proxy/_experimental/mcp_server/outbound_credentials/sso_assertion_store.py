"""Store for the enterprise IdP identity assertion captured at SSO login (EMA).

The ``oauth2_id_jag`` egress arm needs the user's IdP ``id_token`` as its RFC 8693
``subject_token``. A front-door client holds an identity-only ``llm_session_`` bearer, not an
IdP assertion, so the assertion captured at the one SSO login is the only usable subject
source for it. This module owns both sides of that state: the SSO callback persists here
(write-through to the DB so a login on one pod is visible to every pod) and the resolver
seam reads back by ``user_id``. Retention is gated on an ``oauth2_id_jag`` server actually
being registered, so a gateway with no EMA upstream never stores bearer material.

The row is one encrypted payload per user, latest login wins. ``expires_at`` mirrors the
id_token ``exp`` claim and is judged by the reader, never enforced by deletion here: an
expired assertion with a refresh token is still renewable, and the DB row is the source of
truth, the same contract as the per-user OAuth credential store.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import jwt
from pydantic import BaseModel, ConfigDict, SecretStr, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_ASSERTION_DECRYPT_LOG_KEY = "sso_identity_assertion"
_STR_ADAPTER: TypeAdapter[str] = TypeAdapter(str)
_MAYBE_STR_ADAPTER: TypeAdapter[str | None] = TypeAdapter(str | None)


class SSOIdentityAssertion(BaseModel):
    """The IdP material an EMA exchange needs: ``id_token`` is the RFC 8693 subject token,
    ``expires_at`` bounds its usefulness, and the refresh token renews it without re-login."""

    model_config = ConfigDict(frozen=True)

    id_token: SecretStr
    refresh_token: SecretStr | None = None
    issuer: str | None = None
    expires_at: datetime | None = None


class _IdTokenClaims(BaseModel):
    exp: float | None = None
    iss: str | None = None


class _StoredAssertionPayload(BaseModel):
    id_token: str
    refresh_token: str | None = None
    issuer: str | None = None
    expires_at: datetime | None = None


def assertion_from_sso_login(id_token: object, refresh_token: object) -> SSOIdentityAssertion | None:
    """The typed carrier built where the raw token response exists; ``None`` when the provider
    sent no id_token or sent one that is not a decodable JWT, since neither is exchangeable
    under EMA. Inputs are ``object`` because they come straight from the provider's untyped
    token response; this is the one boundary that validates them. The token arrived over TLS
    from the IdP's own token endpoint, so claims are read without signature verification,
    matching how the SSO callback already decodes it for identity."""
    raw_id_token = id_token if isinstance(id_token, str) and id_token else None
    if raw_id_token is None:
        return None
    raw_refresh_token = refresh_token if isinstance(refresh_token, str) and refresh_token else None
    try:
        claims = _IdTokenClaims.model_validate(jwt.decode(raw_id_token, options={"verify_signature": False}))
        expires_at = datetime.fromtimestamp(claims.exp, tz=timezone.utc) if claims.exp is not None else None
    except Exception:  # noqa: BLE001  # decode failure = not retainable; never raise into login
        verbose_proxy_logger.warning(
            "SSO id_token could not be decoded or its claims were unusable; not retaining it for EMA egress."
        )
        return None
    return SSOIdentityAssertion(
        id_token=SecretStr(raw_id_token),
        refresh_token=SecretStr(raw_refresh_token) if raw_refresh_token else None,
        issuer=claims.iss,
        expires_at=expires_at,
    )


async def ema_assertion_retention_enabled() -> bool:
    """Whether any MCP server uses ``oauth2_id_jag``, evaluated per login so the gateway only
    retains bearer material while an EMA upstream exists to spend it on. The local registry is
    the fast path; when it has none, the DB is consulted as the authoritative source, because
    the registry is a per-process snapshot while the assertion write targets the shared DB. A
    server added on another pod (or not yet loaded during startup) must still enable retention
    here, or the drop only surfaces later as an unexplained challenge at the EMA upstream."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (  # noqa: PLC0415  # avoids import cycle
        global_mcp_server_manager,
    )
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global
    from litellm.types.mcp import MCPAuth  # noqa: PLC0415  # runtime global

    servers = global_mcp_server_manager.get_registry().values()
    if any(server.auth_type == MCPAuth.oauth2_id_jag for server in servers):
        return True
    if prisma_client is None:
        return False
    row = await prisma_client.db.litellm_mcpservertable.find_first(where={"auth_type": MCPAuth.oauth2_id_jag.value})
    return row is not None


async def persist_sso_identity_assertion(user_id: str, assertion: SSOIdentityAssertion) -> None:
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper  # noqa: PLC0415  # runtime global
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global

    if prisma_client is None:
        return
    payload: dict[str, str] = {
        "id_token": assertion.id_token.get_secret_value(),
        **({"refresh_token": assertion.refresh_token.get_secret_value()} if assertion.refresh_token else {}),
        **({"issuer": assertion.issuer} if assertion.issuer else {}),
        **({"expires_at": assertion.expires_at.isoformat()} if assertion.expires_at else {}),
    }
    encoded = _STR_ADAPTER.validate_python(encrypt_value_helper(json.dumps(payload)))
    await prisma_client.db.litellm_ssoidentityassertion.upsert(
        where={"user_id": user_id},
        data={
            "create": {"user_id": user_id, "assertion_b64": encoded},
            "update": {"assertion_b64": encoded},
        },
    )


async def fetch_sso_identity_assertion(user_id: str) -> SSOIdentityAssertion | None:
    """The stored assertion for ``user_id``, or ``None`` when absent, undecryptable (salt-key
    rotation), or unparseable. Expiry is not judged here; the reader owns that policy."""
    from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper  # noqa: PLC0415  # runtime global
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global

    if prisma_client is None:
        return None
    row = await prisma_client.db.litellm_ssoidentityassertion.find_unique(where={"user_id": user_id})
    if row is None:
        return None
    raw = _MAYBE_STR_ADAPTER.validate_python(
        decrypt_value_helper(row.assertion_b64, _ASSERTION_DECRYPT_LOG_KEY, exception_type="debug")
    )
    if raw is None:
        return None
    try:
        payload = _StoredAssertionPayload.model_validate_json(raw)
    except ValidationError:
        verbose_proxy_logger.warning(
            "Stored SSO identity assertion for user_id=%s could not be parsed; treating as absent.", user_id
        )
        return None
    return SSOIdentityAssertion(
        id_token=SecretStr(payload.id_token),
        refresh_token=SecretStr(payload.refresh_token) if payload.refresh_token else None,
        issuer=payload.issuer,
        expires_at=payload.expires_at,
    )


async def rotate_sso_identity_assertions_master_key(prisma_client: PrismaClient, new_master_key: str) -> None:
    """Re-encrypt every stored assertion under ``new_master_key`` during a salt-key rotation,
    mirroring the sibling per-user credential tables; an unreadable row is skipped so one
    corrupt row does not abort the rotation. Rows are decrypted one at a time inside the loop
    so the whole table's plaintext is never held in memory at once."""
    from prisma.models import LiteLLM_SSOIdentityAssertion as AssertionRow  # noqa: PLC0415  # generated at runtime

    from litellm.proxy.common_utils.encrypt_decrypt_utils import (  # noqa: PLC0415  # runtime global
        decrypt_value_helper,
        encrypt_value_helper,
    )

    async def _rotate_row(row: AssertionRow) -> bool:
        plaintext = _MAYBE_STR_ADAPTER.validate_python(
            decrypt_value_helper(row.assertion_b64, _ASSERTION_DECRYPT_LOG_KEY, exception_type="debug")
        )
        if plaintext is None:
            verbose_proxy_logger.warning(
                "rotate_sso_identity_assertions_master_key: could not decrypt assertion for user_id=%s, skipping",
                row.user_id,
            )
            return False
        re_encrypted = _STR_ADAPTER.validate_python(encrypt_value_helper(plaintext, new_encryption_key=new_master_key))
        await prisma_client.db.litellm_ssoidentityassertion.update(
            where={"user_id": row.user_id},
            data={"assertion_b64": re_encrypted},
        )
        return True

    rows = await prisma_client.db.litellm_ssoidentityassertion.find_many()
    outcomes = [await _rotate_row(row) for row in rows]
    verbose_proxy_logger.info(
        "rotate_sso_identity_assertions_master_key: rotated %d row(s), skipped %d",
        sum(outcomes),
        len(outcomes) - sum(outcomes),
    )


async def retain_sso_identity_assertion_for_ema(user_id: str, assertion: SSOIdentityAssertion | None) -> None:
    """The SSO-callback hook: a no-op unless there is material AND an EMA server is registered.
    A store failure is logged and swallowed because the login itself must not fail on an
    egress-side write; the cost of a miss is a 401 challenge at the EMA upstream, not a lockout."""
    if assertion is None:
        return
    try:
        if not await ema_assertion_retention_enabled():
            return
        await persist_sso_identity_assertion(user_id, assertion)
    except Exception as exc:  # noqa: BLE001  # the login itself must not fail on an egress-side write
        verbose_proxy_logger.warning(
            "Failed to persist the SSO identity assertion for EMA egress (user_id=%s): %s", user_id, exc
        )
