"""Cold-path identity loader: cache-or-one-DB-query for a hashed token.

The combined-view SQL is reused from ``PrismaClient.get_data`` rather than
duplicated. The cached payload is ``UserAPIKeyAuth``, the proxy's existing
carrier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fastapi import status

from litellm.identity.principal import classify_principal_kind
from litellm.integrations.otel.model.spans import SpanRole
from litellm.integrations.otel.runtime import traced
from litellm.proxy._types import ProxyErrorTypes, ProxyException

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import PrismaClient, ProxyLogging

    from litellm.identity.cache import IdentityCache


@traced(
    "identity.db.combined_view",
    role=SpanRole.DB_CALL,
    attrs=lambda result: {
        "identity.load.outcome": "found" if result is not None else "missing",
        "db.system.name": "postgresql",
    },
)
async def _fetch_from_db(
    *,
    hashed_token: str,
    prisma_client: "PrismaClient",
    user_api_key_cache: "UserApiKeyCache",
    parent_otel_span,
    proxy_logging_obj: Optional["ProxyLogging"],
) -> Optional["UserAPIKeyAuth"]:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.auth_checks import (
        _fetch_key_object_from_db_with_reconnect,
        get_object_permission,
        get_user_object,
    )

    row = await _fetch_key_object_from_db_with_reconnect(
        hashed_token=hashed_token,
        prisma_client=prisma_client,
        parent_otel_span=parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )
    if row is None:
        return None

    uak = UserAPIKeyAuth(**row.model_dump(exclude_none=True))

    if uak.object_permission_id and not uak.object_permission:
        try:
            uak.object_permission = await get_object_permission(
                object_permission_id=uak.object_permission_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception:
            pass

    # Bundle the user row into the identity cache so the auth chain's
    # follow-up `get_user_object` lookup is a free in-memory read on the
    # next request (the user object survives the same TTL as the rest of
    # the identity entry). The lookup itself is wrapped: failures fall
    # back to the legacy per-row cache that `get_user_object` populates.
    if uak.user_id:
        try:
            uak.user = await get_user_object(
                user_id=uak.user_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                user_id_upsert=False,
                parent_otel_span=parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception:
            uak.user = None

    return uak


@traced(
    "identity.load",
    role=SpanRole.SERVICE,
    attrs=lambda result: {
        "identity.principal.kind": classify_principal_kind(result),
        "identity.principal.has_team": bool(result.team_id),
        "identity.principal.has_org": bool(result.org_id),
        "identity.principal.has_project": bool(result.project_id),
        "identity.principal.has_agent": bool(result.agent_id),
    },
)
async def load_identity(
    *,
    hashed_token: str,
    prisma_client: "PrismaClient",
    cache: "IdentityCache",
    user_api_key_cache: "UserApiKeyCache",
    parent_otel_span=None,
    proxy_logging_obj: Optional["ProxyLogging"] = None,
) -> "UserAPIKeyAuth":
    """Return the hydrated ``UserAPIKeyAuth`` for a hashed token.

    Cache-hit returns immediately. Cache-miss issues exactly one combined
    view SQL query. A missing key raises ``ProxyException`` matching the
    legacy ``get_key_object`` contract so callers don't need to special
    case the cutover.
    """
    cached = await cache.get(hashed_token)
    if cached is not None:
        _rehydrate_bundled_user(cached)
        return _hydrated_copy(cached)

    if prisma_client is None:
        raise Exception(
            "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
        )

    uak = await _fetch_from_db(
        hashed_token=hashed_token,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )
    if uak is None:
        raise ProxyException(
            message=(
                "Authentication Error, Invalid proxy server token passed. "
                f"key={hashed_token}, not found in db. Create key via "
                "`/key/generate` call."
            ),
            type=ProxyErrorTypes.token_not_found_in_db,
            param="key",
            code=status.HTTP_401_UNAUTHORIZED,
        )

    await cache.set(hashed_token, uak)
    await _populate_legacy_cache(
        hashed_token=hashed_token,
        uak=uak,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )
    return _hydrated_copy(uak)


async def _populate_legacy_cache(
    *,
    hashed_token: str,
    uak: "UserAPIKeyAuth",
    user_api_key_cache: "UserApiKeyCache",
    proxy_logging_obj: Optional["ProxyLogging"],
) -> None:
    """Keep the legacy ``user_api_key_cache`` populated.

    The pre-DB cache peek in ``_user_api_key_auth_builder`` (the
    ``check_cache_only=True`` admin fast-path) reads from this cache, so
    populating it on every cold load preserves that fast path with no
    code change at the call site.
    """
    from litellm.proxy.auth.auth_checks import _cache_key_object

    try:
        await _cache_key_object(
            hashed_token=hashed_token,
            user_api_key_obj=uak,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    except Exception:
        pass


def _rehydrate_bundled_user(uak: "UserAPIKeyAuth") -> None:
    """Coerce the bundled user dict back into ``LiteLLM_UserTable``.

    ``UserAPIKeyAuth.user`` is typed ``Any`` so the codec round-trips it
    as a plain dict. Consumers read scalar attributes (`tpm_limit`,
    `metadata`, `user_role`, …) off the model, so we restore the typed
    form before handing the cached entry back.
    """
    from litellm.proxy._types import LiteLLM_UserTable

    raw = getattr(uak, "user", None)
    if isinstance(raw, dict):
        try:
            uak.user = LiteLLM_UserTable(**raw)
        except Exception:
            uak.user = None


def _hydrated_copy(uak: "UserAPIKeyAuth") -> "UserAPIKeyAuth":
    """Return a copy safe to mutate per-request.

    The cached entry is shared across requests; consumers mutate fields
    like ``parent_otel_span``, ``request_route``, and ``end_user_id`` on
    the returned object, so we hand each caller its own model copy with
    those request-scoped fields cleared.
    """
    copy = uak.model_copy()
    copy.parent_otel_span = None
    copy.request_route = None
    copy.budget_reservation = None
    return copy
