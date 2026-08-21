"""Membership-based usage attribution.

By default LiteLLM attributes a request to the single team stamped on the
virtual key (or the single team a JWT claim resolved to), and to that team's
organization. A user who belongs to many teams therefore contributes spend to
whichever team the key happens to name and nothing to the rest, and an operator
who wants "what did this team consume?" only gets an answer for keys that
happen to name it.

Two opt-in settings change that. Both default to off, so an existing deployment
keeps the single-team behavior byte for byte:

``track_spend_across_all_user_teams``
    Spend increments, daily rollups, and budget gates apply to every team the
    caller belongs to, and to every organization reached through those teams.

``enforce_rate_limits_across_all_user_teams``
    The same expansion for the RPM/TPM limiter, so a request must fit inside
    every membership's limit rather than only the stamped team's.

They are separate settings because they carry different costs. Spend
attribution is additive bookkeeping: the only surprise is that summing team
spend now exceeds real spend, because one request is deliberately charged to
several teams. Rate-limit expansion is a live behavior change: a caller's
effective limit becomes the MINIMUM across their memberships, so a busy team
can throttle someone who is mostly working for a different team. Operators
should be able to adopt the first without the second.

Nothing here builds a team tree. LiteLLM teams are a flat set, each optionally
belonging to one organization -- there is no ``parent_team_id`` in the schema.
"All memberships" therefore means the caller's teams plus the organizations
those teams (and the caller's own user row) belong to, never a recursive walk.
"""

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_UserTable, UserAPIKeyAuth

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.proxy._types import LiteLLM_TeamTableCachedObj
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import PrismaClient, ProxyLogging

SPEND_ATTRIBUTION_SETTING: Final = "track_spend_across_all_user_teams"
RATE_LIMIT_ATTRIBUTION_SETTING: Final = "enforce_rate_limits_across_all_user_teams"

# One resolved team: its id, and the team row if it could be loaded.
TeamResolution: TypeAlias = tuple[str, "LiteLLM_TeamTableCachedObj | None"]


def spend_attribution_enabled(general_settings: Mapping[str, object] | None) -> bool:
    """Whether spend, rollups, and budget gates fan out across memberships."""
    if not general_settings:
        return False
    return general_settings.get(SPEND_ATTRIBUTION_SETTING) is True


def rate_limit_attribution_enabled(general_settings: Mapping[str, object] | None) -> bool:
    """Whether the RPM/TPM limiter fans out across memberships."""
    if not general_settings:
        return False
    return general_settings.get(RATE_LIMIT_ATTRIBUTION_SETTING) is True


def _attribution_enabled(general_settings: Mapping[str, object] | None) -> bool:
    return spend_attribution_enabled(general_settings) or rate_limit_attribution_enabled(general_settings)


def attributed_team_ids(valid_token: UserAPIKeyAuth | None) -> tuple[str, ...]:
    """Every team this request is attributed to, stamped team first.

    Falls back to the single stamped team whenever attribution is off or
    resolved nothing, so a call site can use this unconditionally and keep
    identical behavior with the settings disabled.
    """
    if valid_token is None:
        return ()
    resolved: Final = valid_token.attributed_team_ids
    if resolved:
        return tuple(resolved)
    return (valid_token.team_id,) if valid_token.team_id else ()


def attributed_org_ids(valid_token: UserAPIKeyAuth | None) -> tuple[str, ...]:
    """Every organization this request is attributed to.

    Same fallback contract as :func:`attributed_team_ids`.
    """
    if valid_token is None:
        return ()
    resolved: Final = valid_token.attributed_org_ids
    if resolved:
        return tuple(resolved)
    return (valid_token.org_id,) if valid_token.org_id else ()


def attribution_targets(attributed_ids: Sequence[str] | None, stamped_id: str | None) -> tuple[str, ...]:
    """The entity ids one request should be charged against.

    The attributed set when membership attribution resolved one, otherwise the
    single stamped id -- so with both settings off this returns exactly
    ``(stamped_id,)`` and every caller keeps its historical behavior.

    Order-preserving dedupe: the stamped team normally also appears in the
    caller's membership list, and charging it twice would double-count.
    """
    if attributed_ids:
        return _ordered_unique(attributed_ids)
    return (stamped_id,) if stamped_id else ()


def _ordered_unique(values: Iterable[str | None]) -> tuple[str, ...]:
    """Dedupe while preserving order, dropping empties.

    Order is preserved so the stamped team stays first. Budget and rate-limit
    errors report the first offending entity, and a caller reading that error
    is best served by hearing about the team their key actually names.
    """
    return tuple(dict.fromkeys(v for v in values if v))


async def resolve_membership_attribution(
    *,
    user_api_key_auth_obj: UserAPIKeyAuth,
    user_object: LiteLLM_UserTable | None,
    general_settings: Mapping[str, object] | None,
    prisma_client: "PrismaClient | None",
    user_api_key_cache: "UserApiKeyCache",
    proxy_logging_obj: "ProxyLogging | None" = None,
) -> None:
    """Populate the attributed-membership fields on ``user_api_key_auth_obj``.

    No-op unless one of the attribution settings is on, so the default path
    pays nothing -- not even a cache read.

    Fails open. A team that cannot be resolved (deleted row, transient DB
    error) is skipped rather than raised: attribution is bookkeeping layered on
    top of an authorization decision that has already been made, and losing one
    team's attribution must never turn an authorized request into a 500. The
    stamped team is seeded first and is never dropped, so a failure degrades to
    exactly the default single-team behavior.
    """
    if not _attribution_enabled(general_settings):
        return

    if prisma_client is None:
        return

    # The stamped team leads, then the caller's SCIM/IdP-maintained membership
    # list. Seeding the stamped team explicitly matters: LiteLLM_UserTable.teams
    # is maintained by SCIM and JWT sync, so it can lag a key that was just
    # pointed at a new team, and losing that team would silently under-charge
    # the one team the operator explicitly named.
    candidate_team_ids: Final = _ordered_unique(
        (user_api_key_auth_obj.team_id, *(user_object.teams if user_object and user_object.teams else ()))
    )

    if not candidate_team_ids:
        _apply_org_attribution(
            user_api_key_auth_obj=user_api_key_auth_obj,
            user_object=user_object,
            team_objects=(),
        )
        return

    team_objects: Final = await _load_team_objects(
        team_ids=candidate_team_ids,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=user_api_key_auth_obj.parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )

    resolved_ids: Final = tuple(team_id for team_id, team_object in team_objects if team_object is not None)
    if resolved_ids:
        team_limits: Final = MappingProxyType(
            {
                team_id: MappingProxyType(
                    {
                        "rpm": getattr(team_object, "rpm_limit", None),
                        "tpm": getattr(team_object, "tpm_limit", None),
                    }
                )
                for team_id, team_object in team_objects
                if team_object is not None
            }
        )
        # The resolved context belongs on the auth object every later stage
        # already reads -- the key/team org fallback immediately above this
        # call does exactly the same. Returning a new object instead would
        # mean rebuilding every consumer of user_api_key_auth.
        user_api_key_auth_obj.attributed_team_ids = resolved_ids  # rebind-ok: stamping resolved auth context
        user_api_key_auth_obj.attributed_team_limits = team_limits  # rebind-ok: stamping resolved auth context

    _apply_org_attribution(
        user_api_key_auth_obj=user_api_key_auth_obj,
        user_object=user_object,
        team_objects=team_objects,
    )


def _apply_org_attribution(
    *,
    user_api_key_auth_obj: UserAPIKeyAuth,
    user_object: LiteLLM_UserTable | None,
    team_objects: Sequence[TeamResolution],
) -> None:
    """Derive the attributed organizations from what is already loaded.

    Sources, in order: the org already resolved onto the token, the caller's own
    ``LiteLLM_UserTable.organization_id``, then the org of each attributed team.
    Deliberately no ``LiteLLM_OrganizationMembership`` query -- that table would
    add a fresh round trip to the hot path, and these three sources already
    cover both the common single-org deployment and a caller whose teams span
    several organizations.
    """
    org_ids: Final = _ordered_unique(
        (
            user_api_key_auth_obj.org_id,
            user_object.organization_id if user_object else None,
            *(
                getattr(team_object, "organization_id", None)
                for _team_id, team_object in team_objects
                if team_object is not None
            ),
        )
    )
    if org_ids:
        user_api_key_auth_obj.attributed_org_ids = org_ids  # rebind-ok: stamping resolved auth context


async def _load_team_objects(
    *,
    team_ids: Sequence[str],
    prisma_client: "PrismaClient",
    user_api_key_cache: "UserApiKeyCache",
    parent_otel_span: "Span | None",
    proxy_logging_obj: "ProxyLogging | None",
) -> tuple[TeamResolution, ...]:
    """Resolve every candidate team, preserving input order.

    Lookups run concurrently: ``get_team_object`` is cache-first, so the steady
    state is N in-memory hits, but a cold pod pays a DB read per team and those
    must not serialize.
    """
    from litellm.proxy.auth.auth_checks import get_team_object

    async def _safe_get(team_id: str) -> TeamResolution:
        try:
            team_object: Final = await get_team_object(
                team_id=team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception as e:  # noqa: BLE001  # attribution must never fail an authorized request
            verbose_proxy_logger.debug(
                "membership attribution: skipping team_id=%s, could not resolve: %s",
                team_id,
                e,
            )
            return team_id, None
        else:
            return team_id, team_object

    return tuple(await asyncio.gather(*(_safe_get(team_id) for team_id in team_ids)))
