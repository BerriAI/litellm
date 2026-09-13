"""The proxy-API side of the native-client sign-in: turning a consented OAuth grant into
the same per-user credential ``lite login`` stores, so the bearer a CLI obtains through
the browser flow is accepted on every proxy route with user and team attribution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from litellm.constants import CLI_JWT_EXPIRATION_HOURS
from litellm.proxy._experimental.mcp_server.bridge_token_flow import load_active_user_by_id
from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import (
    ConsentTeam,
    MintedProxyCredential,
    ProxyCredentialMintFailure,
    ReloadUserFailure,
)
from litellm.proxy._types import LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken
from litellm.proxy.management_endpoints.ui_sso import (
    CliSsoTeamDetail,
    fetch_cli_sso_team_details,
    selected_cli_sso_team_detail,
)


async def lookup_consent_teams(user_id: str) -> tuple[ConsentTeam, ...] | ReloadUserFailure:
    user: Final = await load_active_user_by_id(user_id)
    if isinstance(user, str):
        return user
    details: Final = await _team_details(user.teams)
    if details is None:
        return "unavailable"
    return tuple(
        ConsentTeam(team_id=detail.team_id, team_alias=detail.team_alias)
        for detail in details
        if detail.team_id is not None
    )


async def mint_proxy_credential(
    user_id: str, team_id: str | None
) -> MintedProxyCredential | ProxyCredentialMintFailure:
    """Mint the ``lite login`` credential for a consented grant. Membership is checked
    live, so a team the user left between consent and redemption (or between refreshes)
    refuses the grant instead of minting a credential attributed to a team they are no
    longer on. The team is exactly the one the consent page sealed into the grant; nothing
    is picked on the user's behalf here, so a refresh can never move the credential, and a
    grant that names no team is refused for a user with a live team to pick from (the same
    rule ``lite login`` applies), so a user cannot step outside their teams' attribution by
    posting the consent form without one. Memberships whose team rows are gone count as no
    team at all, the way ``lite login`` treats them, so they can never lock a user out. The
    user row handed to the minter carries no team list, exactly like ``lite login``'s, so
    the minter's own first-team fallback stays inert."""
    user: Final = await load_active_user_by_id(user_id)
    if isinstance(user, str):
        return user
    if user.user_role is None:
        return "no_active_key"
    if team_id is not None and team_id not in user.teams:
        return "not_a_member"
    details: Final = await _team_details(user.teams) if user.teams else ()
    if details is None:
        return "unavailable"
    if team_id is None and any(detail.team_id is not None for detail in details):
        return "team_required"
    selected: Final = selected_cli_sso_team_detail(details, team_id)
    if selected is None:
        return "not_a_member"
    key: Final = ExperimentalUIJWTToken.get_cli_jwt_auth_token(
        user_info=LiteLLM_UserTable(user_id=user.user_id, user_role=user.user_role, models=user.models),
        team_id=team_id,
        team_alias=selected.team_alias,
        team_models=selected.team_models,
        team_model_aliases=selected.team_model_aliases,
    )
    return MintedProxyCredential(
        key=key,
        expires_in=CLI_JWT_EXPIRATION_HOURS * 3600,
        user_id=user.user_id,
        team_id=team_id,
    )


async def _team_details(teams: Sequence[str]) -> tuple[CliSsoTeamDetail, ...] | None:
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # rebound after startup, so read it per call

    if prisma_client is None:
        return None
    return await fetch_cli_sso_team_details(prisma_client, teams)
