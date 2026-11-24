from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import count
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from litellm.proxy.control_plane.auth.jwt import master_key_principal
from litellm.proxy.control_plane.common.fanout import Failure, parse_reply, unreachable_reason
from litellm.proxy.control_plane.dataplanes.client import DataplaneClient, DataplaneReply, DataplaneUnreachable
from litellm.proxy.control_plane.dataplanes.credentials import Caller, CredentialChain, ServiceTokenImpersonation
from litellm.proxy.control_plane.dataplanes.registry import Dataplane
from litellm.proxy.control_plane.store.fleet_teams import TeamMapping
from litellm.types.proxy.control_plane.fleet import FleetTeamBudget

_SERVICE: Final = Caller(master_key_principal(), "")


@dataclass(frozen=True, slots=True)
class BudgetUnreachable(Failure):
    pass


class _DataplanePayload(BaseModel):
    model_config = ConfigDict(frozen=True)


class _ListedTeam(_DataplanePayload):
    team_id: str
    team_alias: str | None


class _TeamPage(_DataplanePayload):
    teams: tuple[_ListedTeam, ...]
    total_pages: int = Field(ge=0)


class _Budget(_DataplanePayload):
    max_budget: float | None
    budget_duration: str | None


class _TeamInfo(_DataplanePayload):
    team_info: _Budget


_PAGE: Final = TypeAdapter(_TeamPage)
_INFO: Final = TypeAdapter(_TeamInfo)


def service_dataplane(dataplane: Dataplane) -> Dataplane | None:
    chain: Final = dataplane.credential
    if not isinstance(chain, CredentialChain):
        return None
    strategies: Final = tuple(
        strategy for strategy in chain.strategies if isinstance(strategy, ServiceTokenImpersonation)
    )
    return (
        replace(dataplane, credential=CredentialChain(strategies), credential_kinds=("service_token",))
        if strategies
        else None
    )


async def lookup_alias(client: DataplaneClient, dataplane: Dataplane, alias: str) -> TeamMapping | Failure:
    matches: Final[set[str]] = set()  # mutable-ok: collect distinct team IDs across pages
    for page in count(1):
        match parse_reply(
            await client.get(
                dataplane,
                "/v2/team/list",
                _SERVICE,
                (("team_alias", alias), ("page", str(page)), ("page_size", "100"), ("sort_by", "team_id")),
            ),
            _PAGE,
        ):
            case Failure() as failure:
                return failure
            case _TeamPage(teams=teams, total_pages=total_pages):
                matches.update(team.team_id for team in teams if team.team_alias == alias)
                if len(matches) > 1:
                    return Failure("Multiple teams match the exact alias.")
                if page >= total_pages:
                    break
    if not matches:
        return Failure("No team matches the exact alias.")
    return TeamMapping(dataplane_id=dataplane.id, dataplane_url=dataplane.url, team_id=next(iter(matches)))


async def differences(
    client: DataplaneClient, dataplane: Dataplane, team_id: str, desired: FleetTeamBudget
) -> Mapping[str, float | str | None] | Failure:
    result: Final = await client.get(
        dataplane, "/team/info", _SERVICE, (("team_id", team_id), ("include_members", "false"), ("key_limit", "1"))
    )
    if isinstance(result, DataplaneUnreachable):
        return BudgetUnreachable(unreachable_reason(result))
    if isinstance(result, DataplaneReply) and not 200 <= result.status < 300:
        return Failure(f"HTTP {result.status} with the {result.credential} credential")
    response: Final = parse_reply(result, _INFO)
    if isinstance(response, Failure):
        return response
    actual: Final = response.team_info
    return {
        field: value
        for field, value, observed in (
            ("max_budget", desired.max_budget, actual.max_budget),
            ("budget_duration", desired.budget_duration, actual.budget_duration),
        )
        if field in desired.model_fields_set and value != observed
    }


async def write_budget(
    client: DataplaneClient, dataplane: Dataplane, team_id: str, changes: Mapping[str, float | str | None]
) -> Failure | None:
    result: Final = await client.post(dataplane, "/team/update", _SERVICE, {"team_id": team_id, **changes})
    if isinstance(result, DataplaneReply) and 200 <= result.status < 300:
        return None
    if isinstance(result, DataplaneUnreachable):
        return BudgetUnreachable(unreachable_reason(result))
    status: Final = f" (HTTP {result.status})" if isinstance(result, DataplaneReply) else ""
    return Failure(f"Dataplane budget update failed{status}.")
