import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol

from litellm._logging import verbose_proxy_logger
from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import LiteLLMRoutes, SpecialModelNames
from litellm.repositories.prisma_protocols import CountTable

DEPRECATION_DISCUSSION_URL: Final = "https://github.com/BerriAI/litellm/discussions/32090"


class CountTableRepository(Protocol):
    @property
    def table(self) -> CountTable: ...


@dataclass(frozen=True, slots=True)
class LegacyEmptyModelsCounts:
    teamless_keys_with_empty_models: int
    teamless_keys_with_all_team_models: int
    team_keys_with_empty_models: int
    teams_with_empty_models: int
    keys_with_access_groups_and_empty_models: int
    teams_with_access_groups_and_empty_models: int


def _model_calling_keys(where: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "AND": [
            where,
            {
                "NOT": [
                    {"allowed_routes": {"equals": [LiteLLMRoutes.management_routes.name]}},
                    {"allowed_routes": {"equals": [LiteLLMRoutes.info_routes.name]}},
                ]
            },
        ]
    }


async def count_legacy_empty_models_rows(keys: CountTable, teams: CountTable) -> LegacyEmptyModelsCounts:
    (
        teamless_empty,
        teamless_all_team,
        team_keys_empty,
        teams_empty,
        keys_with_groups_empty,
        teams_with_groups_empty,
    ) = await asyncio.gather(
        keys.count(
            where=_model_calling_keys(
                {"team_id": None, "models": {"isEmpty": True}, "access_group_ids": {"isEmpty": True}}
            )
        ),
        keys.count(
            where=_model_calling_keys(
                {
                    "team_id": None,
                    "models": {"has": SpecialModelNames.all_team_models.value},
                    "access_group_ids": {"isEmpty": True},
                }
            )
        ),
        keys.count(
            where=_model_calling_keys(
                {
                    "AND": [{"team_id": {"not": None}}, {"team_id": {"not": UI_SESSION_TOKEN_TEAM_ID}}],
                    "models": {"isEmpty": True},
                    "access_group_ids": {"isEmpty": True},
                }
            )
        ),
        teams.count(where={"models": {"isEmpty": True}, "access_group_ids": {"isEmpty": True}}),
        keys.count(
            where=_model_calling_keys(
                {
                    "OR": [{"team_id": None}, {"team_id": {"not": UI_SESSION_TOKEN_TEAM_ID}}],
                    "models": {"isEmpty": True},
                    "NOT": {"access_group_ids": {"isEmpty": True}},
                }
            )
        ),
        teams.count(where={"models": {"isEmpty": True}, "NOT": {"access_group_ids": {"isEmpty": True}}}),
    )
    return LegacyEmptyModelsCounts(
        teamless_keys_with_empty_models=teamless_empty,
        teamless_keys_with_all_team_models=teamless_all_team,
        team_keys_with_empty_models=team_keys_empty,
        teams_with_empty_models=teams_empty,
        keys_with_access_groups_and_empty_models=keys_with_groups_empty,
        teams_with_access_groups_and_empty_models=teams_with_groups_empty,
    )


def _counted(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def legacy_empty_models_warning(counts: LegacyEmptyModelsCounts) -> str | None:
    findings: Final = tuple(
        f"{_counted(count, singular, plural)}, which means {meaning}."
        for count, singular, plural, meaning in (
            (
                counts.teamless_keys_with_empty_models,
                "key without a team has an empty models list",
                "keys without a team have an empty models list",
                "every proxy model",
            ),
            (
                counts.teamless_keys_with_all_team_models,
                "key without a team lists all-team-models",
                "keys without a team list all-team-models",
                "every proxy model",
            ),
            (
                counts.team_keys_with_empty_models,
                "team key has an empty models list",
                "team keys have an empty models list",
                "all of the team's models",
            ),
            (
                counts.teams_with_empty_models,
                "team has an empty models list",
                "teams have an empty models list",
                "every proxy model",
            ),
        )
        if count > 0
    )
    access_group_findings: Final = tuple(
        f"{_counted(count, singular, plural)}."
        for count, singular, plural in (
            (
                counts.keys_with_access_groups_and_empty_models,
                "key with access groups has an empty models list",
                "keys with access groups have an empty models list",
            ),
            (
                counts.teams_with_access_groups_and_empty_models,
                "team with access groups has an empty models list",
                "teams with access groups have an empty models list",
            ),
        )
        if count > 0
    )
    if not findings and not access_group_findings:
        return None
    all_proxy_models: Final = SpecialModelNames.all_proxy_models.value
    all_team_models: Final = SpecialModelNames.all_team_models.value
    keep_access_advice: Final = (
        (
            f"To keep today's access, set models to {all_proxy_models} or a list of models on each one, "
            f"or to {all_team_models} on keys that belong to a team, before you upgrade.",
        )
        if findings
        else ()
    )
    teamless_all_team_models_advice: Final = (
        (
            f"Keys without a team that list {all_team_models} have no team to inherit from, "
            f"so set each one to {all_proxy_models} or attach it to a team.",
        )
        if counts.teamless_keys_with_all_team_models > 0
        else ()
    )
    access_group_advice: Final = (
        ("Check which models each key and team with access groups should reach before you upgrade.",)
        if access_group_findings
        else ()
    )
    return " ".join(
        (
            "Deprecated model access defaults are still in use.",
            *findings,
            *access_group_findings,
            "These counts include blocked and expired keys and blocked teams.",
            "LiteLLM 2.0.0 removes these defaults.",
            *keep_access_advice,
            *teamless_all_team_models_advice,
            *access_group_advice,
            "Keys and teams created without a models list still get an empty one until then, "
            "so these counts can grow back.",
            f"See {DEPRECATION_DISCUSSION_URL}",
        )
    )


async def warn_about_legacy_empty_models(keys: CountTableRepository, teams: CountTableRepository) -> None:
    try:
        counts: Final = await count_legacy_empty_models_rows(keys=keys.table, teams=teams.table)
    except Exception as e:  # noqa: BLE001  # a startup check must never take the proxy down on a DB error
        verbose_proxy_logger.warning("Skipped the deprecated model access defaults check: %s", e)
        return
    message: Final = legacy_empty_models_warning(counts)
    if message is not None:
        verbose_proxy_logger.warning(message)
