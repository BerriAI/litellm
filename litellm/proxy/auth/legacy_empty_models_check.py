import asyncio
from dataclasses import dataclass
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import SpecialModelNames
from litellm.repositories.prisma_protocols import CountTable

DEPRECATION_DISCUSSION_URL: Final = "https://github.com/BerriAI/litellm/discussions/32090"


@dataclass(frozen=True, slots=True)
class LegacyEmptyModelsCounts:
    teamless_keys_with_empty_models: int
    teamless_keys_with_all_team_models: int
    team_keys_with_empty_models: int
    teams_with_empty_models: int


async def count_legacy_empty_models_rows(keys: CountTable, teams: CountTable) -> LegacyEmptyModelsCounts:
    teamless_empty, teamless_all_team, team_keys_empty, teams_empty = await asyncio.gather(
        keys.count(where={"team_id": None, "models": {"isEmpty": True}}),
        keys.count(where={"team_id": None, "models": {"has": SpecialModelNames.all_team_models.value}}),
        keys.count(
            where={
                "AND": [{"team_id": {"not": None}}, {"team_id": {"not": UI_SESSION_TOKEN_TEAM_ID}}],
                "models": {"isEmpty": True},
            }
        ),
        teams.count(where={"models": {"isEmpty": True}}),
    )
    return LegacyEmptyModelsCounts(
        teamless_keys_with_empty_models=teamless_empty,
        teamless_keys_with_all_team_models=teamless_all_team,
        team_keys_with_empty_models=team_keys_empty,
        teams_with_empty_models=teams_empty,
    )


def _finding(count: int, singular: str, plural: str, grant: str) -> str:
    return f"{count} {singular if count == 1 else plural}, which grants {grant}."


def legacy_empty_models_warning(counts: LegacyEmptyModelsCounts) -> str | None:
    findings: Final = tuple(
        _finding(count, singular, plural, grant)
        for count, singular, plural, grant in (
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
    if not findings:
        return None
    return " ".join(
        (
            "Deprecated model access defaults are still in use.",
            *findings,
            "LiteLLM 2.0.0 removes these defaults and rewrites such keys and teams to all-proxy-models or "
            "all-team-models during the upgrade, so their access stays the same. To set it yourself first, pick "
            f"All Proxy Models, All Team Models, or a list of models on each one. See {DEPRECATION_DISCUSSION_URL}",
        )
    )


async def warn_about_legacy_empty_models(keys: CountTable, teams: CountTable) -> None:
    try:
        counts: Final = await count_legacy_empty_models_rows(keys=keys, teams=teams)
    except Exception as e:  # noqa: BLE001  # a startup check must never take the proxy down on a DB error
        verbose_proxy_logger.warning("Skipped the deprecated model access defaults check: %s", e)
        return
    message: Final = legacy_empty_models_warning(counts)
    if message is not None:
        verbose_proxy_logger.warning(message)
