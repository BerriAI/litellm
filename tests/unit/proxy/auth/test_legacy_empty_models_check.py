import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final

import pytest

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.legacy_empty_models_check import (
    DEPRECATION_DISCUSSION_URL,
    CountTableRepository,
    LegacyEmptyModelsCounts,
    legacy_empty_models_warning,
    warn_about_legacy_empty_models,
)

NOTHING_ON_THE_DEFAULTS: Final = LegacyEmptyModelsCounts(
    teamless_keys_with_empty_models=0,
    teamless_keys_with_all_team_models=0,
    team_keys_with_empty_models=0,
    teams_with_empty_models=0,
    keys_with_access_groups_and_empty_models=0,
    teams_with_access_groups_and_empty_models=0,
)


def test_no_warning_when_nothing_relies_on_the_deprecated_defaults() -> None:
    assert legacy_empty_models_warning(NOTHING_ON_THE_DEFAULTS) is None


@pytest.mark.parametrize(
    ("counts", "reported", "left_out"),
    [
        (
            LegacyEmptyModelsCounts(3, 0, 0, 0, 0, 0),
            "3 keys without a team have an empty models list, which means every proxy model.",
            ("all-team-models, which", "team key", "team has", "teams have", "access groups"),
        ),
        (
            LegacyEmptyModelsCounts(0, 1, 0, 0, 0, 0),
            "1 key without a team lists all-team-models, which means every proxy model.",
            ("empty models list", "team key", "access groups"),
        ),
        (
            LegacyEmptyModelsCounts(0, 0, 4, 0, 0, 0),
            "4 team keys have an empty models list, which means all of the team's models.",
            ("without a team", "teams have", "access groups"),
        ),
        (
            LegacyEmptyModelsCounts(0, 0, 0, 1, 0, 0),
            "1 team has an empty models list, which means every proxy model.",
            ("without a team", "team key", "access groups"),
        ),
        (
            LegacyEmptyModelsCounts(0, 0, 0, 0, 2, 0),
            "2 keys with access groups have an empty models list.",
            (
                "which means",
                "without a team",
                "team key",
                "teams with access groups have",
                "team with access groups has",
            ),
        ),
        (
            LegacyEmptyModelsCounts(0, 0, 0, 0, 0, 1),
            "1 team with access groups has an empty models list.",
            ("which means", "without a team", "team key", "keys with access groups have", "key with access groups has"),
        ),
    ],
)
def test_warning_reports_only_the_categories_in_use(
    counts: LegacyEmptyModelsCounts, reported: str, left_out: tuple[str, ...]
) -> None:
    message: Final = legacy_empty_models_warning(counts)

    assert message is not None
    assert reported in message
    assert all(phrase not in message for phrase in left_out), message
    assert DEPRECATION_DISCUSSION_URL in message


KEEP_ACCESS_ADVICE: Final = "To keep today's access, set models to all-proxy-models"
TEAMLESS_ALL_TEAM_MODELS_ADVICE: Final = "Keys without a team that list all-team-models have no team to inherit from"
ACCESS_GROUP_ADVICE: Final = "Check which models each key and team with access groups should reach"


@pytest.mark.parametrize(
    ("counts", "advice"),
    [
        (LegacyEmptyModelsCounts(3, 0, 0, 0, 0, 0), (KEEP_ACCESS_ADVICE,)),
        (LegacyEmptyModelsCounts(0, 0, 4, 0, 0, 0), (KEEP_ACCESS_ADVICE,)),
        (LegacyEmptyModelsCounts(0, 0, 0, 1, 0, 0), (KEEP_ACCESS_ADVICE,)),
        (LegacyEmptyModelsCounts(0, 1, 0, 0, 0, 0), (KEEP_ACCESS_ADVICE, TEAMLESS_ALL_TEAM_MODELS_ADVICE)),
        (LegacyEmptyModelsCounts(0, 0, 0, 0, 2, 0), (ACCESS_GROUP_ADVICE,)),
        (LegacyEmptyModelsCounts(0, 0, 0, 0, 0, 1), (ACCESS_GROUP_ADVICE,)),
    ],
)
def test_warning_gives_only_the_advice_for_the_categories_in_use(
    counts: LegacyEmptyModelsCounts, advice: tuple[str, ...]
) -> None:
    message: Final = legacy_empty_models_warning(counts)

    assert message is not None
    assert {
        phrase
        for phrase in (KEEP_ACCESS_ADVICE, TEAMLESS_ALL_TEAM_MODELS_ADVICE, ACCESS_GROUP_ADVICE)
        if phrase in message
    } == set(advice)


def test_warning_reports_every_category_in_use_at_once() -> None:
    message: Final = legacy_empty_models_warning(LegacyEmptyModelsCounts(2, 5, 7, 11, 13, 17))

    assert message is not None
    assert "2 keys without a team have an empty models list" in message
    assert "5 keys without a team list all-team-models" in message
    assert "7 team keys have an empty models list" in message
    assert "11 teams have an empty models list" in message
    assert "13 keys with access groups have an empty models list" in message
    assert "17 teams with access groups have an empty models list" in message
    assert all(
        phrase in message for phrase in (KEEP_ACCESS_ADVICE, TEAMLESS_ALL_TEAM_MODELS_ADVICE, ACCESS_GROUP_ADVICE)
    )


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def proxy_warnings() -> Iterator[list[str]]:
    handler: Final = _RecordingHandler()
    verbose_proxy_logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        verbose_proxy_logger.removeHandler(handler)


class _FixedCountTable:
    def __init__(self, count: int) -> None:
        self._count = count

    async def count(self, *, where: Mapping[str, object]) -> int:
        return self._count


class _UnreachableTable:
    async def count(self, *, where: Mapping[str, object]) -> int:
        raise ConnectionError("database is down")


@dataclass(frozen=True, slots=True)
class _Repository:
    table: _FixedCountTable | _UnreachableTable


class _UnconnectedRepository:
    @property
    def table(self) -> _FixedCountTable:
        raise RuntimeError("No DB Connected")


def _repository(table: _FixedCountTable | _UnreachableTable) -> CountTableRepository:
    return _Repository(table=table)


@pytest.mark.asyncio
async def test_boot_check_logs_the_warning_when_rows_rely_on_the_defaults(proxy_warnings: list[str]) -> None:
    await warn_about_legacy_empty_models(keys=_repository(_FixedCountTable(2)), teams=_repository(_FixedCountTable(1)))

    assert len(proxy_warnings) == 1
    assert proxy_warnings[0].startswith("Deprecated model access defaults are still in use.")
    assert "1 team has an empty models list" in proxy_warnings[0]


@pytest.mark.asyncio
async def test_boot_check_logs_nothing_when_no_rows_rely_on_the_defaults(proxy_warnings: list[str]) -> None:
    await warn_about_legacy_empty_models(keys=_repository(_FixedCountTable(0)), teams=_repository(_FixedCountTable(0)))

    assert proxy_warnings == []


@pytest.mark.asyncio
async def test_a_database_error_skips_the_boot_check_with_a_warning(proxy_warnings: list[str]) -> None:
    await warn_about_legacy_empty_models(keys=_repository(_UnreachableTable()), teams=_repository(_UnreachableTable()))

    assert proxy_warnings == ["Skipped the deprecated model access defaults check: database is down"]


@pytest.mark.asyncio
async def test_a_repository_without_a_connected_client_skips_the_boot_check_with_a_warning(
    proxy_warnings: list[str],
) -> None:
    await warn_about_legacy_empty_models(keys=_UnconnectedRepository(), teams=_repository(_FixedCountTable(1)))

    assert proxy_warnings == ["Skipped the deprecated model access defaults check: No DB Connected"]
