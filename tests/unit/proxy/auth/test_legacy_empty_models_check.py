import logging
from collections.abc import Iterator, Mapping
from typing import Final

import pytest

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.legacy_empty_models_check import (
    DEPRECATION_DISCUSSION_URL,
    LegacyEmptyModelsCounts,
    legacy_empty_models_warning,
    warn_about_legacy_empty_models,
)

NOTHING_ON_THE_DEFAULTS: Final = LegacyEmptyModelsCounts(
    teamless_keys_with_empty_models=0,
    teamless_keys_with_all_team_models=0,
    team_keys_with_empty_models=0,
    teams_with_empty_models=0,
)


def test_no_warning_when_nothing_relies_on_the_deprecated_defaults() -> None:
    assert legacy_empty_models_warning(NOTHING_ON_THE_DEFAULTS) is None


@pytest.mark.parametrize(
    ("counts", "reported", "left_out"),
    [
        (
            LegacyEmptyModelsCounts(3, 0, 0, 0),
            "3 keys without a team have an empty models list, which grants every proxy model.",
            ("all-team-models, which", "team key", "team has", "teams have"),
        ),
        (
            LegacyEmptyModelsCounts(0, 1, 0, 0),
            "1 key without a team lists all-team-models, which grants every proxy model.",
            ("empty models list", "team key"),
        ),
        (
            LegacyEmptyModelsCounts(0, 0, 4, 0),
            "4 team keys have an empty models list, which grants all of the team's models.",
            ("without a team", "teams have"),
        ),
        (
            LegacyEmptyModelsCounts(0, 0, 0, 1),
            "1 team has an empty models list, which grants every proxy model.",
            ("without a team", "team key"),
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


def test_warning_reports_every_category_in_use_at_once() -> None:
    message: Final = legacy_empty_models_warning(LegacyEmptyModelsCounts(2, 5, 7, 11))

    assert message is not None
    assert "2 keys without a team have an empty models list" in message
    assert "5 keys without a team list all-team-models" in message
    assert "7 team keys have an empty models list" in message
    assert "11 teams have an empty models list" in message


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


@pytest.mark.asyncio
async def test_boot_check_logs_the_warning_when_rows_rely_on_the_defaults(proxy_warnings: list[str]) -> None:
    await warn_about_legacy_empty_models(keys=_FixedCountTable(2), teams=_FixedCountTable(1))

    assert len(proxy_warnings) == 1
    assert proxy_warnings[0].startswith("Deprecated model access defaults are still in use.")
    assert "1 team has an empty models list" in proxy_warnings[0]


@pytest.mark.asyncio
async def test_boot_check_logs_nothing_when_no_rows_rely_on_the_defaults(proxy_warnings: list[str]) -> None:
    await warn_about_legacy_empty_models(keys=_FixedCountTable(0), teams=_FixedCountTable(0))

    assert proxy_warnings == []


@pytest.mark.asyncio
async def test_a_database_error_skips_the_boot_check_with_a_warning(proxy_warnings: list[str]) -> None:
    await warn_about_legacy_empty_models(keys=_UnreachableTable(), teams=_UnreachableTable())

    assert proxy_warnings == ["Skipped the deprecated model access defaults check: database is down"]
