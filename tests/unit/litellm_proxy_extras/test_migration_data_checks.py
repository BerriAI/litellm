from pathlib import Path
from typing import Final

import pytest
from litellm_proxy_extras.migration_data_checks import (
    DEFAULT_MIGRATION_DATA_CHECK_TIMEOUT_SECONDS,
    MIGRATION_DATA_CHECK_TIMEOUT_ENV_VAR,
    BlockedChange,
    Check,
    ColumnType,
    CouldNotVerify,
    _count_query,
    btree_max_entry_bytes,
    checks_for_index,
    migration_data_check_timeout_seconds,
    report,
)


def _shape(checks: tuple[Check, ...]) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    return tuple((check.kind, check.table, check.columns) for check in checks)


def test_a_request_log_index_gets_one_entry_size_check_over_all_its_columns() -> None:
    assert _shape(
        checks_for_index("LiteLLM_SpendLogs", "LiteLLM_SpendLogs_api_key_startTime_idx", ("api_key", "startTime"))
    ) == (("too_long", "LiteLLM_SpendLogs", ("api_key", "startTime")),)


def test_the_btree_entry_limit_on_the_default_page_is_the_number_in_postgres_error() -> None:
    # "index row size 2976 exceeds btree version 4 maximum 2704", Postgres 14, 8 KB page (BTMaxItemSize, nbtree.h)
    assert btree_max_entry_bytes(8192) == 2704


def test_the_entry_size_query_sizes_a_text_column_by_its_stored_bytes_when_postgres_compressed_it() -> None:
    """Postgres compresses a long value before it checks the entry size, so a repetitive
    3000-byte key indexes fine. The query has to size it as stored, never by its length alone."""
    (check,) = checks_for_index(
        "LiteLLM_SpendLogs", "LiteLLM_SpendLogs_api_key_startTime_idx", ("api_key", "startTime")
    )
    types: Final = (ColumnType("text", -1, "i", -1), ColumnType("timestamp", 8, "d", 3))
    query: Final = _count_query(check, "public", types, 2704).as_string(None)
    stored: Final = 'least(octet_length("api_key"), pg_column_size("api_key"))'
    assert stored in query
    assert 'octet_length("api_key")' not in query.replace(stored, ""), query


def test_the_report_names_the_rows_the_statement_and_the_fix_and_says_nothing_was_applied() -> None:
    (check,) = checks_for_index(
        "LiteLLM_SpendLogs", "LiteLLM_SpendLogs_api_key_startTime_idx", ("api_key", "startTime")
    )
    text: Final = report(
        (BlockedChange(check, 2, "with an index entry over 2704 bytes, largest 3232 bytes (api_key up to 3207 bytes)"),)
    )
    assert (
        '"LiteLLM_SpendLogs".api_key, startTime: 2 row(s) with an index entry over 2704 bytes, largest 3232 bytes '
        "(api_key up to 3207 bytes)"
    ) in text
    assert 'blocks: CREATE INDEX "LiteLLM_SpendLogs_api_key_startTime_idx" (request-log index)' in text
    assert 'fix:    Shorten or hash the long "LiteLLM_SpendLogs" (api_key, startTime) values' in text
    assert text.startswith("Existing rows would make a pending schema change fail:")
    assert "No index was built" in text


def test_a_scan_that_timed_out_is_reported_as_unverified_with_the_knob_to_turn() -> None:
    (check,) = checks_for_index(
        "LiteLLM_SpendLogs", "LiteLLM_SpendLogs_api_key_startTime_idx", ("api_key", "startTime")
    )
    text: Final = report((CouldNotVerify(check, 600.0),))
    assert (
        '"LiteLLM_SpendLogs".api_key, startTime: could not be checked within 600 s, refusing rather than guessing'
        in text
    )
    assert 'blocks: CREATE INDEX "LiteLLM_SpendLogs_api_key_startTime_idx" (request-log index)' in text
    assert MIGRATION_DATA_CHECK_TIMEOUT_ENV_VAR in text


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        (None, DEFAULT_MIGRATION_DATA_CHECK_TIMEOUT_SECONDS),
        ("30", 30.0),
        ("nope", DEFAULT_MIGRATION_DATA_CHECK_TIMEOUT_SECONDS),
    ),
)
def test_the_scan_timeout_comes_from_the_environment_and_falls_back_on_garbage(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: float
) -> None:
    if raw is None:
        monkeypatch.delenv(MIGRATION_DATA_CHECK_TIMEOUT_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(MIGRATION_DATA_CHECK_TIMEOUT_ENV_VAR, raw)
    assert migration_data_check_timeout_seconds() == expected
