import asyncio
import time
from collections.abc import Sequence
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.errors import PrismaError

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import SPEND_LOG_KEY_METADATA_CACHE_TTL, SPEND_LOG_KEY_METADATA_MISS_CACHE_TTL
from litellm.proxy.spend_tracking.key_metadata_recovery import (
    fill_missing_api_key_aliases,
    recover_double_hashed_key_metadata,
    recover_key_metadata_from_spend_logs,
)
from litellm.proxy.utils import hash_token


def _digest_row(digest: str, key_alias: str | None, team_id: str | None, user_id: str | None) -> dict[str, str | None]:
    return {"digest": digest, "key_alias": key_alias, "team_id": team_id, "user_id": user_id}


def _query_raw_spend_logs(rows: Sequence[dict[str, str | None]]) -> AsyncMock:
    async def query_raw(sql: str, *params: object) -> list[dict[str, str | None]]:
        if '"LiteLLM_SpendLogs"' in sql:
            return list(rows)
        raise AssertionError(f"unexpected query: {sql}")

    return AsyncMock(side_effect=query_raw)


def _query_raw_by_table(
    active_rows: Sequence[dict[str, str | None]],
    deleted_rows: Sequence[dict[str, str | None]],
) -> AsyncMock:
    async def query_raw(sql: str, *params: object) -> list[dict[str, str | None]]:
        if '"LiteLLM_VerificationToken"' in sql:
            return list(active_rows)
        if '"LiteLLM_DeletedVerificationToken"' in sql:
            return list(deleted_rows)
        raise AssertionError(f"unexpected query: {sql}")

    return AsyncMock(side_effect=query_raw)


@pytest.mark.asyncio
async def test_recover_double_hashed_key_metadata_via_active_token_digest():
    double_hashed = hash_token("a" * 64)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_by_table(
        active_rows=[_digest_row(double_hashed, "batch-worker", "team-1", "alice")],
        deleted_rows=[],
    )

    result = await recover_double_hashed_key_metadata(mock_prisma, {double_hashed})

    assert result[double_hashed]["key_alias"] == "batch-worker"
    assert result[double_hashed]["team_id"] == "team-1"
    assert result[double_hashed]["user_id"] == "alice"
    ((_, digests),) = [call.args for call in mock_prisma.db.query_raw.call_args_list]
    assert digests == [double_hashed]


@pytest.mark.asyncio
async def test_recover_double_hashed_key_metadata_falls_back_to_deleted_tokens():
    double_hashed = hash_token("y" * 64)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_by_table(
        active_rows=[],
        deleted_rows=[_digest_row(double_hashed, "deleted-key", "team-del", "erin")],
    )

    result = await recover_double_hashed_key_metadata(mock_prisma, {double_hashed})

    assert result[double_hashed]["key_alias"] == "deleted-key"
    assert result[double_hashed]["team_id"] == "team-del"
    assert result[double_hashed]["user_id"] == "erin"
    assert [call.args[1] for call in mock_prisma.db.query_raw.call_args_list] == [[double_hashed], [double_hashed]]


@pytest.mark.asyncio
async def test_recover_only_asks_deleted_tokens_for_digests_active_keys_missed():
    found_active = hash_token("1" * 64)
    found_deleted = hash_token("2" * 64)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_by_table(
        active_rows=[_digest_row(found_active, "active-key", None, None)],
        deleted_rows=[_digest_row(found_deleted, "deleted-key", None, None)],
    )

    result = await recover_double_hashed_key_metadata(mock_prisma, {found_active, found_deleted})

    assert result[found_active]["key_alias"] == "active-key"
    assert result[found_deleted]["key_alias"] == "deleted-key"
    assert [call.args[1] for call in mock_prisma.db.query_raw.call_args_list] == [
        sorted((found_active, found_deleted)),
        [found_deleted],
    ]


@pytest.mark.asyncio
async def test_recover_permanent_miss_costs_two_digest_lookups_and_no_table_walk():
    double_hashed = hash_token("b" * 64)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_by_table(active_rows=[], deleted_rows=[])

    result = await recover_double_hashed_key_metadata(mock_prisma, {double_hashed})

    assert result == {}
    assert len(mock_prisma.db.query_raw.call_args_list) == 2
    mock_prisma.db.litellm_verificationtoken.find_many.assert_not_called()
    mock_prisma.db.litellm_deletedverificationtoken.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_recover_skips_keys_that_are_not_sha256_digests():
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(return_value=[])

    result = await recover_double_hashed_key_metadata(mock_prisma, {"sk-plain-key", "key-hash-short"})

    assert result == {}
    mock_prisma.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_recover_returns_empty_when_digest_lookup_raises_prisma_error():
    double_hashed = hash_token("c" * 64)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(side_effect=PrismaError("db down"))

    result = await recover_double_hashed_key_metadata(mock_prisma, {double_hashed})

    assert result == {}


@pytest.mark.asyncio
async def test_fill_missing_api_key_aliases_updates_null_alias_and_email_rows():
    double_hashed = hash_token("d" * 64)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_by_table(
        active_rows=[_digest_row(double_hashed, "recovered-alias", "team-9", "bob")],
        deleted_rows=[],
    )
    mock_prisma.db.litellm_usertable.find_many = AsyncMock(
        return_value=[SimpleNamespace(user_id="bob", user_email="bob@example.com")]
    )

    rows = (
        {
            "api_key": double_hashed,
            "api_key_alias": None,
            "team_id": None,
            "user_email": None,
            "spend": 12.5,
        },
        {
            "api_key": "already-joined-token",
            "api_key_alias": "named-key",
            "team_id": "team-ok",
            "user_email": "other@example.com",
            "spend": 1.0,
        },
    )

    filled = await fill_missing_api_key_aliases(mock_prisma, rows)

    assert filled[0]["api_key_alias"] == "recovered-alias"
    assert filled[0]["team_id"] == "team-9"
    assert filled[0]["user_email"] == "bob@example.com"
    assert filled[0]["spend"] == 12.5
    assert filled[1]["api_key_alias"] == "named-key"
    assert mock_prisma.db.litellm_usertable.find_many.call_args.kwargs["where"] == {"user_id": {"in": ["bob"]}}


@pytest.mark.asyncio
async def test_fill_missing_api_key_aliases_leaves_rows_untouched_when_nothing_is_missing():
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(return_value=[])
    rows = ({"api_key": hash_token("e" * 64), "api_key_alias": "named", "user_email": "x@example.com"},)

    filled = await fill_missing_api_key_aliases(mock_prisma, rows)

    assert filled == rows
    mock_prisma.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_fill_missing_api_key_aliases_keeps_spend_user_email_when_alias_is_missing():
    double_hashed = hash_token("f" * 64)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_by_table(
        active_rows=[_digest_row(double_hashed, "team-key", "team-9", "key-owner")],
        deleted_rows=[],
    )
    mock_prisma.db.litellm_usertable.find_many = AsyncMock(
        return_value=[SimpleNamespace(user_id="key-owner", user_email="owner@example.com")]
    )

    rows = (
        {
            "api_key": double_hashed,
            "api_key_alias": None,
            "team_id": None,
            "user_email": "spender@example.com",
            "spend": 4.0,
        },
    )

    filled = await fill_missing_api_key_aliases(mock_prisma, rows)

    assert filled[0]["api_key_alias"] == "team-key"
    assert filled[0]["team_id"] == "team-9"
    assert filled[0]["user_email"] == "spender@example.com"


@pytest.mark.asyncio
async def test_fill_missing_api_key_aliases_skips_named_keys_that_have_no_email():
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(return_value=[])
    rows = (
        {
            "api_key": hash_token("g" * 64),
            "api_key_alias": "service-key",
            "team_id": "team-svc",
            "user_email": None,
        },
    )

    filled = await fill_missing_api_key_aliases(mock_prisma, rows)

    assert filled == rows
    mock_prisma.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_resolves_session_token_from_metadata():
    session_digest = hash_token("cli-session-repro-user-6852")
    window = (datetime(2026, 9, 7), datetime(2026, 9, 10))
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_spend_logs(
        [_digest_row(session_digest, "cli-session-repro-user-6852", None, "repro-user-6852")]
    )

    result = await recover_key_metadata_from_spend_logs(mock_prisma, {session_digest}, window, cache=InMemoryCache())

    assert result[session_digest]["key_alias"] == "cli-session-repro-user-6852"
    assert result[session_digest]["user_id"] == "repro-user-6852"
    ((_, digests, start, end),) = [call.args for call in mock_prisma.db.query_raw.call_args_list]
    assert digests == [session_digest]
    assert (start, end) == window


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_skips_query_when_no_missing_keys():
    window = (datetime(2026, 9, 7), datetime(2026, 9, 10))
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(return_value=[])

    result = await recover_key_metadata_from_spend_logs(mock_prisma, set(), window, cache=InMemoryCache())

    assert result == {}
    mock_prisma.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_returns_empty_on_prisma_error():
    window = (datetime(2026, 9, 7), datetime(2026, 9, 10))
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(side_effect=PrismaError("db down"))

    result = await recover_key_metadata_from_spend_logs(
        mock_prisma, {hash_token("cli-session-x")}, window, cache=InMemoryCache()
    )

    assert result == {}


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_ignores_foreign_and_all_null_rows():
    wanted = hash_token("cli-session-wanted")
    all_null = hash_token("cli-session-null")
    foreign = hash_token("cli-session-foreign")
    window = (datetime(2026, 9, 7), datetime(2026, 9, 10))
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_spend_logs(
        [
            _digest_row(wanted, "kept-alias", None, "owner-1"),
            _digest_row(all_null, None, None, None),
            _digest_row(foreign, "foreign-alias", None, "owner-2"),
        ]
    )

    result = await recover_key_metadata_from_spend_logs(mock_prisma, {wanted, all_null}, window, cache=InMemoryCache())

    assert set(result) == {wanted}
    assert result[wanted]["key_alias"] == "kept-alias"
    assert result[wanted]["user_id"] == "owner-1"


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_skips_non_sha256_keys():
    window = (datetime(2026, 9, 7), datetime(2026, 9, 10))
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(return_value=[])

    result = await recover_key_metadata_from_spend_logs(
        mock_prisma, {"cli-session-raw-1798", "key-hash-short"}, window, cache=InMemoryCache()
    )

    assert result == {}
    mock_prisma.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_accepts_hashed_jwt_digests():
    jwt_digest = f"hashed-jwt-{hash_token('jwt-subject-1')}"
    window = (datetime(2026, 9, 7), datetime(2026, 9, 10))
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_spend_logs([_digest_row(jwt_digest, None, "team-jwt", "jwt-user")])

    result = await recover_key_metadata_from_spend_logs(mock_prisma, {jwt_digest}, window, cache=InMemoryCache())

    assert result[jwt_digest]["team_id"] == "team-jwt"
    assert result[jwt_digest]["user_id"] == "jwt-user"
    ((_, digests, _, _),) = [call.args for call in mock_prisma.db.query_raw.call_args_list]
    assert digests == [jwt_digest]


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_serves_repeat_lookups_from_the_cache():
    found = hash_token("cli-session-found")
    unknown = hash_token("cli-session-unknown")
    window = (datetime(2026, 9, 7), datetime(2026, 9, 10))
    cache = InMemoryCache()
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_spend_logs([_digest_row(found, "found-alias", None, "owner-1")])

    first = await recover_key_metadata_from_spend_logs(mock_prisma, {found, unknown}, window, cache=cache)
    second = await recover_key_metadata_from_spend_logs(mock_prisma, {found, unknown}, window, cache=cache)

    assert first == second
    assert set(first) == {found}
    assert first[found]["key_alias"] == "found-alias"
    assert mock_prisma.db.query_raw.await_count == 1


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_only_queries_digests_the_cache_has_not_seen():
    cached_digest = hash_token("cli-session-cached")
    new_digest = hash_token("cli-session-new")
    window = (datetime(2026, 9, 7), datetime(2026, 9, 10))
    cache = InMemoryCache()
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_spend_logs([_digest_row(cached_digest, "cached-alias", None, None)])
    await recover_key_metadata_from_spend_logs(mock_prisma, {cached_digest}, window, cache=cache)
    mock_prisma.db.query_raw = _query_raw_spend_logs([_digest_row(new_digest, "new-alias", None, None)])

    result = await recover_key_metadata_from_spend_logs(mock_prisma, {cached_digest, new_digest}, window, cache=cache)

    assert result[cached_digest]["key_alias"] == "cached-alias"
    assert result[new_digest]["key_alias"] == "new-alias"
    ((_, digests, _, _),) = [call.args for call in mock_prisma.db.query_raw.call_args_list]
    assert digests == [new_digest]


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_rescans_when_the_window_changes():
    digest = hash_token("cli-session-windowed")
    cache = InMemoryCache()
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_spend_logs([])
    await recover_key_metadata_from_spend_logs(
        mock_prisma, {digest}, (datetime(2026, 9, 1), datetime(2026, 9, 4)), cache=cache
    )
    mock_prisma.db.query_raw = _query_raw_spend_logs([_digest_row(digest, "later-alias", None, None)])

    result = await recover_key_metadata_from_spend_logs(
        mock_prisma, {digest}, (datetime(2026, 9, 7), datetime(2026, 9, 10)), cache=cache
    )

    assert result[digest]["key_alias"] == "later-alias"
    assert mock_prisma.db.query_raw.await_count == 1


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_does_not_cache_a_failed_query():
    digest = hash_token("cli-session-retry")
    window = (datetime(2026, 9, 7), datetime(2026, 9, 10))
    cache = InMemoryCache()
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(side_effect=PrismaError("db down"))
    assert await recover_key_metadata_from_spend_logs(mock_prisma, {digest}, window, cache=cache) == {}
    mock_prisma.db.query_raw = _query_raw_spend_logs([_digest_row(digest, "back-online", None, None)])

    result = await recover_key_metadata_from_spend_logs(mock_prisma, {digest}, window, cache=cache)

    assert result[digest]["key_alias"] == "back-online"


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_forgets_a_miss_long_before_a_hit():
    found = hash_token("cli-session-found")
    unknown = hash_token("cli-session-unknown")
    window = (datetime(2026, 9, 7), datetime(2026, 9, 10))
    cache = InMemoryCache(default_ttl=SPEND_LOG_KEY_METADATA_CACHE_TTL)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_spend_logs([_digest_row(found, "found-alias", None, None)])
    started = time.time()

    await recover_key_metadata_from_spend_logs(mock_prisma, {found, unknown}, window, cache=cache)

    hit_expires = next(deadline for key, deadline in cache.ttl_dict.items() if found in key)
    miss_expires = next(deadline for key, deadline in cache.ttl_dict.items() if unknown in key)
    assert miss_expires - started <= SPEND_LOG_KEY_METADATA_MISS_CACHE_TTL + 1
    assert hit_expires - started >= SPEND_LOG_KEY_METADATA_CACHE_TTL - 1


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_runs_one_query_for_concurrent_lookups():
    digest = hash_token("cli-session-shared")
    window = (datetime(2026, 9, 7), datetime(2026, 9, 10))
    cache = InMemoryCache()
    lock = asyncio.Lock()
    mock_prisma = MagicMock()

    async def slow_query_raw(sql: str, *params: object) -> list[dict[str, str | None]]:
        await asyncio.sleep(0.01)
        return [_digest_row(digest, "shared-alias", None, None)]

    mock_prisma.db.query_raw = AsyncMock(side_effect=slow_query_raw)

    results = await asyncio.gather(
        *(
            recover_key_metadata_from_spend_logs(mock_prisma, {digest}, window, cache=cache, lock=lock)
            for _ in range(9)
        )
    )

    assert all(result[digest]["key_alias"] == "shared-alias" for result in results)
    assert mock_prisma.db.query_raw.await_count == 1


@pytest.mark.asyncio
async def test_recover_key_metadata_from_spend_logs_keeps_a_repeated_miss_as_long_as_a_hit():
    unknown = hash_token("cli-session-never-named")
    window = (datetime(2026, 9, 7), datetime(2026, 9, 10))
    cache = InMemoryCache(default_ttl=SPEND_LOG_KEY_METADATA_CACHE_TTL)
    mock_prisma = MagicMock()
    mock_prisma.db.query_raw = _query_raw_spend_logs([])
    await recover_key_metadata_from_spend_logs(mock_prisma, {unknown}, window, cache=cache)
    first_miss_key = next(key for key in cache.ttl_dict if unknown in key and not key.endswith(":missed-before"))
    cache.ttl_dict[first_miss_key] = time.time() - 1
    started = time.time()

    await recover_key_metadata_from_spend_logs(mock_prisma, {unknown}, window, cache=cache)

    assert mock_prisma.db.query_raw.await_count == 2
    assert cache.ttl_dict[first_miss_key] - started >= SPEND_LOG_KEY_METADATA_CACHE_TTL - 1
