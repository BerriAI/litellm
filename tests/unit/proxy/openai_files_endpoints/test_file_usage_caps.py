from typing import Final

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.openai_files_endpoints.file_usage_caps import (
    FileUsageLimit,
    ScopedFileUsageLimit,
    batch_file_record_limit,
    consume_file_usage,
    enforce_batch_file_upload_limit,
    enforce_file_download_limit,
    resolve_scoped_limits,
)
from litellm.proxy.utils import InternalUsageCache

DAY: Final = 86400
MIDDAY: Final = 20_000 * DAY + DAY / 2


def _cache() -> InternalUsageCache:
    return InternalUsageCache(dual_cache=DualCache())


def _scoped(scope, scope_id, value, source="key", setting="max_batch_file_uploads_per_day"):
    return ScopedFileUsageLimit(
        scope=scope, scope_id=scope_id, limit=FileUsageLimit(setting=setting, value=value, source=source)
    )


@pytest.mark.parametrize(
    "key_metadata, team_metadata, general_settings, expected",
    [
        ({}, {}, {}, None),
        ({}, {}, {"max_batch_file_records": 50}, FileUsageLimit("max_batch_file_records", 50, "general_settings")),
        (
            {"max_batch_file_records": 80},
            {},
            {"max_batch_file_records": 50},
            FileUsageLimit("max_batch_file_records", 80, "key"),
        ),
        (
            {"max_batch_file_records": 80},
            {"max_batch_file_records": 30},
            {},
            FileUsageLimit("max_batch_file_records", 30, "team"),
        ),
        (
            {},
            {"max_batch_file_records": 90},
            {"max_batch_file_records": 50},
            FileUsageLimit("max_batch_file_records", 50, "general_settings"),
        ),
        ({"max_batch_file_records": 0}, {"max_batch_file_records": "lots"}, {}, None),
        ({"max_batch_file_records": "40"}, {}, {}, FileUsageLimit("max_batch_file_records", 40, "key")),
    ],
)
def test_record_limit_is_the_lowest_applicable_limit(key_metadata, team_metadata, general_settings, expected):
    caller: Final = UserAPIKeyAuth(api_key="hashed", team_id="t1", metadata=key_metadata, team_metadata=team_metadata)
    assert batch_file_record_limit(caller, general_settings) == expected


@pytest.mark.parametrize(
    "caller, general_settings, expected",
    [
        (UserAPIKeyAuth(api_key="hashed", team_id="t1"), {}, ()),
        (
            UserAPIKeyAuth(api_key="hashed", team_id="t1"),
            {"max_batch_file_uploads_per_day": 5},
            (_scoped("key", "hashed", 5, "general_settings"),),
        ),
        (
            UserAPIKeyAuth(
                api_key="hashed",
                team_id="t1",
                metadata={"max_batch_file_uploads_per_day": 9},
                team_metadata={"max_batch_file_uploads_per_day": 20},
            ),
            {"max_batch_file_uploads_per_day": 5},
            (_scoped("key", "hashed", 9, "key"), _scoped("team", "t1", 20, "team")),
        ),
        (
            UserAPIKeyAuth(api_key="hashed", team_metadata={"max_batch_file_uploads_per_day": 20}),
            {},
            (),
        ),
    ],
)
def test_counter_scopes_pair_each_limit_with_its_own_counter(caller, general_settings, expected):
    assert resolve_scoped_limits(caller, general_settings, "max_batch_file_uploads_per_day") == expected


async def test_a_scope_admits_exactly_its_limit_per_window_and_resets_on_the_next():
    cache: Final = _cache()
    limits: Final = (_scoped("key", "hashed", 2),)

    outcomes: Final = [await consume_file_usage(cache, limits, DAY, "", MIDDAY + i) for i in range(3)]
    next_window: Final = await consume_file_usage(cache, limits, DAY, "", MIDDAY + DAY)

    assert outcomes[:2] == [None, None]
    assert outcomes[2] is not None
    assert outcomes[2].limit == limits[0]
    assert outcomes[2].retry_after_seconds == DAY / 2 - 2
    assert next_window is None


async def test_a_rejection_on_one_scope_gives_back_the_slot_it_took_on_the_other():
    cache: Final = _cache()
    key_scope: Final = _scoped("key", "hashed", 3)
    team_scope: Final = _scoped("team", "t1", 1, "team")

    first: Final = await consume_file_usage(cache, (key_scope, team_scope), DAY, "", MIDDAY)
    rejected: Final = [await consume_file_usage(cache, (key_scope, team_scope), DAY, "", MIDDAY) for _ in range(5)]
    key_only: Final = [await consume_file_usage(cache, (key_scope,), DAY, "", MIDDAY) for _ in range(3)]

    assert first is None
    assert {outcome.limit for outcome in rejected if outcome is not None} == {team_scope}
    assert len([outcome for outcome in rejected if outcome is not None]) == 5
    assert key_only[:2] == [None, None]
    assert key_only[2] is not None and key_only[2].limit == key_scope


async def test_a_rejection_does_not_use_up_the_scope_that_rejected_it():
    cache: Final = _cache()
    limits: Final = (_scoped("key", "hashed", 1),)

    await consume_file_usage(cache, limits, DAY, "", MIDDAY)
    for _ in range(4):
        await consume_file_usage(cache, limits, DAY, "", MIDDAY)
    raised_limit: Final = (_scoped("key", "hashed", 2),)

    assert await consume_file_usage(cache, raised_limit, DAY, "", MIDDAY) is None


async def test_download_counters_are_per_file():
    cache: Final = _cache()
    limits: Final = (_scoped("key", "hashed", 1, setting="max_file_downloads_per_minute"),)

    assert await consume_file_usage(cache, limits, 60, "file-a", MIDDAY) is None
    assert await consume_file_usage(cache, limits, 60, "file-a", MIDDAY) is not None
    assert await consume_file_usage(cache, limits, 60, "file-b", MIDDAY) is None


async def test_upload_limit_error_names_the_setting_value_scope_and_reset():
    cache: Final = _cache()
    caller: Final = UserAPIKeyAuth(api_key="hashed", team_id="t1", team_metadata={"max_batch_file_uploads_per_day": 1})

    await enforce_batch_file_upload_limit(cache, caller, {}, clock=lambda: MIDDAY)
    with pytest.raises(ProxyException) as exc:
        await enforce_batch_file_upload_limit(cache, caller, {}, clock=lambda: MIDDAY + 100)

    assert exc.value.code == "429"
    assert exc.value.type == "rate_limit_error"
    assert exc.value.headers == {"retry-after": str(DAY // 2 - 100)}
    assert "max_batch_file_uploads_per_day is 1 for team t1" in exc.value.message
    assert "this team's metadata" in exc.value.message


async def test_download_limit_error_names_the_file_and_general_settings_default():
    cache: Final = _cache()
    caller: Final = UserAPIKeyAuth(api_key="hashed")
    general_settings: Final = {"max_file_downloads_per_minute": 2}

    for _ in range(2):
        await enforce_file_download_limit(cache, caller, general_settings, "file-a", clock=lambda: MIDDAY + 15)
    with pytest.raises(ProxyException) as exc:
        await enforce_file_download_limit(cache, caller, general_settings, "file-a", clock=lambda: MIDDAY + 15)

    assert exc.value.code == "429"
    assert exc.value.headers == {"retry-after": "45"}
    assert "file-a" in exc.value.message
    assert "max_file_downloads_per_minute is 2 for this key (set in general_settings)" in exc.value.message


async def test_no_configured_limit_never_rejects():
    cache: Final = _cache()
    caller: Final = UserAPIKeyAuth(api_key="hashed", team_id="t1")

    for _ in range(50):
        await enforce_batch_file_upload_limit(cache, caller, {}, clock=lambda: MIDDAY)
        await enforce_file_download_limit(cache, caller, {}, "file-a", clock=lambda: MIDDAY)
