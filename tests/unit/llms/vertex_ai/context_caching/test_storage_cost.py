from typing import Final

import pytest

from litellm.llms.vertex_ai.context_caching.storage_cost import (
    context_cache_storage_token_hours,
)


@pytest.mark.parametrize(
    "created, requested_ttl, expected",
    [
        (
            {
                "usageMetadata": {"totalTokenCount": 5000},
                "createTime": "2026-01-01T00:00:00.123456789Z",
                "expireTime": "2026-01-01T02:00:00.123456789Z",
            },
            "60s",
            10000.0,
        ),
        ({"usageMetadata": {"totalTokenCount": 5000}}, "1800s", 2500.0),
        ({"usageMetadata": {"totalTokenCount": 5000}}, "90.5s", 5000 * 90.5 / 3600),
        (
            {"usageMetadata": {"totalTokenCount": 4000}, "expireTime": "2026-01-01T02:00:00Z"},
            "900s",
            1000.0,
        ),
        ({"name": "cachedContents/1"}, "3600s", 0.0),
        ({"usageMetadata": {"totalTokenCount": "many"}}, "3600s", 0.0),
        ({"usageMetadata": {"totalTokenCount": 5000}, "expireTime": "not-a-time"}, "3600s", 0.0),
        ("not-an-object", "3600s", 0.0),
    ],
)
def test_token_hours_follow_reported_lifetime_then_requested_ttl(
    created: object, requested_ttl: str | None, expected: float
) -> None:
    assert context_cache_storage_token_hours(created, requested_ttl) == pytest.approx(expected)


def test_token_hours_default_to_one_hour_without_lifetime_or_ttl() -> None:
    # Source: https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-create
    # "The default expiration time of a context cache is 60 minutes after it's created" (read 2026-10-07)
    assert context_cache_storage_token_hours({"usageMetadata": {"totalTokenCount": 5000}}, None) == 5000.0


@pytest.mark.parametrize("requested_ttl", ["", "s", "abc", "10m", "-5s"])
def test_unparseable_requested_ttl_falls_back_to_default_lifetime(requested_ttl: str) -> None:
    created: Final = {"usageMetadata": {"totalTokenCount": 3600}}

    assert context_cache_storage_token_hours(created, requested_ttl) == context_cache_storage_token_hours(created, None)
