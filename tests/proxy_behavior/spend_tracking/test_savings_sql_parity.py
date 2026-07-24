"""
The hourly savings endpoint reads saved-token counts out of spend-log metadata
in Postgres, while the daily rollup writer reads the same fields in Python. Both
readers must agree, or the hourly chart will not add up to the daily numbers it
is drawn against. These tests run each SQL expression and its Python twin over
the same metadata shapes against a real Postgres and demand identical answers.
"""

import os

import pytest
import pytest_asyncio
from prisma import Prisma

from litellm.proxy.spend_tracking.cache_savings import (
    CACHE_READ_INPUT_TOKENS_SQL,
    extract_cache_read_tokens,
)
from litellm.proxy.spend_tracking.compression_savings import (
    COMPRESSION_SAVED_TOKENS_SQL,
    extract_compression_saved_tokens,
)

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.skipif(
        not os.environ.get("DATABASE_URL"),
        reason="SQL/Python parity needs a live Postgres",
    ),
]

COMPRESSION_SHAPES = (
    ("native_only", {"compression_savings": {"tokens_saved": 120}}),
    (
        "headroom_array",
        {"guardrail_information": [{"guardrail_provider": "headroom", "guardrail_response": {"tokens_saved": 50}}]},
    ),
    (
        "headroom_bare_dict",
        {"guardrail_information": {"guardrail_provider": "headroom", "guardrail_response": {"tokens_saved": 7}}},
    ),
    (
        "native_and_headroom_are_additive",
        {
            "compression_savings": {"tokens_saved": 10},
            "guardrail_information": [{"guardrail_provider": "headroom", "guardrail_response": {"tokens_saved": 5}}],
        },
    ),
    (
        "other_guardrails_ignored",
        {"guardrail_information": [{"guardrail_provider": "presidio", "guardrail_response": {"tokens_saved": 999}}]},
    ),
    (
        "multiple_headroom_entries",
        {
            "guardrail_information": [
                {"guardrail_provider": "headroom", "guardrail_response": {"tokens_saved": 3}},
                {"guardrail_provider": "headroom", "guardrail_response": {"tokens_saved": 4}},
            ]
        },
    ),
    (
        "redacted_guardrail_response",
        {"guardrail_information": [{"guardrail_provider": "headroom", "guardrail_response": "REDACTED"}]},
    ),
    ("negative_clamped", {"compression_savings": {"tokens_saved": -30}}),
    ("boolean_is_not_a_token_count", {"compression_savings": {"tokens_saved": True}}),
    ("string_is_not_a_token_count", {"compression_savings": {"tokens_saved": "40"}}),
    ("float_truncated", {"compression_savings": {"tokens_saved": 9.9}}),
    ("guardrail_information_null", {"guardrail_information": None}),
    ("empty", {}),
)

CACHE_READ_SHAPES = (
    ("anthropic_top_level", {"cache_read_input_tokens": 80, "prompt_tokens_details": {"cached_tokens": 5}}),
    ("openai_fallback", {"cache_read_input_tokens": 0, "prompt_tokens_details": {"cached_tokens": 22016}}),
    ("openai_only", {"prompt_tokens_details": {"cached_tokens": 640}}),
    ("both_zero", {"cache_read_input_tokens": 0, "prompt_tokens_details": {"cached_tokens": 0}}),
    ("null_explicit_falls_through", {"cache_read_input_tokens": None, "prompt_tokens_details": {"cached_tokens": 12}}),
    ("boolean_falls_through", {"cache_read_input_tokens": True, "prompt_tokens_details": {"cached_tokens": 12}}),
    ("string_is_not_a_token_count", {"cache_read_input_tokens": "80"}),
    ("float_truncated", {"cache_read_input_tokens": 7.9}),
    ("details_not_an_object", {"prompt_tokens_details": "REDACTED"}),
    ("empty", {}),
)


@pytest_asyncio.fixture(scope="session")
async def db():
    client = Prisma()
    await client.connect()
    yield client
    await client.disconnect()


async def _sql_result(db: Prisma, expression: str, metadata: dict) -> int:
    rows = await db.query_raw(
        f"SELECT ({expression})::bigint AS result FROM (SELECT $1::jsonb AS metadata) sl",
        metadata,
    )
    return rows[0]["result"]


@pytest.mark.parametrize("name,metadata", COMPRESSION_SHAPES, ids=[s[0] for s in COMPRESSION_SHAPES])
async def test_compression_sql_matches_python(db, name, metadata):
    assert await _sql_result(db, COMPRESSION_SAVED_TOKENS_SQL, metadata) == extract_compression_saved_tokens(metadata)


@pytest.mark.parametrize("name,usage_object", CACHE_READ_SHAPES, ids=[s[0] for s in CACHE_READ_SHAPES])
async def test_cache_read_sql_matches_python(db, name, usage_object):
    result = await _sql_result(db, CACHE_READ_INPUT_TOKENS_SQL, {"usage_object": usage_object})
    assert result == extract_cache_read_tokens(usage_object)


async def test_shapes_would_catch_a_drifting_reader():
    """A parity suite where every shape reads zero on both sides proves nothing."""
    compression = [extract_compression_saved_tokens(m) for _, m in COMPRESSION_SHAPES]
    cache_read = [extract_cache_read_tokens(u) for _, u in CACHE_READ_SHAPES]
    assert len([v for v in compression if v > 0]) >= 5
    assert len([v for v in cache_read if v > 0]) >= 5
