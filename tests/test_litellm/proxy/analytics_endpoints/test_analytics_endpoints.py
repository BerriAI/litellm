from typing import TypedDict, cast

import pytest

from litellm.proxy.analytics_endpoints.analytics_endpoints import get_global_activity


class CacheActivityRow(TypedDict):
    api_key: str
    call_type: str
    model: str
    total_rows: int
    cache_hit_true_rows: int
    cached_completion_tokens: int
    generated_completion_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_token_rows: int
    cache_activity_rows: int


class _FakeDB:
    def __init__(self):
        self.sql = ""

    async def query_raw(self, sql: str, *_args: object) -> list[CacheActivityRow]:
        self.sql = sql
        if "cache_read_input_tokens" not in sql:
            raise AssertionError("cache_read_input_tokens missing from cache dashboard query")
        if "cache_creation_input_tokens" not in sql:
            raise AssertionError("cache_creation_input_tokens missing from cache dashboard query")
        if "cache_read_input_token_rows" not in sql:
            raise AssertionError("cache_read_input_token_rows missing from cache dashboard query")
        if "cache_activity_rows" not in sql:
            raise AssertionError("cache_activity_rows missing from cache dashboard query")
        return [
            {
                "api_key": "sk-test",
                "call_type": "aresponses",
                "model": "bedrock_mantle/openai.gpt-5.5",
                "total_rows": 52,
                "cache_hit_true_rows": 0,
                "cached_completion_tokens": 0,
                "generated_completion_tokens": 100,
                "cache_read_input_tokens": 86941,
                "cache_creation_input_tokens": 0,
                "cache_read_input_token_rows": 42,
                "cache_activity_rows": 42,
            }
        ]


class _FakePrismaClient:
    def __init__(self):
        self.db = _FakeDB()


@pytest.mark.asyncio
async def test_cache_hits_activity_includes_provider_prompt_cache_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_prisma = _FakePrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", fake_prisma)

    result = cast(
        list[CacheActivityRow],
        await get_global_activity(start_date="2026-07-02", end_date="2026-07-03"),
    )

    assert result[0]["cache_read_input_tokens"] == 86941
    assert result[0]["cache_creation_input_tokens"] == 0
    assert result[0]["cache_read_input_token_rows"] == 42
    assert result[0]["cache_activity_rows"] == 42
