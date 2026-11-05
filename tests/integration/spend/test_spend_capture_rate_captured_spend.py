from datetime import date
from typing import Final

import pytest

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.spend_capture_rate import captured_spend_by_day
from litellm.proxy.utils import PrismaClient, ProxyLogging
from tests.integration._support.database import scratch_database, write_rows

_DAILY_USER_SPEND_DDL: Final = """
    CREATE TABLE "LiteLLM_DailyUserSpend" (
        id TEXT PRIMARY KEY,
        date TEXT NOT NULL,
        custom_llm_provider TEXT,
        spend DOUBLE PRECISION DEFAULT 0
    )
"""


@pytest.mark.asyncio
async def test_captured_spend_sums_only_the_openai_billed_providers_inside_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_database() as database_url:
        monkeypatch.setenv("DATABASE_URL", database_url)
        write_rows(_DAILY_USER_SPEND_DDL, (), database_url=database_url)
        for index, (day, provider, spend) in enumerate(
            (
                ("2026-09-19", "openai", 1.0),
                ("2026-09-20", "openai", 2.0),
                ("2026-09-20", "openai", 3.0),
                ("2026-09-20", "text-completion-openai", 0.5),
                ("2026-09-20", "anthropic", 100.0),
                ("2026-09-21", "azure", 100.0),
                ("2026-09-22", "openai", 4.0),
            )
        ):
            write_rows(
                'INSERT INTO "LiteLLM_DailyUserSpend" (id, date, custom_llm_provider, spend) VALUES (%s, %s, %s, %s)',
                (f"row-{index}", day, provider, str(spend)),
                database_url=database_url,
            )
        client: Final = PrismaClient(database_url, ProxyLogging(UserApiKeyCache()))
        await client.connect()
        try:
            captured: Final = await captured_spend_by_day(
                client,
                litellm_providers=("openai", "text-completion-openai"),
                start_date=date(2026, 9, 20),
                end_date=date(2026, 9, 21),
            )
        finally:
            await client.disconnect()

    assert dict(captured) == {"2026-09-20": 5.5}
