from datetime import datetime, timezone
from typing import Final

import pytest

from litellm.models.budget_window_spend import BudgetWindowSpend
from litellm.repositories.budget_window_spend_repository import BudgetWindowSpendRepository
from tests.unit.proxy.rust_auth_stores import install_rust_auth_stores

WINDOW: Final = ("model_max_budget", "virtual_key_spend:key-hash:claude-opus*:1d", "1d")
READER_ROW: Final = BudgetWindowSpend(window_start=datetime(2026, 11, 1, 9, 30, tzinfo=timezone.utc), spend=2.5)
WRITER_ROW: Final = BudgetWindowSpend(window_start=datetime(2026, 11, 2, 9, 30, tzinfo=timezone.utc), spend=0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("use_writer", "expected"), [(False, READER_ROW), (True, WRITER_ROW)], ids=["reader", "writer"]
)
async def test_find_reads_the_window_from_the_side_asked_for(
    monkeypatch: pytest.MonkeyPatch, use_writer: bool, expected: BudgetWindowSpend
) -> None:
    stores: Final = install_rust_auth_stores(monkeypatch)
    stores.budget_window_spend.reader[WINDOW] = READER_ROW
    stores.budget_window_spend.writer[WINDOW] = WRITER_ROW

    assert await BudgetWindowSpendRepository(prisma_client=None).find(*WINDOW, use_writer=use_writer) == expected


@pytest.mark.asyncio
async def test_find_answers_none_for_a_window_never_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    install_rust_auth_stores(monkeypatch)

    assert await BudgetWindowSpendRepository(prisma_client=None).find("key", "tok-1", "30d") is None
