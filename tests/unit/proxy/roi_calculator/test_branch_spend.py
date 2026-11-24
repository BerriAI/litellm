import json
from datetime import date
from typing import Final

import pytest

from litellm.proxy.roi_calculator.branch_spend import read_branch_spend
from litellm.types.roi_calculator import ROIBranchSpend


class _SpendDatabase:
    async def query_raw(self, query: str, *args: object) -> object:
        assert args == (
            "2026-01-31T00:00:00+00:00",
            "2026-02-01T00:00:00+00:00",
            json.dumps(("gitlab.com/group/project",)),
            False,
        )
        return [{"repo": "gitlab.com/group/project", "branch": "feature", "spend": 0.000027, "requests": 3}]


@pytest.mark.asyncio
async def test_branch_spend_includes_the_final_utc_day_and_preserves_fractional_costs() -> None:
    result: Final = await read_branch_spend(
        _SpendDatabase(), date(2026, 1, 31), date(2026, 1, 31), ("gitlab.com/group/project",)
    )
    assert result == (ROIBranchSpend(repo="gitlab.com/group/project", branch="feature", spend=0.000027, requests=3),)


@pytest.mark.asyncio
async def test_no_repositories_returns_no_spend_without_querying_the_database() -> None:
    assert await read_branch_spend(_SpendDatabase(), date(2026, 1, 1), date(2026, 1, 31), ()) == ()
