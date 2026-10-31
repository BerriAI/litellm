from datetime import datetime, timezone
from itertools import chain
from typing import Final
from uuid import uuid4

import pytest

from litellm.litellm_core_utils.duration_parser import duration_in_seconds_from
from tests.integration._support.client import Gateway, Scenario


@pytest.mark.parametrize("descending", (False, True))
def test_budget_reset_sort_orders_durations_before_pagination(
    descending: bool,
    gateway: Gateway,
    scenario: Scenario,
) -> None:
    prefix: Final = f"sort-{uuid4().hex}"
    durations: Final = ("30d", "7d", "1h", "60m", "1mo", "weekly", "monthly", "٥d", "۵d", "५d", "５d", "𝟝d", "5d", None)
    ids: Final = tuple(
        scenario.budget(budget_id=f"{prefix}-{index}", budget_duration=duration, max_budget=10)
        for index, duration in enumerate(durations)
    )
    now: Final = datetime.now(timezone.utc)
    expected: Final = tuple(
        budget_id
        for _, budget_id in sorted(
            zip(durations, ids, strict=True),
            key=lambda item: (
                item[0] is None,
                duration_in_seconds_from(now, item[0]) * (-1 if descending else 1) if item[0] else 0,
                item[1],
            ),
        )
    )
    pages: Final = tuple(
        gateway.request(
            "GET",
            "/management/v1/budgets",
            params={
                "q": prefix,
                "sort": "-budget_duration" if descending else "budget_duration",
                "page_size": "3",
                "page": str(page),
            },
        )
        for page in range(1, -(-len(ids) // 3) + 1)
    )
    assert all(page.status_code == 200 for page in pages), [page.text for page in pages]
    assert tuple(row["budget_id"] for row in chain.from_iterable(page.json()["data"] for page in pages)) == expected
    filtered: Final = gateway.request(
        "GET",
        "/management/v1/budgets",
        params={
            "q": prefix,
            "sort": "budget_duration",
            "filter[budget_duration][in]": "7d,30d",
        },
    )
    assert filtered.status_code == 200, filtered.text
    assert [row["budget_duration"] for row in filtered.json()["data"]] == ["7d", "30d"]


@pytest.mark.parametrize("field", ("max_budget", "tpm_limit", "rpm_limit", "tpd_limit"))
@pytest.mark.parametrize("descending", (False, True))
def test_budget_limits_sort_unlimited_as_infinity(
    field: str, descending: bool, gateway: Gateway, scenario: Scenario
) -> None:
    prefix: Final = f"sort-unlimited-{uuid4().hex}"
    values: Final = (20, None, 0, 3)
    ids: Final = tuple(scenario.budget(budget_id=f"{prefix}-{i}", **{field: value}) for i, value in enumerate(values))
    expected: Final = tuple(
        identity
        for value, identity in sorted(
            zip(values, ids, strict=True),
            key=lambda item: float("inf") if item[0] is None else item[0],
            reverse=descending,
        )
    )
    pages: Final = tuple(
        gateway.request(
            "GET",
            "/management/v1/budgets",
            params={
                "q": prefix,
                "sort": ("-" if descending else "") + field,
                "page_size": "2",
                "page": str(page),
            },
        )
        for page in (1, 2)
    )
    assert all(page.status_code == 200 for page in pages), [page.text for page in pages]
    assert tuple(row["budget_id"] for row in chain.from_iterable(page.json()["data"] for page in pages)) == expected
