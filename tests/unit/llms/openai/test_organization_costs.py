from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import Final

import httpx
import pytest

from litellm.constants import OPENAI_ORGANIZATION_COSTS_URL
from litellm.llms.openai.organization_costs import OpenAICostsRequestFailed, fetch_openai_daily_costs

_ADMIN_KEY: Final = "sk-admin-test"


def _utc_midnight(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


def _bucket(day: str, *amounts: float) -> dict[str, object]:
    return {
        "object": "bucket",
        "start_time": _utc_midnight(day),
        "end_time": _utc_midnight(day) + 86400,
        "results": [
            {"object": "organization.costs.result", "amount": {"value": amount, "currency": "usd"}, "line_item": None}
            for amount in amounts
        ],
    }


class _FakeCostsApi:
    """Serves ``pages`` in order and records every request it saw."""

    def __init__(self, *pages: dict[str, object] | httpx.Response | Exception) -> None:
        self._pages = list(pages)
        self.calls: list[tuple[str, Mapping[str, object], Mapping[str, str]]] = []

    async def __call__(self, url: str, params: Mapping[str, object], headers: Mapping[str, str]) -> httpx.Response:
        self.calls.append((url, dict(params), dict(headers)))
        page = self._pages.pop(0)
        if isinstance(page, Exception):
            raise page
        if isinstance(page, httpx.Response):
            return page
        return httpx.Response(200, json=page)


def _page(*buckets: dict[str, object], next_page: str | None = None) -> dict[str, object]:
    return {"object": "page", "data": list(buckets), "has_more": next_page is not None, "next_page": next_page}


@pytest.mark.asyncio
async def test_openai_costs_are_summed_per_utc_day_across_pages_and_line_items():
    api = _FakeCostsApi(
        _page(_bucket("2026-09-20", 10.0, 2.5), _bucket("2026-09-21", 4.0), next_page="page_2"),
        _page(_bucket("2026-09-22", 1.0)),
    )

    billed = await fetch_openai_daily_costs(date(2026, 9, 20), date(2026, 9, 22), admin_key=_ADMIN_KEY, http_get=api)

    assert dict(billed) == {"2026-09-20": 12.5, "2026-09-21": 4.0, "2026-09-22": 1.0}
    first, second = api.calls
    assert first[0] == OPENAI_ORGANIZATION_COSTS_URL
    assert first[2] == {"Authorization": f"Bearer {_ADMIN_KEY}"}
    assert first[1]["start_time"] == _utc_midnight("2026-09-20")
    assert first[1]["end_time"] == _utc_midnight("2026-09-23")
    assert first[1]["bucket_width"] == "1d"
    assert "page" not in first[1]
    assert "project_ids[]" not in first[1]
    assert second[1] == {**first[1], "page": "page_2"}


@pytest.mark.asyncio
async def test_openai_costs_are_scoped_to_the_configured_projects():
    api = _FakeCostsApi(_page())

    billed = await fetch_openai_daily_costs(
        date(2026, 9, 20), date(2026, 9, 20), admin_key=_ADMIN_KEY, project_ids=("proj_a", "proj_b"), http_get=api
    )

    assert dict(billed) == {}
    assert api.calls[0][1]["project_ids[]"] == ("proj_a", "proj_b")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "page, detail_fragment",
    [
        (httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}}), "HTTP 401"),
        (httpx.ConnectError("connection refused"), "request failed"),
        ({"object": "page", "data": [{"start_time": "not-a-timestamp"}]}, "unexpected response shape"),
    ],
)
async def test_an_unreadable_openai_bill_is_a_request_failure_not_an_exception(page, detail_fragment):
    api = _FakeCostsApi(page)

    billed = await fetch_openai_daily_costs(date(2026, 9, 20), date(2026, 9, 20), admin_key=_ADMIN_KEY, http_get=api)

    assert isinstance(billed, OpenAICostsRequestFailed)
    assert detail_fragment in billed.detail


@pytest.mark.asyncio
async def test_a_failure_on_a_later_page_fails_the_whole_read():
    api = _FakeCostsApi(
        _page(_bucket("2026-09-20", 10.0), next_page="page_2"),
        httpx.Response(429, json={"error": {"message": "rate limited"}}),
    )

    billed = await fetch_openai_daily_costs(date(2026, 9, 20), date(2026, 9, 21), admin_key=_ADMIN_KEY, http_get=api)

    assert isinstance(billed, OpenAICostsRequestFailed)
    assert "HTTP 429" in billed.detail
