from datetime import date

import httpx

from litellm.proxy.control_plane.auth.rbac import USAGE_VIEWER_ROLE
from litellm.proxy.control_plane.usage.service import MAX_PAGES
from tests.unit.proxy.control_plane.support import MASTER_KEY, response, rig

EU = "https://eu.test/user/daily/activity"
US = "https://us.test/user/daily/activity"
WINDOW = {"start_date": "2026-10-01", "end_date": "2026-10-02"}


def _page(rows: list[dict[str, object]], has_more: bool = False) -> dict[str, object]:
    return {"results": rows, "metadata": {"has_more": has_more, "page": 1, "total_pages": 2 if has_more else 1}}


def _row(day: str, spend: float, models: dict[str, float] | None = None) -> dict[str, object]:
    return {
        "date": day,
        "metrics": {"spend": spend, "api_requests": 1},
        "breakdown": {"models": {name: {"metrics": {"spend": cost}} for name, cost in (models or {}).items()}},
    }


def test_summary_adds_both_dataplanes_and_forwards_the_window_and_caller() -> None:
    client, transport, minter = rig(
        {
            EU: [response(200, _page([_row("2026-10-01", 1.5), _row("2026-10-02", 0.5)]))],
            US: [response(200, _page([_row("2026-10-01", 2.0)]))],
        }
    )
    body = client.get("/api/v1/usage/summary", params=WINDOW, headers=minter.headers(role=USAGE_VIEWER_ROLE)).json()
    assert body["window"] == WINDOW
    assert (body["totals"]["spend"], body["totals"]["api_requests"]) == (4.0, 3)
    assert [(dp["id"], dp["metrics"]["spend"]) for dp in body["by_dataplane"]] == [("eu", 2.0), ("us", 2.0)]
    assert body["partial"] is False and body["unreachable"] == []
    url, params, headers = transport.calls[0]
    assert params == {**WINDOW, "page": "1", "page_size": "1000"}
    assert headers is not None and headers["X-LiteLLM-Impersonate-User"] == "alice"


def test_summary_pages_through_a_dataplane_until_has_more_is_false() -> None:
    client, transport, minter = rig(
        {
            EU: [
                response(200, _page([_row("2026-10-01", 1.0)], has_more=True)),
                response(200, _page([_row("2026-10-02", 1.0)])),
            ],
            US: [response(200, _page([]))],
        }
    )
    body = client.get("/api/v1/usage/summary", params=WINDOW, headers=minter.headers()).json()
    assert body["totals"]["spend"] == 2.0
    eu_pages = [params["page"] for url, params, _ in transport.calls if url == EU and params]
    assert eu_pages == ["1", "2"]


def test_one_dataplane_down_gives_partial_numbers_and_names_it() -> None:
    client, _, minter = rig(
        {EU: [response(200, _page([_row("2026-10-01", 1.0)]))], US: [httpx.ConnectError("refused")] * 3}
    )
    body = client.get("/api/v1/usage/summary", params=WINDOW, headers=minter.headers()).json()
    assert body["totals"]["spend"] == 1.0 and body["partial"] is True
    assert [u["id"] for u in body["unreachable"]] == ["us"]
    assert "refused" in body["unreachable"][0]["reason"]


def test_every_dataplane_down_is_a_503_problem() -> None:
    client, _, minter = rig({EU: [response(503, "maintenance")] * 3, US: [httpx.ReadTimeout("slow")] * 3})
    down = client.get("/api/v1/usage/summary", params=WINDOW, headers=minter.headers())
    assert down.status_code == 503
    assert down.json()["type"] == "urn:litellm:error:service-unavailable"
    assert "eu:" in down.json()["detail"] and "us:" in down.json()["detail"]


def test_binding_and_dataplane_filter_narrow_the_fan_out() -> None:
    client, transport, minter = rig(
        {
            EU: [response(200, _page([_row("2026-10-01", 1.0)]))] * 2,
            US: [response(200, _page([_row("2026-10-01", 9.0)]))],
        }
    )
    bound = client.get("/api/v1/usage/summary", params=WINDOW, headers=minter.headers("eu")).json()
    assert [dp["id"] for dp in bound["by_dataplane"]] == ["eu"]
    filtered = client.get(
        "/api/v1/usage/summary", params={**WINDOW, "dataplane": "eu"}, headers={"Authorization": f"Bearer {MASTER_KEY}"}
    ).json()
    assert [dp["id"] for dp in filtered["by_dataplane"]] == ["eu"]
    assert all(url == EU for url, _, _ in transport.calls)
    assert (
        client.get(
            "/api/v1/usage/summary", params={**WINDOW, "dataplane": "us"}, headers=minter.headers("eu")
        ).status_code
        == 403
    )


def test_daily_groups_by_model_across_dataplanes_and_top_ranks_them() -> None:
    client, _, minter = rig(
        {
            EU: [response(200, _page([_row("2026-10-01", 3.0, {"gpt-4o": 2.0, "claude": 1.0})]))] * 2,
            US: [response(200, _page([_row("2026-10-01", 5.0, {"gpt-4o": 5.0})]))] * 2,
        }
    )
    daily = client.get("/api/v1/usage/daily", params={**WINDOW, "group_by": "model"}, headers=minter.headers()).json()
    assert [(row["date"], row["group"], row["metrics"]["spend"]) for row in daily["rows"]] == [
        ("2026-10-01", "claude", 1.0),
        ("2026-10-01", "gpt-4o", 7.0),
    ]
    top = client.get("/api/v1/usage/top", params={**WINDOW, "limit": 1}, headers=minter.headers()).json()
    assert top["group_by"] == "model" and top["limit"] == 1
    assert [(item["group"], item["metrics"]["spend"]) for item in top["items"]] == [("gpt-4o", 7.0)]


def test_usage_routes_need_the_usage_read_scope_and_default_the_window_to_a_month() -> None:
    client, transport, minter = rig({EU: [response(200, _page([]))], US: [response(200, _page([]))]})
    assert client.get("/api/v1/usage/daily", params=WINDOW).status_code == 401
    body = client.get("/api/v1/usage/summary", headers=minter.headers()).json()
    start, end = date.fromisoformat(body["window"]["start_date"]), date.fromisoformat(body["window"]["end_date"])
    assert (end - start).days == 29


def test_a_dataplane_that_still_has_more_after_the_page_cap_is_reported_not_truncated() -> None:
    client, transport, minter = rig(
        {
            EU: [response(200, _page([_row("2026-10-01", 1.0)], has_more=True)) for _ in range(MAX_PAGES)],
            US: [response(200, _page([_row("2026-10-01", 2.0)]))],
        }
    )
    body = client.get("/api/v1/usage/summary", params=WINDOW, headers=minter.headers()).json()
    assert body["totals"]["spend"] == 2.0, "eu's incomplete pages must not count as final numbers"
    assert body["partial"] is True and [u["id"] for u in body["unreachable"]] == ["eu"]
    assert "pages" in body["unreachable"][0]["reason"]
    assert sum(1 for url, _, _ in transport.calls if url == EU) == MAX_PAGES


def test_reversed_window_is_a_400_before_any_dataplane_is_asked() -> None:
    client, transport, minter = rig({EU: [], US: []})
    reversed_window = client.get(
        "/api/v1/usage/summary",
        params={"start_date": "2026-10-03", "end_date": "2026-10-01"},
        headers=minter.headers(),
    )
    assert reversed_window.status_code == 400 and "after end_date" in reversed_window.json()["detail"]
    assert transport.calls == []
