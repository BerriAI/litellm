"""Breadth check: query every route on the spend read surface and show what it
returns.

Spend tracking sprawls across many routes (model-cost / key / user / team / org /
customer aggregation, tags, and activity reports). Most are served with
`include_in_schema=False`, so they do NOT appear in `/openapi.json` - discovery
from the schema alone misses ~70% of the surface. So we probe a curated, verified
list directly, plus any spend route the schema does list (to auto-catch new ones).

Each probe captures status AND body, so a failure shows the proxy's actual error
(a 500 traceback, a 404 meaning the route was removed) rather than a bare code.
Run with `-rA` (or `-s`) to print every route's response, not just failures.

Healthy == route exists (not 404) and handler did not crash (not 5xx). A 4xx
(missing params / auth nuance) still means the route is wired and ran. Cheap and
fast: no batch-write wait, no provider calls.
"""

from datetime import datetime, timedelta, timezone

import pytest

from models import DateRangeParams
from spend_e2e_client import SpendClient

pytestmark = pytest.mark.e2e

# Verified present and responsive on a live proxy. One per row of the spend
# surface: key / user / team / org / customer aggregation, model-cost, tags,
# activity. Most are include_in_schema=False, so keep this list exhaustive by
# hand; the schema test below only auto-catches the visible minority. Excluded
# by design: path-param routes (/spend/logs/ui/{request_id}), POST readers
# (/spend/calculate has its own test, /global/spend/end_users), mutating
# POSTs (/global/spend/reset, /global/spend/refresh), and /provider/budgets,
# which 500s whenever router_settings.provider_budget_config is absent, so it
# is only probeable on a proxy configured with provider budget routing.
SPEND_ROUTES = (
    "/spend/keys",
    "/spend/users",
    "/spend/tags",
    "/spend/logs",
    "/spend/logs/ui",
    "/spend/logs/v2",
    "/spend/logs/session/ui",
    "/global/spend",
    "/global/spend/keys",
    "/global/spend/teams",
    "/global/spend/models",
    "/global/spend/provider",
    "/global/spend/report",
    "/global/spend/tags",
    "/global/spend/logs",
    "/global/spend/all_tag_names",
    "/global/all_end_users",
    "/global/activity",
    "/global/activity/model",
    "/global/activity/exceptions",
    "/global/activity/exceptions/deployment",
    "/user/daily/activity",
    "/user/daily/activity/aggregated",
    "/team/daily/activity",
    "/organization/daily/activity",
    "/customer/daily/activity",
    "/end_user/daily/activity",
    "/tag/daily/activity",
    "/key/list",
    "/user/list",
    "/team/list",
    "/organization/list",
    "/customer/list",
)

_SPEND_PREFIXES = ("/spend", "/global/spend", "/global/activity")


def _date_range() -> DateRangeParams:
    # Satisfies date-required endpoints (report/activity/provider); ignored elsewhere.
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=1)
    return DateRangeParams(start_date=start.isoformat(), end_date=end.isoformat())


@pytest.mark.parametrize("route", SPEND_ROUTES)
def test_spend_route_responsive(client: SpendClient, route: str) -> None:
    result = client.probe(route, params=_date_range())
    print(f"{route} -> {result.status_code}\n{result.body[:600]}")
    assert result.healthy, f"{route} -> {result.status_code}\n{result.body[:600]}"


def test_schema_listed_spend_routes_are_responsive(client: SpendClient) -> None:
    """Probe any spend GET route the schema lists that isn't in SPEND_ROUTES."""
    schema = client.openapi()
    assert schema.paths, "/openapi.json had no paths"

    discovered = [
        path
        for path, spec in schema.paths.items()
        if "get" in spec.methods
        and "{" not in path
        and any(path.startswith(prefix) for prefix in _SPEND_PREFIXES)
    ]
    extras = [path for path in discovered if path not in SPEND_ROUTES]

    params = _date_range()
    results = [(path, client.probe(path, params=params)) for path in extras]
    for path, result in results:
        print(f"{path} -> {result.status_code}")
    offenders = [
        f"{path} -> {result.status_code}\n{result.body[:600]}"
        for path, result in results
        if not result.healthy
    ]
    assert not offenders, "non-responsive schema spend routes:\n" + "\n".join(offenders)
