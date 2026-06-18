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
from typing import Dict, List

import pytest

from spend_e2e_client import SpendE2EClient

pytestmark = pytest.mark.e2e

# Verified present and responsive on a live proxy. One per row of the spend
# surface: key / user / team / org / customer aggregation, model-cost, tags,
# activity.
SPEND_ROUTES = (
    "/spend/keys",
    "/spend/users",
    "/spend/tags",
    "/spend/logs",
    "/spend/logs/ui",
    "/global/spend",
    "/global/spend/keys",
    "/global/spend/teams",
    "/global/spend/models",
    "/global/spend/provider",
    "/global/spend/report",
    "/global/spend/tags",
    "/global/spend/logs",
    "/global/spend/all_tag_names",
    "/global/activity",
    "/global/activity/model",
    "/global/activity/exceptions",
    "/key/list",
    "/user/list",
    "/team/list",
    "/organization/list",
    "/customer/list",
)

_SPEND_PREFIXES = ("/spend", "/global/spend", "/global/activity")


def _default_params() -> Dict[str, str]:
    # Satisfies date-required endpoints (report/activity/provider); ignored elsewhere.
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=1)
    return {"start_date": start.isoformat(), "end_date": end.isoformat()}


@pytest.mark.parametrize("route", SPEND_ROUTES)
def test_spend_route_responsive(client: SpendE2EClient, route: str) -> None:
    result = client.probe(route, params=_default_params())
    print(result)  # shown on failure, and for all routes under `-rA` / `-s`
    assert result.healthy, str(result)


def test_schema_listed_spend_routes_are_responsive(client: SpendE2EClient) -> None:
    """Probe any spend GET route the schema lists that isn't in SPEND_ROUTES."""
    paths = client.get_openapi().get("paths", {})
    assert isinstance(paths, dict) and paths, "/openapi.json had no paths"

    discovered: List[str] = [
        path
        for path, operations in paths.items()
        if isinstance(operations, dict)
        and "get" in {m.lower() for m in operations}
        and "{" not in path
        and any(path.startswith(prefix) for prefix in _SPEND_PREFIXES)
    ]
    extras = [path for path in discovered if path not in SPEND_ROUTES]

    params = _default_params()
    results = [client.probe(path, params=params) for path in extras]
    for result in results:
        print(result)
    offenders = [str(result) for result in results if not result.healthy]
    assert not offenders, "non-responsive schema spend routes:\n" + "\n".join(offenders)
