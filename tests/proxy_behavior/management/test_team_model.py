import pytest

from .actors import Actor
from .conftest import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")

_MARKER_MODEL = "behavior-pin-team-model-marker"
_ROUTE_URL = {"add": "/team/model/add", "delete": "/team/model/delete"}


# POST /team/model/add + /team/model/delete. The handler gate is PROXY_ADMIN
# or team admin or org admin, but the management-route gate fronts it — these
# are neither internal_user nor org-admin nor info routes, so every
# non-proxy-admin is 401 before the handler runs. Only PROXY_ADMIN reaches the
# handler, making the team-admin / org-admin handler branches unreachable here.
_MATRIX = [
    ("proxy_admin", Actor.PROXY_ADMIN, 200),
    ("org_admin", Actor.ORG_ADMIN, 401),
    ("team_admin", Actor.TEAM_ADMIN, 401),
    ("internal_user", Actor.INTERNAL_USER, 401),
    ("owner", Actor.OWNER, 401),
    ("unrelated_same_org", Actor.UNRELATED_SAME_ORG, 401),
    ("cross_org_user", Actor.CROSS_ORG_USER, 401),
    ("service_account", Actor.SERVICE_ACCOUNT, 401),
    ("org_b_admin", Actor.ORG_B_ADMIN, 401),
]


@pytest.mark.parametrize("route", ["add", "delete"])
@pytest.mark.parametrize(
    "actor,expected_status",
    [(a, s) for (_id, a, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_team_model_authz_matrix(
    route: str,
    actor: Actor,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    initial = [] if route == "add" else [_MARKER_MODEL]
    await create_scratch_team(
        prisma, scratch.prefix, organization_id=world.org_a_id, models=initial
    )
    caller = world.keys[actor]

    resp = await proxy_client.post(
        _ROUTE_URL[route],
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"team_id": scratch.prefix, "models": [_MARKER_MODEL]},
    )
    assert (
        resp.status_code == expected_status
    ), f"{route} {actor.value}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is not None
    if expected_status == 200:
        assert (_MARKER_MODEL in row.models) is (route == "add")
    else:
        assert list(row.models) == initial, "denied but models mutated"


@pytest.mark.parametrize("route", ["add", "delete"])
async def test_team_model_missing_team_is_404(route: str, proxy_client, world):
    """A team_id absent from the DB is 404 — the existence check precedes authz."""
    resp = await proxy_client.post(
        _ROUTE_URL[route],
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"team_id": "behavior-pin-no-such-team", "models": [_MARKER_MODEL]},
    )
    assert resp.status_code == 404, resp.text
