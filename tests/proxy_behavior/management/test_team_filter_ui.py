import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# GET /team/filter/ui (ui_view_teams) — include_in_schema=False. The handler
# body has no role/org check and never reads user_api_key_dict, but the
# endpoint is still effectively PROXY-ADMIN-only as its docstring claims: the
# management-route gate fronts it (not an internal_user / info / org-admin
# route) and 401s every non-proxy-admin before the handler runs. PROXY_ADMIN
# reaches the unscoped find_many and sees teams across every org.
@pytest.mark.parametrize("actor", list(Actor), ids=[a.value for a in Actor])
async def test_team_filter_ui_is_proxy_admin_only(actor: Actor, proxy_client, world):
    resp = await proxy_client.get(
        "/team/filter/ui",
        headers={"Authorization": f"Bearer {world.keys[actor].cleartext}"},
    )
    expected = 200 if actor == Actor.PROXY_ADMIN else 401
    assert (
        resp.status_code == expected
    ), f"{actor.value}: {resp.status_code} {resp.text}"


async def test_team_filter_ui_proxy_admin_sees_cross_org_teams(proxy_client, world):
    """The handler runs an unscoped query — PROXY_ADMIN sees teams from every
    org, including the three seeded world teams."""
    resp = await proxy_client.get(
        "/team/filter/ui",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
    )
    assert resp.status_code == 200, resp.text
    team_ids = {t.get("team_id") for t in resp.json() if isinstance(t, dict)}
    assert {
        world.team_alpha_id,
        world.team_beta_id,
        world.team_gamma_id,
    } <= team_ids
