import pytest

from litellm.proxy._types import LitellmUserRoles

from .actors import ORG_A, ORG_B
from .conftest import MASTER_KEY, SCRATCH_PREFIX, create_scratch_actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# The minting tests run in file order, then _b runs after their fixture
# teardown and asserts no scratch row survived in any reclaimed table. A leak
# in either direction fails _b on the next collection.


async def test_a_scratch_key_lands_in_db(proxy_client, prisma, scratch):
    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        json={"key_alias": scratch.prefix},
    )
    assert resp.status_code == 200, resp.text

    rows = await prisma.db.litellm_verificationtoken.find_many(
        where={"key_alias": scratch.prefix}
    )
    assert len(rows) == 1


async def test_a2_scratch_actor_lands_in_db(proxy_client, prisma, scratch):
    actor = await create_scratch_actor(
        prisma,
        scratch.prefix,
        user_role=LitellmUserRoles.ORG_ADMIN.value,
        org_admin_of=(ORG_A, ORG_B),
    )
    user_row = await prisma.db.litellm_usertable.find_unique(
        where={"user_id": actor.user_id}
    )
    assert user_row is not None
    info = await proxy_client.get(
        "/key/info", headers={"Authorization": f"Bearer {actor.cleartext}"}
    )
    assert info.status_code == 200, info.text
    memberships = await prisma.db.litellm_organizationmembership.find_many(
        where={"user_id": actor.user_id}
    )
    assert {m.organization_id for m in memberships} == {ORG_A, ORG_B}


async def test_b_scratch_namespace_is_clean(prisma):
    tokens = await prisma.db.litellm_verificationtoken.find_many(
        where={"key_alias": {"startswith": SCRATCH_PREFIX}}
    )
    users = await prisma.db.litellm_usertable.find_many(
        where={"user_id": {"startswith": SCRATCH_PREFIX}}
    )
    memberships = await prisma.db.litellm_organizationmembership.find_many(
        where={"user_id": {"startswith": SCRATCH_PREFIX}}
    )
    assert tokens == [] and users == [] and memberships == []
