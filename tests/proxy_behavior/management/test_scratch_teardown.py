import pytest

from .conftest import MASTER_KEY, SCRATCH_PREFIX

pytestmark = pytest.mark.asyncio(loop_scope="session")


# The two tests run in file order: _a writes a scratch-tagged key and asserts
# it lands; _b runs after _a's fixture teardown and asserts no scratch row
# survived. A leak in either direction fails _b on the next collection.


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


async def test_b_scratch_namespace_is_clean(prisma):
    rows = await prisma.db.litellm_verificationtoken.find_many(
        where={"key_alias": {"startswith": SCRATCH_PREFIX}}
    )
    assert rows == []
