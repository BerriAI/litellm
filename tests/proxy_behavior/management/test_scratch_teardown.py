"""Slice 5 smoke: scratch namespace fixture cleans up after itself.

Two ordered tests. The first writes a key tagged with the scratch prefix and
asserts it lands. The second runs after the first's teardown and asserts no
scratch-namespaced rows survived. Together they prove the per-test cleanup
filter is the right shape — any leaked row will surface as a test-2 failure on
the very next run.
"""

import pytest

from .conftest import MASTER_KEY, SCRATCH_PREFIX

pytestmark = pytest.mark.asyncio(loop_scope="session")


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
    assert (
        len(rows) == 1
    ), f"expected exactly one scratch-tagged token, found {len(rows)}"


async def test_b_scratch_namespace_is_clean(prisma):
    """Runs after test_a's teardown — proves nothing leaked."""
    rows = await prisma.db.litellm_verificationtoken.find_many(
        where={"key_alias": {"startswith": SCRATCH_PREFIX}}
    )
    assert rows == [], (
        f"scratch teardown leaked {len(rows)} rows; first key_alias: "
        f"{rows[0].key_alias if rows else None!r}"
    )
