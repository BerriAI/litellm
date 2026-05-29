import pytest

from .conftest import MASTER_KEY

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_liveliness(proxy_client):
    resp = await proxy_client.get("/health/liveliness")
    assert resp.status_code == 200


async def test_key_generate_lands_in_db(proxy_client, prisma, scratch):
    from litellm.proxy.utils import hash_token

    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        json={"key_alias": scratch.prefix},
    )
    assert resp.status_code == 200, resp.text
    cleartext = resp.json()["key"]
    assert cleartext.startswith("sk-")

    hashed = hash_token(cleartext)
    row = await prisma.db.litellm_verificationtoken.find_unique(where={"token": hashed})
    assert row is not None
    assert row.token == hashed != cleartext
