import pytest


pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_key_generate_lands_in_db(gateway_client, db, resources):
    from litellm.proxy.utils import hash_token

    resp = await gateway_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {gateway_client.gateway.key}"},
        json={"key_alias": resources.prefix},
    )
    assert resp.status_code == 200, resp.text
    cleartext = resp.json()["key"]
    assert cleartext.startswith("sk-")

    hashed = hash_token(cleartext)
    row = await db.db.litellm_verificationtoken.find_unique(where={"token": hashed})
    assert row is not None
    assert row.token == hashed != cleartext
