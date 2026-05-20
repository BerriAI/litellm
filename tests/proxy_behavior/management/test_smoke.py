"""Smoke tests proving the harness boots and talks to the proxy app."""

import pytest

from .conftest import MASTER_KEY

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_liveliness(proxy_client):
    resp = await proxy_client.get("/health/liveliness")
    assert resp.status_code == 200


async def test_key_generate_lands_in_db(proxy_client):
    """De-risk gate: prove the harness exercises the full stack end-to-end.

    A successful ``/key/generate`` requires:
      * the FastAPI lifespan ran (``proxy_startup_event``),
      * ``prisma_client`` connected,
      * ``user_api_key_auth`` accepted the master key,
      * the real ``generate_key_helper_fn`` wrote a hashed row to
        ``LiteLLM_VerificationToken``.

    All four collapse to a single 200 + ``sk-`` token check here, with a
    follow-up prisma read to prove the row landed (and that the token is the
    hashed form, not the cleartext returned over the wire).
    """
    from litellm.proxy import proxy_server
    from litellm.proxy.utils import hash_token

    assert (
        proxy_server.prisma_client is not None
    ), "FastAPI lifespan did not connect prisma — harness is wrong."

    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        json={},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    cleartext_key = body["key"]
    assert cleartext_key.startswith("sk-")

    hashed = hash_token(cleartext_key)
    row = await proxy_server.prisma_client.db.litellm_verificationtoken.find_unique(
        where={"token": hashed}
    )
    assert row is not None, "Generated key did not land in LiteLLM_VerificationToken"
    assert row.token == hashed
    assert (
        row.token != cleartext_key
    ), "Cleartext token stored — credential boundary broken"
