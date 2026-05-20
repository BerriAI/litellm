"""Smoke tests proving the harness boots and talks to the proxy app."""

import pytest

from .conftest import MASTER_KEY

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_liveliness(proxy_client):
    resp = await proxy_client.get("/health/liveliness")
    assert resp.status_code == 200


async def test_key_generate_lands_in_db(proxy_client, prisma, scratch):
    """De-risk gate: prove the harness exercises the full stack end-to-end.

    A successful ``/key/generate`` requires:
      * the FastAPI lifespan ran (``proxy_startup_event``),
      * ``prisma_client`` connected,
      * ``user_api_key_auth`` accepted the master key,
      * the real ``generate_key_helper_fn`` wrote a hashed row to
        ``LiteLLM_VerificationToken``.

    The scratch fixture tags the row with its prefix so the per-test teardown
    cleans it up — keeps the proxy DB free of accumulated cruft on repeated
    local runs.
    """
    from litellm.proxy.utils import hash_token

    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        json={"key_alias": scratch.prefix},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    cleartext_key = body["key"]
    assert cleartext_key.startswith("sk-")

    hashed = hash_token(cleartext_key)
    row = await prisma.db.litellm_verificationtoken.find_unique(where={"token": hashed})
    assert row is not None, "Generated key did not land in LiteLLM_VerificationToken"
    assert row.token == hashed
    assert (
        row.token != cleartext_key
    ), "Cleartext token stored — credential boundary broken"
