"""Smoke tests proving the harness boots and talks to the proxy app."""


async def test_liveliness(proxy_client):
    resp = await proxy_client.get("/health/liveliness")
    assert resp.status_code == 200
