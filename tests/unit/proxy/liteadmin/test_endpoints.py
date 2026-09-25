import asyncio
from typing import Final

import pytest

from litellm.proxy.liteadmin.endpoints import _coordinate, inference_url


@pytest.mark.parametrize(
    ("configured", "candidate", "expected"),
    (
        ("", "http://untrusted.example", "http://127.0.0.1:4000"),
        ("https://inference.example/v1", "https://inference.example/v1/", "https://inference.example/v1"),
        ("/gateway", "https://dashboard.example/gateway", "http://127.0.0.1:4000/gateway"),
    ),
)
def test_inference_destination_comes_only_from_trusted_gateway_settings(
    monkeypatch: pytest.MonkeyPatch, configured: str, candidate: str, expected: str
) -> None:
    monkeypatch.setenv("LITELLM_UI_API_DOC_BASE_URL", configured)
    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    assert inference_url(candidate, "http://127.0.0.1:4000") == expected


def test_a_changed_inference_destination_requires_reloading_for_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_UI_API_DOC_BASE_URL", "https://inference.example")
    with pytest.raises(ValueError, match="Reload LiteAdmin"):
        inference_url("https://different.example", "http://127.0.0.1:4000")


@pytest.mark.asyncio
async def test_disconnect_cancels_the_running_worker_forwarder() -> None:
    started: Final = asyncio.Event()
    cancelled: Final = asyncio.Event()

    async def work() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def receive() -> None:
        await started.wait()
        raise ConnectionError("Client disconnected")

    with pytest.raises(ConnectionError, match="Client disconnected"):
        await _coordinate(work, receive)
    assert cancelled.is_set()
