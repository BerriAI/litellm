import asyncio
import uuid

import pytest
import pytest_asyncio

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER


@pytest.fixture
def eden_key(monkeypatch) -> str:
    monkeypatch.delenv("EDENAI_API_BASE", raising=False)
    monkeypatch.setenv("EDENAI_API_KEY", "eden-test-key")
    monkeypatch.setattr(litellm, "api_key", None)
    return "eden-test-key"


@pytest.fixture
def no_eden_key(monkeypatch) -> None:
    monkeypatch.delenv("EDENAI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)


class SpendCapture(CustomLogger):
    """Records the cost the spend logs would store for one call, matched by its call id."""

    def __init__(self, call_id: str):
        super().__init__()
        self.call_id = call_id
        self.costs: list[object] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if kwargs.get("litellm_call_id") == self.call_id:
            self.costs.append((kwargs.get("standard_logging_object") or {}).get("response_cost"))

    async def settle(self) -> None:
        await asyncio.sleep(0)
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)


@pytest_asyncio.fixture
async def spend_capture(monkeypatch) -> SpendCapture:
    GLOBAL_LOGGING_WORKER.start()  # rebinds the worker's queue to this test's event loop
    capture = SpendCapture(call_id=f"eden-{uuid.uuid4()}")
    monkeypatch.setattr(litellm, "callbacks", [capture])
    return capture


@pytest.fixture
def httpx_transport(monkeypatch):
    """respx fakes httpx, so the async client must not sit on LiteLLM's default aiohttp transport."""
    monkeypatch.setattr(  # test-quality-ok: respx needs HTTPX enabled to fake the provider HTTP boundary.
        litellm,
        "disable_aiohttp_transport",
        True,
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()
