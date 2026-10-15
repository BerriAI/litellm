import uuid
from collections.abc import Iterator
from typing import Final

import httpx
import pytest
import pytest_asyncio
import respx

import litellm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from tests.unit.llms.sail.helpers import SAIL_API_BASE, SpendCapture, chat_completion_body


@pytest.fixture
def sail_env(local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("SAIL_API_KEY", "sail-test-key")
    monkeypatch.delenv("SAIL_API_BASE", raising=False)
    monkeypatch.setattr(
        litellm,
        "disable_aiohttp_transport",
        True,
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest_asyncio.fixture
async def spend_capture(monkeypatch: pytest.MonkeyPatch) -> SpendCapture:
    GLOBAL_LOGGING_WORKER.start()
    capture: Final = SpendCapture(call_id=f"sail-{uuid.uuid4()}")
    monkeypatch.setattr(litellm, "callbacks", [capture])
    return capture


@pytest.fixture
def chat_route(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(f"{SAIL_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_completion_body())
    )
