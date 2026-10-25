import json
from collections.abc import Iterator
from typing import Final

import httpx
import litellm
import pytest
import respx
from litellm import Router

VECTOR_STORES_URL: Final = "https://api.openai.com/v1/vector_stores"


@pytest.fixture(autouse=True)
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-vector-store-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.asyncio
async def test_avector_store_update_sends_name_and_metadata_to_provider() -> None:
    with respx.mock(assert_all_called=True) as mock:
        route: Final = mock.post(f"{VECTOR_STORES_URL}/vs_test123").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "vs_test123",
                    "object": "vector_store",
                    "created_at": 1699061776,
                    "name": "Updated Name",
                    "metadata": {"key": "value"},
                    "status": "completed",
                },
            )
        )
        result: Final = await Router(model_list=[]).avector_store_update(
            vector_store_id="vs_test123",
            name="Updated Name",
            metadata={"key": "value"},
            custom_llm_provider="openai",
        )
        sent: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert sent["name"] == "Updated Name"
    assert sent["metadata"] == {"key": "value"}
    assert result["id"] == "vs_test123"
    assert result["name"] == "Updated Name"
    assert result["metadata"]["key"] == "value"


def test_vector_store_list_forwards_pagination_params_to_provider() -> None:
    with respx.mock(assert_all_called=True) as mock:
        route: Final = mock.get(VECTOR_STORES_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": f"vs_{index}", "object": "vector_store"} for index in range(5)],
                    "has_more": True,
                    "first_id": "vs_0",
                    "last_id": "vs_4",
                },
            )
        )
        result: Final = Router(model_list=[]).vector_store_list(
            limit=5, after="vs_previous", order="asc", custom_llm_provider="openai"
        )
        params: Final = route.calls.last.request.url.params
    assert params["limit"] == "5"
    assert params["after"] == "vs_previous"
    assert params["order"] == "asc"
    assert result["has_more"] is True
    assert len(result["data"]) == 5


def test_vector_store_update_forwards_expires_after_to_provider() -> None:
    expires_after: Final = {"anchor": "last_active_at", "days": 7}
    with respx.mock(assert_all_called=True) as mock:
        route: Final = mock.post(f"{VECTOR_STORES_URL}/vs_test123").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "vs_test123",
                    "object": "vector_store",
                    "expires_after": expires_after,
                    "expires_at": 1699668576,
                },
            )
        )
        result: Final = Router(model_list=[]).vector_store_update(
            vector_store_id="vs_test123", expires_after=expires_after, custom_llm_provider="openai"
        )
        sent: Final = json.loads(route.calls.last.request.content)
    assert sent["expires_after"] == expires_after
    assert result["expires_after"]["days"] == 7
    assert result["expires_at"] == 1699668576


def test_vector_store_retrieve_and_delete_return_provider_responses() -> None:
    with respx.mock(assert_all_called=True) as mock:
        retrieve_route: Final = mock.get(f"{VECTOR_STORES_URL}/vs_test123").mock(
            return_value=httpx.Response(
                200,
                json={"id": "vs_test123", "object": "vector_store", "status": "completed"},
            )
        )
        delete_route: Final = mock.delete(f"{VECTOR_STORES_URL}/vs_test123").mock(
            return_value=httpx.Response(
                200,
                json={"id": "vs_test123", "object": "vector_store.deleted", "deleted": True},
            )
        )
        router: Final = Router(model_list=[])
        retrieved: Final = router.vector_store_retrieve(
            vector_store_id="vs_test123",
            custom_llm_provider="openai",
        )
        deleted: Final = router.vector_store_delete(
            vector_store_id="vs_test123",
            custom_llm_provider="openai",
        )

    assert retrieve_route.call_count == 1
    assert delete_route.call_count == 1
    assert retrieved["id"] == "vs_test123"
    assert deleted["deleted"] is True


@pytest.mark.asyncio
async def test_avector_store_retrieve_and_delete_return_provider_responses() -> None:
    with respx.mock(assert_all_called=True) as mock:
        retrieve_route: Final = mock.get(f"{VECTOR_STORES_URL}/vs_test123").mock(
            return_value=httpx.Response(
                200,
                json={"id": "vs_test123", "object": "vector_store", "status": "completed"},
            )
        )
        delete_route: Final = mock.delete(f"{VECTOR_STORES_URL}/vs_test123").mock(
            return_value=httpx.Response(
                200,
                json={"id": "vs_test123", "object": "vector_store.deleted", "deleted": True},
            )
        )
        router: Final = Router(model_list=[])
        retrieved: Final = await router.avector_store_retrieve(
            vector_store_id="vs_test123",
            custom_llm_provider="openai",
        )
        deleted: Final = await router.avector_store_delete(
            vector_store_id="vs_test123",
            custom_llm_provider="openai",
        )

    assert retrieve_route.call_count == 1
    assert delete_route.call_count == 1
    assert retrieved["id"] == "vs_test123"
    assert deleted["deleted"] is True
