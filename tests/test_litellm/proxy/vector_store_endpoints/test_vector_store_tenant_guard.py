from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request, Response

import litellm
from litellm.proxy._types import LiteLLM_ManagedVectorStoresTable, UserAPIKeyAuth


def _mock_request() -> MagicMock:
    request = MagicMock(spec=Request)
    request.headers = {}
    request.method = "POST"
    request.query_params = {}
    request.url.path = "/v1/vector_stores/vs_path/search"
    return request


@pytest.mark.asyncio
async def test_vector_store_search_forces_path_id_over_body_id():
    from litellm.proxy.vector_store_endpoints.endpoints import vector_store_search

    captured_data = {}

    async def fake_base_process(self, **kwargs):
        captured_data.update(self.data)
        return {"ok": True}

    request = _mock_request()
    with (
        patch(
            "litellm.proxy.proxy_server._read_request_body",
            new=AsyncMock(
                return_value={
                    "vector_store_id": "vs_body_victim",
                    "query": "test",
                }
            ),
        ),
        patch.object(litellm, "vector_store_registry", None),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch(
            "litellm.proxy.vector_store_endpoints.endpoints.ProxyBaseLLMRequestProcessing.base_process_llm_request",
            new=fake_base_process,
        ),
    ):
        response = await vector_store_search(
            request=request,
            vector_store_id="vs_path_allowed",
            fastapi_response=Response(),
            user_api_key_dict=UserAPIKeyAuth(team_id="team-a"),
        )

    assert response == {"ok": True}
    assert captured_data["vector_store_id"] == "vs_path_allowed"


@pytest.mark.asyncio
async def test_vector_store_file_create_forces_path_id_over_body_id():
    from litellm.proxy.vector_store_files_endpoints.endpoints import (
        vector_store_file_create,
    )

    captured_data = {}

    async def fake_base_process(self, **kwargs):
        captured_data.update(self.data)
        return {"ok": True}

    mock_registry = MagicMock()
    mock_registry.get_litellm_managed_vector_store_from_registry.return_value = {
        "vector_store_id": "vs_path_allowed",
        "custom_llm_provider": "openai",
        "team_id": "team-a",
    }

    request = _mock_request()
    with (
        patch(
            "litellm.proxy.proxy_server._read_request_body",
            new=AsyncMock(
                return_value={
                    "vector_store_id": "vs_body_victim",
                    "file_id": "file_123",
                }
            ),
        ),
        patch.object(litellm, "vector_store_registry", mock_registry),
        patch(
            "litellm.proxy.vector_store_files_endpoints.endpoints.ProxyBaseLLMRequestProcessing.base_process_llm_request",
            new=fake_base_process,
        ),
    ):
        response = await vector_store_file_create(
            vector_store_id="vs_path_allowed",
            request=request,
            fastapi_response=Response(),
            user_api_key_dict=UserAPIKeyAuth(team_id="team-a"),
        )

    assert response == {"ok": True}
    assert captured_data["vector_store_id"] == "vs_path_allowed"
    assert captured_data["custom_llm_provider"] == "openai"
    mock_registry.get_litellm_managed_vector_store_from_registry.assert_called_once_with(
        vector_store_id="vs_path_allowed"
    )


@pytest.mark.asyncio
async def test_vector_store_file_create_denies_other_team_path_store():
    from litellm.proxy.vector_store_files_endpoints.endpoints import (
        vector_store_file_create,
    )

    mock_registry = MagicMock()
    mock_registry.get_litellm_managed_vector_store_from_registry.return_value = {
        "vector_store_id": "vs_other_team",
        "custom_llm_provider": "openai",
        "team_id": "team-b",
    }

    request = _mock_request()
    with (
        patch(
            "litellm.proxy.proxy_server._read_request_body",
            new=AsyncMock(return_value={"file_id": "file_123"}),
        ),
        patch.object(litellm, "vector_store_registry", mock_registry),
        patch(
            "litellm.proxy.vector_store_files_endpoints.endpoints.ProxyBaseLLMRequestProcessing.base_process_llm_request",
            new=AsyncMock(),
        ) as mock_base_process,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await vector_store_file_create(
                vector_store_id="vs_other_team",
                request=request,
                fastapi_response=Response(),
                user_api_key_dict=UserAPIKeyAuth(team_id="team-a"),
            )

    assert exc_info.value.status_code == 403
    mock_base_process.assert_not_called()


@pytest.mark.asyncio
async def test_rag_query_denies_nested_other_team_vector_store():
    from litellm.proxy.rag_endpoints.endpoints import rag_query

    mock_registry = MagicMock()
    mock_registry.get_litellm_managed_vector_store_from_registry.return_value = {
        "vector_store_id": "vs_other_team",
        "custom_llm_provider": "openai",
        "team_id": "team-b",
    }

    request = _mock_request()
    with (
        patch(
            "litellm.proxy.rag_endpoints.endpoints._read_request_body",
            new=AsyncMock(
                return_value={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hello"}],
                    "retrieval_config": {"vector_store_id": "vs_other_team"},
                }
            ),
        ),
        patch.object(litellm, "vector_store_registry", mock_registry),
        patch(
            "litellm.proxy.rag_endpoints.endpoints.litellm.aquery",
            new=AsyncMock(),
        ) as mock_aquery,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await rag_query(
                request=request,
                fastapi_response=Response(),
                user_api_key_dict=UserAPIKeyAuth(team_id="team-a"),
            )

    assert exc_info.value.status_code == 403
    mock_aquery.assert_not_called()


@pytest.mark.asyncio
async def test_rag_ingest_denies_nested_other_team_vector_store():
    from litellm.proxy.rag_endpoints.endpoints import rag_ingest

    mock_registry = MagicMock()
    mock_registry.get_litellm_managed_vector_store_from_registry.return_value = {
        "vector_store_id": "vs_other_team",
        "custom_llm_provider": "openai",
        "team_id": "team-b",
    }

    request = _mock_request()
    with (
        patch(
            "litellm.proxy.rag_endpoints.endpoints.parse_rag_ingest_request",
            new=AsyncMock(
                return_value=(
                    {
                        "vector_store": {
                            "custom_llm_provider": "openai",
                            "vector_store_id": "vs_other_team",
                        }
                    },
                    None,
                    "https://example.com/file.txt",
                    None,
                )
            ),
        ),
        patch.object(litellm, "vector_store_registry", mock_registry),
        patch(
            "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
            new=AsyncMock(),
        ) as mock_aingest,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await rag_ingest(
                request=request,
                fastapi_response=Response(),
                user_api_key_dict=UserAPIKeyAuth(team_id="team-a"),
            )

    assert exc_info.value.status_code == 403
    mock_aingest.assert_not_called()


def test_rag_payload_scan_rejects_excessive_nesting():
    from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
    from litellm.proxy.rag_endpoints.endpoints import (
        _collect_vector_store_ids_from_payload,
    )

    payload = {}
    current = payload
    for _ in range(DEFAULT_MAX_RECURSE_DEPTH + 1):
        current["nested"] = {}
        current = current["nested"]
    current["vector_store_id"] = "vs_too_deep"

    with pytest.raises(HTTPException) as exc_info:
        _collect_vector_store_ids_from_payload(payload)

    assert exc_info.value.status_code == 400


def test_rag_payload_scan_accepts_vector_store_id_at_depth_limit():
    from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
    from litellm.proxy.rag_endpoints.endpoints import (
        _collect_vector_store_ids_from_payload,
    )

    payload = {}
    current = payload
    for _ in range(DEFAULT_MAX_RECURSE_DEPTH):
        current["nested"] = {}
        current = current["nested"]
    current["vector_store_id"] = "vs_at_limit"

    assert _collect_vector_store_ids_from_payload(payload) == {"vs_at_limit"}


def test_rag_payload_scan_ignores_primitive_list_beyond_depth_limit():
    from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
    from litellm.proxy.rag_endpoints.endpoints import (
        _collect_vector_store_ids_from_payload,
    )

    payload = {}
    current = payload
    for _ in range(DEFAULT_MAX_RECURSE_DEPTH):
        current["nested"] = {}
        current = current["nested"]
    current["labels"] = ["alpha", "beta"]

    assert _collect_vector_store_ids_from_payload(payload) == set()


@pytest.mark.asyncio
async def test_responses_file_search_denies_other_team_vector_store():
    from litellm.proxy.common_request_processing import (
        _authorize_response_file_search_vector_stores,
    )

    mock_registry = MagicMock()
    mock_registry.get_litellm_managed_vector_store_from_registry.return_value = {
        "vector_store_id": "vs_other_team",
        "custom_llm_provider": "openai",
        "team_id": "team-b",
    }

    with patch.object(litellm, "vector_store_registry", mock_registry):
        with pytest.raises(HTTPException) as exc_info:
            await _authorize_response_file_search_vector_stores(
                data={
                    "tools": [
                        {
                            "type": "file_search",
                            "vector_store_ids": ["vs_other_team"],
                        }
                    ]
                },
                user_api_key_dict=UserAPIKeyAuth(team_id="team-a"),
            )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_vertex_discovery_denies_other_team_vector_store_credentials():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _base_vertex_proxy_route,
    )

    request = _mock_request()
    request.method = "GET"
    vector_store_credentials = {
        "vector_store_id": "vs_other_team",
        "custom_llm_provider": "vertex_ai",
        "team_id": "team-b",
    }

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
        new=AsyncMock(return_value=UserAPIKeyAuth(team_id="team-a")),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _base_vertex_proxy_route(
                endpoint="projects/p/locations/us-central1/dataStores/vs_other_team",
                request=request,
                fastapi_response=Response(),
                get_vertex_pass_through_handler=MagicMock(),
                router_credentials=vector_store_credentials,
            )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_get_managed_vector_store_uses_shared_cache_helper_for_db_fallback():
    from litellm.proxy.vector_store_endpoints.utils import (
        get_litellm_managed_vector_store,
    )

    mock_registry = MagicMock()
    mock_registry.get_litellm_managed_vector_store_from_registry.return_value = None
    cache_helper = AsyncMock(
        return_value=[
            LiteLLM_ManagedVectorStoresTable(
                vector_store_id="vs_cached",
                custom_llm_provider="openai",
                vector_store_name=None,
                vector_store_description=None,
                vector_store_metadata=None,
                created_at=None,
                updated_at=None,
                litellm_credential_name=None,
                litellm_params={"api_base": "https://example.com"},
                team_id="team-a",
                user_id=None,
            )
        ]
    )

    with (
        patch.object(litellm, "vector_store_registry", mock_registry),
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
        patch(
            "litellm.proxy.auth.auth_checks.get_managed_vector_store_rows_by_uuids",
            new=cache_helper,
        ),
    ):
        vector_store = await get_litellm_managed_vector_store(
            vector_store_id="vs_cached"
        )

    assert vector_store is not None
    assert vector_store["vector_store_id"] == "vs_cached"
    assert vector_store["team_id"] == "team-a"
    cache_helper.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_managed_vector_store_fails_closed_on_lookup_error():
    from litellm.proxy.vector_store_endpoints.utils import (
        get_litellm_managed_vector_store,
    )

    mock_registry = MagicMock()
    mock_registry.get_litellm_managed_vector_store_from_registry.side_effect = (
        RuntimeError("registry unavailable")
    )

    with patch.object(litellm, "vector_store_registry", mock_registry):
        with pytest.raises(HTTPException) as exc_info:
            await get_litellm_managed_vector_store(vector_store_id="vs_registry_only")

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_vertex_discovery_allows_unregistered_provider_native_datastore_id():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        vertex_discovery_proxy_route,
    )

    request = _mock_request()
    request.method = "GET"

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_litellm_managed_vector_store",
            new=AsyncMock(return_value=None),
        ) as mock_lookup,
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._base_vertex_proxy_route",
            new=AsyncMock(return_value={"ok": True}),
        ) as mock_base_route,
    ):
        response = await vertex_discovery_proxy_route(
            endpoint="projects/p/locations/us-central1/dataStores/vs_unknown",
            request=request,
            fastapi_response=Response(),
        )

    assert response == {"ok": True}
    mock_lookup.assert_awaited_once_with(vector_store_id="vs_unknown")
    assert mock_base_route.call_args.kwargs["router_credentials"] is None


@pytest.mark.asyncio
async def test_milvus_passthrough_denies_other_team_vector_store_index():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        milvus_proxy_route,
    )

    request = _mock_request()
    request.url.path = "/milvus/v2/vectordb/entities/search"

    index_object = MagicMock()
    index_object.litellm_params.vector_store_name = "tenant-b-store"
    index_object.litellm_params.vector_store_index = "tenant_b_collection"

    mock_index_registry = MagicMock()
    mock_index_registry.is_vector_store_index.return_value = True
    mock_index_registry.get_vector_store_index_by_name.return_value = index_object

    mock_vector_registry = MagicMock()
    mock_vector_registry.get_litellm_managed_vector_store_from_registry_by_name.return_value = {
        "vector_store_id": "vs_other_team",
        "custom_llm_provider": "milvus",
        "team_id": "team-b",
        "litellm_params": {"api_base": "https://milvus.example.com"},
    }

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=MagicMock(),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            new=AsyncMock(return_value={"collectionName": "managed_index"}),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint",
            return_value=True,
        ),
        patch.object(litellm, "vector_store_index_registry", mock_index_registry),
        patch.object(litellm, "vector_store_registry", mock_vector_registry),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await milvus_proxy_route(
                endpoint="v2/vectordb/entities/search",
                request=request,
                fastapi_response=Response(),
                user_api_key_dict=UserAPIKeyAuth(team_id="team-a"),
            )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_azure_passthrough_denies_other_team_vector_store_index():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        azure_proxy_route,
    )

    request = _mock_request()
    request.url.path = "/azure/indexes/managed_index/docs/search"

    index_object = MagicMock()
    index_object.litellm_params.vector_store_name = "tenant-b-store"

    mock_index_registry = MagicMock()
    mock_index_registry.is_vector_store_index.side_effect = (
        lambda vector_store_index_name: vector_store_index_name == "managed_index"
    )
    mock_index_registry.get_vector_store_index_by_name.return_value = index_object

    mock_vector_registry = MagicMock()
    mock_vector_registry.get_litellm_managed_vector_store_from_registry_by_name.return_value = {
        "vector_store_id": "vs_other_team",
        "custom_llm_provider": "azure_ai",
        "team_id": "team-b",
        "litellm_params": {"api_base": "https://azure.example.com"},
    }

    with (
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_passthrough_request_using_router_model",
            return_value=False,
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=MagicMock(),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint",
            return_value=True,
        ),
        patch.object(litellm, "vector_store_index_registry", mock_index_registry),
        patch.object(litellm, "vector_store_registry", mock_vector_registry),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await azure_proxy_route(
                endpoint="indexes/managed_index/docs/search",
                request=request,
                fastapi_response=Response(),
                user_api_key_dict=UserAPIKeyAuth(team_id="team-a"),
            )

    assert exc_info.value.status_code == 403
