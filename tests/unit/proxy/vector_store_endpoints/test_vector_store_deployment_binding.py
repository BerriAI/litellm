from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.vector_store_endpoints.management_endpoints import new_vector_store
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
from litellm.vector_stores.deployments import parse_vector_store_deployments, vector_store_deployment_registry

_ADMIN = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN.value, user_id="admin")
_INTERNAL_USER = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER.value, user_id="user-1", team_id="team-a")


@pytest.fixture(autouse=True)
def kb_search_group() -> Iterator[None]:
    vector_store_deployment_registry.load(
        parse_vector_store_deployments(
            {
                "vector_store_deployments": [
                    {
                        "deployment_name": "kb-search",
                        "deployment_id": "eastus",
                        "litellm_params": {
                            "custom_llm_provider": "azure_ai",
                            "api_base": "https://eastus.search.windows.net",
                            "api_key": "key-eastus",
                        },
                    }
                ]
            }
        )
    )
    yield
    vector_store_deployment_registry.load(())


def _store(**litellm_params: object) -> LiteLLM_ManagedVectorStore:
    return LiteLLM_ManagedVectorStore(  # type: ignore[call-arg]
        vector_store_id="claims-docs", custom_llm_provider="azure_ai", litellm_params=litellm_params
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user", "store", "status", "message"),
    [
        (_INTERNAL_USER, _store(vector_store_deployment="kb-search"), 403, "Only proxy admins"),
        (_ADMIN, _store(vector_store_deployment="missing"), 400, "'missing' is not defined"),
        (_ADMIN, _store(vector_store_deployment="kb-search", api_base="https://other.example"), 400, "['api_base']"),
    ],
)
async def test_creating_a_store_bound_to_a_group_is_refused_unless_an_admin_binds_an_existing_group_cleanly(
    user, store, status, message
):
    create = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.vector_store_endpoints.management_endpoints.create_vector_store_in_db", create),
        pytest.raises(HTTPException) as refused,
    ):
        await new_vector_store(vector_store=store, user_api_key_dict=user)

    assert refused.value.status_code == status
    assert message in str(refused.value.detail)
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_admin_can_create_a_store_bound_to_an_existing_group():
    create = AsyncMock(return_value={"vector_store_id": "claims-docs", "custom_llm_provider": "azure_ai"})

    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.vector_store_endpoints.management_endpoints.create_vector_store_in_db", create),
    ):
        response = await new_vector_store(
            vector_store=_store(vector_store_deployment="kb-search"), user_api_key_dict=_ADMIN
        )

    assert response["status"] == "success"
    assert create.await_args.kwargs["litellm_params"] == {"vector_store_deployment": "kb-search"}

