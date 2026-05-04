import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.container_endpoints import ownership
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.containers.main import ContainerListResponse, ContainerObject


@pytest.fixture(autouse=True)
def clear_in_memory_container_owners():
    ownership._IN_MEMORY_CONTAINER_OWNERS.clear()
    ownership._CONTAINER_OWNER_CACHE.clear()
    yield
    ownership._IN_MEMORY_CONTAINER_OWNERS.clear()
    ownership._CONTAINER_OWNER_CACHE.clear()


def _container(container_id: str) -> ContainerObject:
    return ContainerObject(
        id=container_id,
        object="container",
        created_at=1,
        status="active",
    )


@pytest.mark.asyncio
async def test_should_record_container_owner_with_original_provider_id(monkeypatch):
    table = AsyncMock()
    table.find_unique.return_value = None
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    auth = UserAPIKeyAuth(user_id="user-1")

    response = _container("cntr_provider")

    await ownership.record_container_owner(
        response=response,
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    table.create.assert_awaited_once()
    data = table.create.await_args.kwargs["data"]
    assert data["model_object_id"] == "container:openai:cntr_provider"
    assert data["file_purpose"] == ownership.CONTAINER_OBJECT_PURPOSE
    assert data["created_by"] == "user-1"


@pytest.mark.asyncio
async def test_should_not_mutate_dict_container_response_when_recording_owner(
    monkeypatch,
):
    table = AsyncMock()
    table.find_unique.return_value = None
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    auth = UserAPIKeyAuth(user_id="user-1")
    response = {"id": "cntr_provider", "object": "container"}

    returned = await ownership.record_container_owner(
        response=response,
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    assert returned == {"id": "cntr_provider", "object": "container"}
    data = table.create.await_args.kwargs["data"]
    assert data["file_object"]["custom_llm_provider"] == "openai"
    assert data["file_object"]["provider_container_id"] == "cntr_provider"


@pytest.mark.asyncio
async def test_should_record_team_owner_for_keys_without_user_id(monkeypatch):
    table = AsyncMock()
    table.find_unique.return_value = None
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    auth = UserAPIKeyAuth(team_id="team-1")

    await ownership.record_container_owner(
        response=_container("cntr_provider"),
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    data = table.create.await_args.kwargs["data"]
    assert data["created_by"] == "team:team-1"
    assert data["updated_by"] == "team:team-1"


@pytest.mark.asyncio
async def test_should_record_token_owner_for_keys_without_user_team_or_org(monkeypatch):
    table = AsyncMock()
    table.find_unique.return_value = None
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    auth = UserAPIKeyAuth(token="hashed-token")

    await ownership.record_container_owner(
        response=_container("cntr_provider"),
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    data = table.create.await_args.kwargs["data"]
    assert data["created_by"] == "key:hashed-token"
    assert data["updated_by"] == "key:hashed-token"


@pytest.mark.asyncio
async def test_should_record_unscoped_owner_for_identityless_proxy_auth(monkeypatch):
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=None),
    )
    auth = UserAPIKeyAuth()

    await ownership.record_container_owner(
        response=_container("cntr_provider"),
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    assert (
        ownership._IN_MEMORY_CONTAINER_OWNERS["container:openai:cntr_provider"]
        == "__litellm_unscoped_proxy__"
    )
    original_id, provider = await ownership.assert_user_can_access_container(
        container_id="cntr_provider",
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )
    assert original_id == "cntr_provider"
    assert provider == "openai"


@pytest.mark.asyncio
async def test_should_skip_owner_record_when_provider_response_has_no_id(monkeypatch):
    table = AsyncMock()
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    response = {"object": "container"}

    returned = await ownership.record_container_owner(
        response=response,
        user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
        custom_llm_provider="openai",
    )

    assert returned == response
    table.find_unique.assert_not_awaited()
    table.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_should_fallback_to_memory_when_persistent_owner_record_fails(
    monkeypatch,
):
    table = AsyncMock()
    table.find_unique.side_effect = Exception("db unavailable")
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    auth = UserAPIKeyAuth(user_id="user-1")

    await ownership.record_container_owner(
        response=_container("cntr_provider"),
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    assert (
        ownership._IN_MEMORY_CONTAINER_OWNERS["container:openai:cntr_provider"]
        == "user-1"
    )


@pytest.mark.asyncio
async def test_should_track_container_owner_in_memory_without_prisma(monkeypatch):
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=None),
    )
    auth = UserAPIKeyAuth(user_id="user-1")

    await ownership.record_container_owner(
        response=_container("cntr_provider"),
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    original_id, provider = await ownership.assert_user_can_access_container(
        container_id="cntr_provider",
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    assert original_id == "cntr_provider"
    assert provider == "openai"


@pytest.mark.asyncio
async def test_should_bound_in_memory_container_owner_tracking(monkeypatch):
    monkeypatch.setattr(ownership, "MAX_IN_MEMORY_CONTAINER_OWNERS", 2)
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=None),
    )
    auth = UserAPIKeyAuth(user_id="user-1")

    for container_id in ("cntr_1", "cntr_2", "cntr_3"):
        await ownership.record_container_owner(
            response=_container(container_id),
            user_api_key_dict=auth,
            custom_llm_provider="openai",
        )

    assert list(ownership._IN_MEMORY_CONTAINER_OWNERS.keys()) == [
        "container:openai:cntr_2",
        "container:openai:cntr_3",
    ]


@pytest.mark.asyncio
async def test_should_deny_container_access_for_different_owner(monkeypatch):
    table = AsyncMock()
    table.find_first.return_value = SimpleNamespace(created_by="user-2")
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    auth = UserAPIKeyAuth(user_id="user-1")

    with pytest.raises(HTTPException) as exc:
        await ownership.assert_user_can_access_container(
            container_id="cntr_provider",
            user_api_key_dict=auth,
            custom_llm_provider="openai",
        )

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_should_deny_untracked_container_access_by_default(monkeypatch):
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=None),
    )
    auth = UserAPIKeyAuth(user_id="user-1")

    with pytest.raises(HTTPException) as exc:
        await ownership.assert_user_can_access_container(
            container_id="cntr_untracked",
            user_api_key_dict=auth,
            custom_llm_provider="openai",
        )

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_should_fallback_to_memory_when_owner_lookup_fails(monkeypatch):
    table = AsyncMock()
    table.find_first.side_effect = Exception("db unavailable")
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    ownership._IN_MEMORY_CONTAINER_OWNERS["container:openai:cntr_owned"] = "user-1"
    auth = UserAPIKeyAuth(user_id="user-1")

    original_id, provider = await ownership.assert_user_can_access_container(
        container_id="cntr_owned",
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    assert original_id == "cntr_owned"
    assert provider == "openai"


@pytest.mark.asyncio
async def test_should_use_memory_owner_when_db_recovers_without_row(monkeypatch):
    table = AsyncMock()
    table.find_first.return_value = None
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    ownership._IN_MEMORY_CONTAINER_OWNERS["container:openai:cntr_owned"] = "user-1"
    auth = UserAPIKeyAuth(user_id="user-1")

    original_id, provider = await ownership.assert_user_can_access_container(
        container_id="cntr_owned",
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    assert original_id == "cntr_owned"
    assert provider == "openai"


@pytest.mark.asyncio
async def test_should_fail_closed_when_owner_lookup_fails_without_memory(monkeypatch):
    table = AsyncMock()
    table.find_first.side_effect = Exception("db unavailable")
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    auth = UserAPIKeyAuth(user_id="user-1")

    with pytest.raises(HTTPException) as exc:
        await ownership.assert_user_can_access_container(
            container_id="cntr_owned",
            user_api_key_dict=auth,
            custom_llm_provider="openai",
        )

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_untracked_container_is_admin_only(monkeypatch):
    """Pre-isolation containers with no ownership row are admin-only.
    Non-admin callers see them as 403, with no opt-out flag re-opening
    the cross-tenant access primitive."""
    from fastapi import HTTPException

    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=None),
    )
    auth = UserAPIKeyAuth(user_id="user-1")

    with pytest.raises(HTTPException) as exc:
        await ownership.assert_user_can_access_container(
            container_id="cntr_untracked",
            user_api_key_dict=auth,
            custom_llm_provider="openai",
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_should_not_reassign_existing_container_to_different_owner(monkeypatch):
    table = AsyncMock()
    table.find_unique.return_value = SimpleNamespace(
        file_purpose=ownership.CONTAINER_OBJECT_PURPOSE,
        created_by="user-2",
    )
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    auth = UserAPIKeyAuth(user_id="user-1")

    with pytest.raises(HTTPException) as exc:
        await ownership.record_container_owner(
            response=_container("cntr_existing"),
            user_api_key_dict=auth,
            custom_llm_provider="openai",
        )

    assert exc.value.status_code == 403
    table.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_should_filter_container_list_to_owned_records(monkeypatch):
    table = AsyncMock()
    table.find_many.return_value = [
        SimpleNamespace(model_object_id="container:openai:cntr_owned"),
    ]
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    auth = UserAPIKeyAuth(user_id="user-1")
    response = ContainerListResponse(
        object="list",
        data=[_container("cntr_owned"), _container("cntr_other")],
        has_more=True,
    )

    filtered = await ownership.filter_container_list_response(
        response=response,
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    assert [item.id for item in filtered.data] == ["cntr_owned"]
    assert filtered.first_id == "cntr_owned"
    assert filtered.last_id == "cntr_owned"
    assert filtered.has_more is False
    where = table.find_many.await_args.kwargs["where"]
    assert where["file_purpose"] == ownership.CONTAINER_OBJECT_PURPOSE
    assert where["created_by"]["in"] == ["user-1", "user:user-1"]


@pytest.mark.asyncio
async def test_should_clear_has_more_when_filtered_container_list_is_empty(
    monkeypatch,
):
    table = AsyncMock()
    table.find_many.return_value = [
        SimpleNamespace(model_object_id="container:openai:cntr_owned"),
    ]
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    auth = UserAPIKeyAuth(user_id="user-1")
    response = ContainerListResponse(
        object="list",
        data=[_container("cntr_other")],
        has_more=True,
    )

    filtered = await ownership.filter_container_list_response(
        response=response,
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    assert filtered.data == []
    assert filtered.first_id is None
    assert filtered.last_id is None
    assert filtered.has_more is False


@pytest.mark.asyncio
async def test_should_clear_dict_has_more_when_filtered_container_list_is_empty(
    monkeypatch,
):
    table = AsyncMock()
    table.find_many.return_value = [
        SimpleNamespace(model_object_id="container:openai:cntr_owned"),
    ]
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    auth = UserAPIKeyAuth(user_id="user-1")
    response = {
        "object": "list",
        "data": [{"id": "cntr_other"}],
        "has_more": True,
    }

    filtered = await ownership.filter_container_list_response(
        response=response,
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    assert filtered["data"] == []
    assert filtered["first_id"] is None
    assert filtered["last_id"] is None
    assert filtered["has_more"] is False


@pytest.mark.asyncio
async def test_should_filter_container_list_with_in_memory_ownership(monkeypatch):
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=None),
    )
    auth = UserAPIKeyAuth(user_id="user-1")

    await ownership.record_container_owner(
        response=_container("cntr_owned"),
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    response = ContainerListResponse(
        object="list",
        data=[_container("cntr_owned"), _container("cntr_other")],
        has_more=False,
    )

    filtered = await ownership.filter_container_list_response(
        response=response,
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    assert [item.id for item in filtered.data] == ["cntr_owned"]


@pytest.mark.asyncio
async def test_should_filter_container_list_with_memory_when_db_lookup_fails(
    monkeypatch,
):
    table = AsyncMock()
    table.find_many.side_effect = Exception("db unavailable")
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    ownership._IN_MEMORY_CONTAINER_OWNERS["container:openai:cntr_owned"] = "user-1"
    auth = UserAPIKeyAuth(user_id="user-1")
    response = ContainerListResponse(
        object="list",
        data=[_container("cntr_owned"), _container("cntr_other")],
        has_more=True,
    )

    filtered = await ownership.filter_container_list_response(
        response=response,
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    assert [item.id for item in filtered.data] == ["cntr_owned"]
    assert filtered.has_more is False


@pytest.mark.asyncio
async def test_should_include_memory_container_list_when_db_recovers_without_row(
    monkeypatch,
):
    table = AsyncMock()
    table.find_many.return_value = []
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )
    ownership._IN_MEMORY_CONTAINER_OWNERS["container:openai:cntr_owned"] = "user-1"
    auth = UserAPIKeyAuth(user_id="user-1")
    response = ContainerListResponse(
        object="list",
        data=[_container("cntr_owned"), _container("cntr_other")],
        has_more=True,
    )

    filtered = await ownership.filter_container_list_response(
        response=response,
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    assert [item.id for item in filtered.data] == ["cntr_owned"]
    assert filtered.has_more is False


@pytest.mark.asyncio
async def test_should_validate_owner_and_forward_decoded_id_for_proxy_forwarding(
    monkeypatch,
):
    from litellm.proxy.container_endpoints import handler_factory

    proxy_server_stub = SimpleNamespace(
        general_settings={},
        llm_router=None,
        proxy_config=None,
        proxy_logging_obj=None,
        select_data_generator=None,
        user_api_base=None,
        user_max_tokens=None,
        user_model=None,
        user_request_timeout=None,
        user_temperature=None,
        version="test",
    )
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_stub)

    captured = {}

    class FakeProcessor:
        def __init__(self, data):
            captured["data"] = data

        async def base_process_llm_request(self, **kwargs):
            return captured["data"]

        async def _handle_llm_api_exception(self, **kwargs):
            raise kwargs["e"]

    monkeypatch.setattr(
        handler_factory,
        "ProxyBaseLLMRequestProcessing",
        FakeProcessor,
    )
    access_check = AsyncMock(return_value=("cntr_provider", "azure"))
    monkeypatch.setattr(
        handler_factory,
        "assert_user_can_access_container",
        access_check,
    )
    encoded_id = ResponsesAPIRequestUtils._build_container_id(
        custom_llm_provider="azure",
        model_id="router-gpt",
        container_id="cntr_provider",
    )

    result = await handler_factory._process_request(
        request=SimpleNamespace(query_params={}, headers={}),
        fastapi_response=SimpleNamespace(),
        user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
        route_type="alist_container_files",
        path_params={"container_id": encoded_id},
    )

    access_check.assert_awaited_once()
    assert access_check.await_args.kwargs["container_id"] == encoded_id
    assert result["container_id"] == "cntr_provider"
    assert result["custom_llm_provider"] == "azure"
    assert result["model_id"] == "router-gpt"


@pytest.mark.asyncio
async def test_should_validate_owner_and_forward_decoded_id_for_multipart_upload(
    monkeypatch,
):
    from litellm.proxy.common_utils import http_parsing_utils
    from litellm.proxy.container_endpoints import handler_factory

    proxy_server_stub = SimpleNamespace(
        general_settings={},
        llm_router=None,
        proxy_config=None,
        proxy_logging_obj=None,
        select_data_generator=None,
        user_api_base=None,
        user_max_tokens=None,
        user_model=None,
        user_request_timeout=None,
        user_temperature=None,
        version="test",
    )
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_stub)

    captured = {}

    class FakeProcessor:
        def __init__(self, data):
            captured["data"] = data

        async def base_process_llm_request(self, **kwargs):
            return captured["data"]

        async def _handle_llm_api_exception(self, **kwargs):
            raise kwargs["e"]

    monkeypatch.setattr(
        handler_factory,
        "ProxyBaseLLMRequestProcessing",
        FakeProcessor,
    )
    access_check = AsyncMock(return_value=("cntr_provider", "azure"))
    monkeypatch.setattr(
        handler_factory,
        "assert_user_can_access_container",
        access_check,
    )
    monkeypatch.setattr(
        http_parsing_utils,
        "get_form_data",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        http_parsing_utils,
        "convert_upload_files_to_file_data",
        AsyncMock(return_value={"file": ["file-data"]}),
    )
    encoded_id = ResponsesAPIRequestUtils._build_container_id(
        custom_llm_provider="azure",
        model_id="router-gpt",
        container_id="cntr_provider",
    )

    result = await handler_factory._process_multipart_upload_request(
        request=SimpleNamespace(query_params={}, headers={}),
        fastapi_response=SimpleNamespace(),
        user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
        route_type="aupload_container_file",
        container_id=encoded_id,
    )

    access_check.assert_awaited_once()
    assert access_check.await_args.kwargs["container_id"] == encoded_id
    assert result["container_id"] == "cntr_provider"
    assert result["custom_llm_provider"] == "azure"
    assert result["model_id"] == "router-gpt"
    assert result["file"] == "file-data"


@pytest.mark.asyncio
async def test_should_forward_decoded_container_id_for_proxy_retrieve(monkeypatch):
    from litellm.proxy.container_endpoints import endpoints

    proxy_server_stub = SimpleNamespace(
        general_settings={},
        llm_router=None,
        proxy_config=None,
        proxy_logging_obj=None,
        select_data_generator=None,
        user_api_base=None,
        user_max_tokens=None,
        user_model=None,
        user_request_timeout=None,
        user_temperature=None,
        version="test",
    )
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_stub)

    captured = {}

    class FakeProcessor:
        def __init__(self, data):
            captured["data"] = data

        async def base_process_llm_request(self, **kwargs):
            return captured["data"]

        async def _handle_llm_api_exception(self, **kwargs):
            raise kwargs["e"]

    monkeypatch.setattr(endpoints, "ProxyBaseLLMRequestProcessing", FakeProcessor)
    monkeypatch.setattr(
        endpoints,
        "assert_user_can_access_container",
        AsyncMock(return_value=("cntr_provider", "azure")),
    )
    encoded_id = ResponsesAPIRequestUtils._build_container_id(
        custom_llm_provider="azure",
        model_id="router-gpt",
        container_id="cntr_provider",
    )

    result = await endpoints.retrieve_container(
        request=SimpleNamespace(query_params={}, headers={}),
        container_id=encoded_id,
        fastapi_response=SimpleNamespace(),
        user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
    )

    assert result["container_id"] == "cntr_provider"
    assert result["custom_llm_provider"] == "azure"
    assert result["model_id"] == "router-gpt"


@pytest.mark.asyncio
async def test_should_record_container_owner_inside_create_endpoint(monkeypatch):
    from litellm.proxy.container_endpoints import endpoints

    proxy_server_stub = SimpleNamespace(
        general_settings={},
        llm_router=None,
        proxy_config=None,
        proxy_logging_obj=None,
        select_data_generator=None,
        user_api_base=None,
        user_max_tokens=None,
        user_model=None,
        user_request_timeout=None,
        user_temperature=None,
        version="test",
    )
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_stub)

    response = _container("cntr_provider")

    class FakeProcessor:
        def __init__(self, data):
            pass

        async def base_process_llm_request(self, **kwargs):
            return response

        async def _handle_llm_api_exception(self, **kwargs):
            raise kwargs["e"]

    record_owner = AsyncMock(return_value=response)
    monkeypatch.setattr(endpoints, "ProxyBaseLLMRequestProcessing", FakeProcessor)
    monkeypatch.setattr(endpoints, "record_container_owner", record_owner)

    result = await endpoints.create_container(
        request=SimpleNamespace(
            query_params={},
            headers={},
            json=AsyncMock(return_value={}),
            body=AsyncMock(return_value=b"{}"),
        ),
        fastapi_response=SimpleNamespace(),
        user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
    )

    assert result == response
    record_owner.assert_awaited_once_with(
        response=response,
        user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
        custom_llm_provider="openai",
    )


@pytest.mark.asyncio
async def test_should_not_route_owner_record_errors_through_llm_error_handler(
    monkeypatch,
):
    from litellm.proxy.container_endpoints import endpoints

    proxy_server_stub = SimpleNamespace(
        general_settings={},
        llm_router=None,
        proxy_config=None,
        proxy_logging_obj=None,
        select_data_generator=None,
        user_api_base=None,
        user_max_tokens=None,
        user_model=None,
        user_request_timeout=None,
        user_temperature=None,
        version="test",
    )
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_stub)

    class FakeProcessor:
        def __init__(self, data):
            pass

        async def base_process_llm_request(self, **kwargs):
            return _container("cntr_provider")

        async def _handle_llm_api_exception(self, **kwargs):
            raise AssertionError("ownership errors should not use LLM error handler")

    monkeypatch.setattr(endpoints, "ProxyBaseLLMRequestProcessing", FakeProcessor)
    monkeypatch.setattr(
        endpoints,
        "record_container_owner",
        AsyncMock(side_effect=HTTPException(status_code=403, detail="Forbidden")),
    )

    with pytest.raises(HTTPException) as exc:
        await endpoints.create_container(
            request=SimpleNamespace(
                query_params={},
                headers={},
                json=AsyncMock(return_value={}),
                body=AsyncMock(return_value=b"{}"),
            ),
            fastapi_response=SimpleNamespace(),
            user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
        )

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_should_return_response_when_owner_recording_raises_unexpected(
    monkeypatch,
):
    """If record_container_owner raises a non-HTTPException after upstream create,
    the upstream container exists but is untracked. The caller still gets the
    response (not a 500) so they aren't billed for an unusable resource — an
    operator reconciles via logs.
    """
    from litellm.proxy.container_endpoints import endpoints

    proxy_server_stub = SimpleNamespace(
        general_settings={},
        llm_router=None,
        proxy_config=None,
        proxy_logging_obj=None,
        select_data_generator=None,
        user_api_base=None,
        user_max_tokens=None,
        user_model=None,
        user_request_timeout=None,
        user_temperature=None,
        version="test",
    )
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_stub)

    created = _container("cntr_provider")

    class FakeProcessor:
        def __init__(self, data):
            pass

        async def base_process_llm_request(self, **kwargs):
            return created

        async def _handle_llm_api_exception(self, **kwargs):
            raise AssertionError("upstream-create errors only")

    monkeypatch.setattr(endpoints, "ProxyBaseLLMRequestProcessing", FakeProcessor)
    monkeypatch.setattr(
        endpoints,
        "record_container_owner",
        AsyncMock(side_effect=RuntimeError("transient db blip")),
    )

    response = await endpoints.create_container(
        request=SimpleNamespace(
            query_params={},
            headers={},
            json=AsyncMock(return_value={}),
            body=AsyncMock(return_value=b"{}"),
        ),
        fastapi_response=SimpleNamespace(),
        user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
    )

    assert response is created


@pytest.mark.asyncio
async def test_should_filter_container_list_inside_list_endpoint(monkeypatch):
    from litellm.proxy.container_endpoints import endpoints

    proxy_server_stub = SimpleNamespace(
        general_settings={},
        llm_router=None,
        proxy_config=None,
        proxy_logging_obj=None,
        select_data_generator=None,
        user_api_base=None,
        user_max_tokens=None,
        user_model=None,
        user_request_timeout=None,
        user_temperature=None,
        version="test",
    )
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_stub)

    response = ContainerListResponse(
        object="list",
        data=[_container("cntr_provider")],
        has_more=False,
    )

    class FakeProcessor:
        def __init__(self, data):
            pass

        async def base_process_llm_request(self, **kwargs):
            return response

        async def _handle_llm_api_exception(self, **kwargs):
            raise kwargs["e"]

    filter_response = AsyncMock(return_value=response)
    monkeypatch.setattr(endpoints, "ProxyBaseLLMRequestProcessing", FakeProcessor)
    monkeypatch.setattr(
        endpoints,
        "filter_container_list_response",
        filter_response,
    )

    result = await endpoints.list_containers(
        request=SimpleNamespace(query_params={}, headers={}),
        fastapi_response=SimpleNamespace(),
        user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
    )

    assert result == response
    filter_response.assert_awaited_once_with(
        response=response,
        user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
        custom_llm_provider="openai",
    )


@pytest.mark.asyncio
async def test_should_forward_decoded_container_id_for_proxy_delete(monkeypatch):
    from litellm.proxy.container_endpoints import endpoints

    proxy_server_stub = SimpleNamespace(
        general_settings={},
        llm_router=None,
        proxy_config=None,
        proxy_logging_obj=None,
        select_data_generator=None,
        user_api_base=None,
        user_max_tokens=None,
        user_model=None,
        user_request_timeout=None,
        user_temperature=None,
        version="test",
    )
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_stub)

    captured = {}

    class FakeProcessor:
        def __init__(self, data):
            captured["data"] = data

        async def base_process_llm_request(self, **kwargs):
            return captured["data"]

        async def _handle_llm_api_exception(self, **kwargs):
            raise kwargs["e"]

    monkeypatch.setattr(endpoints, "ProxyBaseLLMRequestProcessing", FakeProcessor)
    monkeypatch.setattr(
        endpoints,
        "assert_user_can_access_container",
        AsyncMock(return_value=("cntr_provider", "azure")),
    )
    encoded_id = ResponsesAPIRequestUtils._build_container_id(
        custom_llm_provider="azure",
        model_id="router-gpt",
        container_id="cntr_provider",
    )

    result = await endpoints.delete_container(
        request=SimpleNamespace(query_params={}, headers={}),
        container_id=encoded_id,
        fastapi_response=SimpleNamespace(),
        user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
    )

    assert result["container_id"] == "cntr_provider"
    assert result["custom_llm_provider"] == "azure"
    assert result["model_id"] == "router-gpt"


# ── Cache layer ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_container_owner_uses_cache_after_first_db_hit(monkeypatch):
    """Repeated access checks within the TTL window must not hit the DB.

    Greptile's P1 was that ownership reads issued a Prisma query on every
    request. The cache here mirrors `_byok_cred_cache`: TTL'd, capped, and
    invalidated on writes.
    """
    table = AsyncMock()
    fake_row = SimpleNamespace(
        created_by="user-1", file_purpose=ownership.CONTAINER_OBJECT_PURPOSE
    )
    table.find_first.return_value = fake_row
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )

    owner_first = await ownership._get_container_owner("cntr_x", "openai")
    owner_second = await ownership._get_container_owner("cntr_x", "openai")
    owner_third = await ownership._get_container_owner("cntr_x", "openai")

    assert owner_first == "user-1"
    assert owner_second == "user-1"
    assert owner_third == "user-1"
    # Single DB call across three reads — the cache absorbs the rest.
    assert table.find_first.await_count == 1


@pytest.mark.asyncio
async def test_get_container_owner_caches_negative_lookups(monkeypatch):
    """`None` (untracked) must also be cached so repeated misses don't query."""
    table = AsyncMock()
    table.find_first.return_value = None
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )

    assert await ownership._get_container_owner("cntr_x", "openai") is None
    assert await ownership._get_container_owner("cntr_x", "openai") is None
    assert table.find_first.await_count == 1


@pytest.mark.asyncio
async def test_record_container_owner_invalidates_cache(monkeypatch):
    """A recorded owner must drop the cached value so the next read re-fetches.

    Otherwise a stale `None` from a prior negative lookup would survive the
    create and the new owner would be invisible until the TTL elapses.
    """
    # Seed the cache with a stale negative result.
    ownership._write_container_owner_cache("container:openai:cntr_new", None)
    cached_hit, cached_value = ownership._read_container_owner_cache(
        "container:openai:cntr_new"
    )
    assert cached_hit and cached_value is None

    table = AsyncMock()
    table.find_unique.return_value = None
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )

    await ownership.record_container_owner(
        response=_container("cntr_new"),
        user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
        custom_llm_provider="openai",
    )

    # Invalidation drops the entry — next read goes to the DB.
    cached_hit, _ = ownership._read_container_owner_cache("container:openai:cntr_new")
    assert not cached_hit


@pytest.mark.asyncio
async def test_get_container_owner_does_not_cache_on_db_error(monkeypatch):
    """DB errors must skip caching so transient failures don't pin a `None`."""
    table = AsyncMock()
    table.find_first.side_effect = Exception("db unavailable")
    prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=table)
    )
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=prisma_client),
    )

    result = await ownership._get_container_owner("cntr_x", "openai")
    assert result is None
    cached_hit, _ = ownership._read_container_owner_cache("container:openai:cntr_x")
    assert not cached_hit


def test_container_owner_cache_expires_after_ttl(monkeypatch):
    """Entries past the TTL count as misses so writes elsewhere are eventually
    visible to this process."""
    monkeypatch.setattr(ownership, "_CONTAINER_OWNER_CACHE_TTL", 0.0)
    ownership._write_container_owner_cache("k", "user-1")
    cached_hit, _ = ownership._read_container_owner_cache("k")
    # TTL of 0 means anything in the cache is already stale.
    assert not cached_hit


def test_container_owner_cache_evicts_when_at_capacity(monkeypatch):
    """The cache must not grow unbounded; reaching capacity LRU-evicts the
    oldest entry, not the entire cache."""
    monkeypatch.setattr(ownership, "_CONTAINER_OWNER_CACHE_MAX_SIZE", 2)
    ownership._write_container_owner_cache("a", "user-a")
    ownership._write_container_owner_cache("b", "user-b")
    ownership._write_container_owner_cache("c", "user-c")
    # ``a`` was the oldest and is dropped; ``b`` and ``c`` survive.
    assert list(ownership._CONTAINER_OWNER_CACHE.keys()) == ["b", "c"]


def test_container_owner_cache_read_marks_as_recently_used(monkeypatch):
    """Reading an entry should reset its position so a subsequent eviction
    drops a less-recently-used entry instead of the just-touched one."""
    monkeypatch.setattr(ownership, "_CONTAINER_OWNER_CACHE_MAX_SIZE", 2)
    ownership._write_container_owner_cache("a", "user-a")
    ownership._write_container_owner_cache("b", "user-b")
    # Touch ``a`` so it becomes the most-recently-used.
    ownership._read_container_owner_cache("a")
    ownership._write_container_owner_cache("c", "user-c")
    # ``b`` is the LRU at this point; ``a`` and ``c`` survive.
    assert list(ownership._CONTAINER_OWNER_CACHE.keys()) == ["a", "c"]
