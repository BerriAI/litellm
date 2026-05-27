import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.container_endpoints import ownership
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.containers.main import ContainerListResponse, ContainerObject


@pytest.fixture(autouse=True)
def clear_container_owner_cache():
    for cache in (
        ownership._CONTAINER_OWNER_CACHE,
        ownership._ALLOWED_CONTAINER_IDS_CACHE,
    ):
        cache.cache_dict.clear()
        cache.ttl_dict.clear()
    yield
    for cache in (
        ownership._CONTAINER_OWNER_CACHE,
        ownership._ALLOWED_CONTAINER_IDS_CACHE,
    ):
        cache.cache_dict.clear()
        cache.ttl_dict.clear()


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
async def test_should_reject_record_for_identityless_proxy_auth(monkeypatch):
    """Identity-less callers (no user_id / team_id / org_id / api_key /
    token) cannot record ownership — stamping a shared sentinel would let
    any two such callers see each other's containers."""
    from fastapi import HTTPException

    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=None),
    )
    auth = UserAPIKeyAuth()

    with pytest.raises(HTTPException) as exc:
        await ownership.record_container_owner(
            response=_container("cntr_provider"),
            user_api_key_dict=auth,
            custom_llm_provider="openai",
        )
    assert exc.value.status_code == 403
    assert "identity scope" in str(exc.value.detail)


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
async def test_allowed_container_ids_uses_cache_after_first_db_hit(monkeypatch):
    """``GET /v1/containers`` filtering must not issue a fresh ``find_many``
    on every list call within the cache TTL window."""
    table = AsyncMock()
    table.find_many.return_value = [
        SimpleNamespace(model_object_id="container:openai:cntr_a"),
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

    first = await ownership._get_allowed_container_ids(auth)
    second = await ownership._get_allowed_container_ids(auth)
    third = await ownership._get_allowed_container_ids(auth)

    assert first == {"container:openai:cntr_a"}
    assert second == first
    assert third == first
    # Single DB call across three list filterings — the cache absorbs the rest.
    assert table.find_many.await_count == 1


@pytest.mark.asyncio
async def test_record_container_owner_invalidates_caller_list_cache(monkeypatch):
    """A just-created container must show up on the caller's next ``GET
    /v1/containers`` — recording the owner has to drop the caller's
    list-cache entry, otherwise the new container is invisible for up
    to the cache TTL."""
    table = AsyncMock()
    table.find_unique.return_value = None
    table.find_many.return_value = [
        SimpleNamespace(model_object_id="container:openai:cntr_old"),
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

    # Prime the list cache.
    await ownership._get_allowed_container_ids(auth)
    assert table.find_many.await_count == 1

    # Recording a new owner invalidates the caller's list-cache entry.
    table.find_many.return_value = [
        SimpleNamespace(model_object_id="container:openai:cntr_old"),
        SimpleNamespace(model_object_id="container:openai:cntr_new"),
    ]
    await ownership.record_container_owner(
        response=_container("cntr_new"),
        user_api_key_dict=auth,
        custom_llm_provider="openai",
    )

    # Next list call refreshes from DB and picks up the new container.
    refreshed = await ownership._get_allowed_container_ids(auth)
    assert "container:openai:cntr_new" in refreshed
    assert table.find_many.await_count == 2


@pytest.mark.asyncio
async def test_admin_with_identity_records_container_ownership(monkeypatch):
    """The admin early-return only short-circuits when there's literally no
    container ID to stamp. An admin with identity (the master-key path
    populates ``user_id`` + ``api_key``) creates an owned row like any
    other caller, so admin-created containers aren't permanently
    untracked."""
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
    admin_auth = UserAPIKeyAuth(
        user_id="proxy-admin",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
    )

    await ownership.record_container_owner(
        response=_container("cntr_admin"),
        user_api_key_dict=admin_auth,
        custom_llm_provider="openai",
    )

    table.create.assert_awaited_once()
    created_data = table.create.await_args.kwargs["data"]
    assert created_data["created_by"] == "proxy-admin"
