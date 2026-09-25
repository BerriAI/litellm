import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
    file_obj = json.loads(data["file_object"])
    assert file_obj["custom_llm_provider"] == "openai"
    assert file_obj["provider_container_id"] == "cntr_provider"


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


def _owned_containers_in_db(monkeypatch, *model_object_ids: str) -> AsyncMock:
    table = AsyncMock()
    table.find_many.return_value = [SimpleNamespace(model_object_id=object_id) for object_id in model_object_ids]
    monkeypatch.setattr(
        ownership,
        "_get_prisma_client",
        AsyncMock(return_value=SimpleNamespace(db=SimpleNamespace(litellm_managedobjecttable=table))),
    )
    return table


def _upstream(pages_by_after):
    calls = []

    async def fetch_page(after, limit):
        calls.append((after, limit))
        return pages_by_after[after]

    return fetch_page, calls


def _page(*container_ids: str, has_more: bool) -> ContainerListResponse:
    return ContainerListResponse(
        object="list",
        data=[_container(container_id) for container_id in container_ids],
        has_more=has_more,
    )


async def _list_owned(fetch_page, after=None, limit=None):
    return await ownership.list_owned_containers(
        fetch_page=fetch_page,
        after=after,
        limit=limit,
        user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
        custom_llm_provider="openai",
    )


@pytest.mark.asyncio
async def test_should_page_upstream_until_owned_containers_fill_the_limit(monkeypatch):
    table = _owned_containers_in_db(monkeypatch, "container:openai:cntr_owned")
    fetch_page, calls = _upstream(
        {
            None: _page("cntr_other_1", "cntr_other_2", has_more=True),
            "cntr_other_2": _page("cntr_owned", has_more=False),
        }
    )

    listed = await _list_owned(fetch_page, limit=1)

    assert [item.id for item in listed.data] == ["cntr_owned"]
    assert listed.first_id == "cntr_owned"
    assert listed.last_id == "cntr_owned"
    assert listed.has_more is False
    assert calls == [(None, 100), ("cntr_other_2", 100)]
    where = table.find_many.await_args.kwargs["where"]
    assert where["file_purpose"] == ownership.CONTAINER_OBJECT_PURPOSE
    assert where["created_by"]["in"] == ["user-1", "user:user-1"]


@pytest.mark.asyncio
async def test_should_trim_owned_containers_to_the_limit_without_mutating_the_upstream_page(monkeypatch):
    _owned_containers_in_db(monkeypatch, "container:openai:cntr_owned_1", "container:openai:cntr_owned_2")
    upstream_page = _page("cntr_owned_1", "cntr_other", "cntr_owned_2", has_more=False)
    fetch_page, calls = _upstream({None: upstream_page})

    listed = await _list_owned(fetch_page, limit=1)

    assert [item.id for item in listed.data] == ["cntr_owned_1"]
    assert listed.first_id == "cntr_owned_1"
    assert listed.last_id == "cntr_owned_1"
    assert listed.has_more is True
    assert calls == [(None, 100)]
    assert [item.id for item in upstream_page.data] == ["cntr_owned_1", "cntr_other", "cntr_owned_2"]
    assert upstream_page.has_more is False


@pytest.mark.asyncio
async def test_should_start_paging_from_the_requested_cursor(monkeypatch):
    _owned_containers_in_db(monkeypatch, "container:openai:cntr_owned_2")
    fetch_page, calls = _upstream({"cntr_owned_1": _page("cntr_other", "cntr_owned_2", has_more=False)})

    listed = await _list_owned(fetch_page, after="cntr_owned_1", limit=1)

    assert [item.id for item in listed.data] == ["cntr_owned_2"]
    assert listed.has_more is False
    assert calls == [("cntr_owned_1", 100)]


@pytest.mark.asyncio
async def test_should_default_to_twenty_owned_containers_per_page(monkeypatch):
    owned_ids = tuple(f"cntr_owned_{index}" for index in range(21))
    _owned_containers_in_db(monkeypatch, *(f"container:openai:{container_id}" for container_id in owned_ids))
    fetch_page, _ = _upstream({None: _page(*owned_ids, has_more=False)})

    listed = await _list_owned(fetch_page)

    assert [item.id for item in listed.data] == list(owned_ids[:20])
    assert listed.last_id == "cntr_owned_19"
    assert listed.has_more is True


@pytest.mark.asyncio
async def test_should_stop_after_five_upstream_pages_and_keep_has_more(monkeypatch):
    _owned_containers_in_db(monkeypatch, "container:openai:cntr_owned")
    fetch_page, calls = _upstream(
        {
            None: _page("cntr_other_0", has_more=True),
            **{f"cntr_other_{index}": _page(f"cntr_other_{index + 1}", has_more=True) for index in range(6)},
        }
    )

    listed = await _list_owned(fetch_page, limit=1)

    assert listed.data == []
    assert listed.first_id is None
    assert listed.last_id is None
    assert listed.has_more is True
    assert len(calls) == 5


@pytest.mark.asyncio
async def test_should_stop_when_upstream_has_no_more_pages(monkeypatch):
    _owned_containers_in_db(monkeypatch, "container:openai:cntr_owned")
    fetch_page, calls = _upstream({None: _page("cntr_other", has_more=False)})

    listed = await _list_owned(fetch_page, limit=1)

    assert listed.data == []
    assert listed.has_more is False
    assert calls == [(None, 100)]


@pytest.mark.asyncio
async def test_should_build_dict_pages_without_mutating_the_upstream_page(monkeypatch):
    _owned_containers_in_db(monkeypatch, "container:openai:cntr_owned")
    upstream_page = {"object": "list", "data": [{"id": "cntr_other"}, {"id": "cntr_owned"}], "has_more": False}
    fetch_page, _ = _upstream({None: upstream_page})

    listed = await _list_owned(fetch_page, limit=1)

    assert listed == {
        "object": "list",
        "data": [{"id": "cntr_owned"}],
        "first_id": "cntr_owned",
        "last_id": "cntr_owned",
        "has_more": False,
    }
    assert [item["id"] for item in upstream_page["data"]] == ["cntr_other", "cntr_owned"]


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
async def test_should_list_owned_containers_inside_list_endpoint(monkeypatch):
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

    upstream_page = _page("cntr_provider", has_more=False)
    processor_cls = MagicMock(
        side_effect=lambda data: SimpleNamespace(base_process_llm_request=AsyncMock(return_value=upstream_page))
    )
    monkeypatch.setattr(endpoints, "ProxyBaseLLMRequestProcessing", processor_cls)
    list_owned = AsyncMock(return_value=upstream_page)
    monkeypatch.setattr(endpoints, "list_owned_containers", list_owned)

    result = await endpoints.list_containers(
        request=SimpleNamespace(query_params={}, headers={}),
        fastapi_response=SimpleNamespace(),
        user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
        after="cntr_prev",
        limit=2,
        order="desc",
    )

    assert result == upstream_page
    kwargs = list_owned.await_args.kwargs
    assert kwargs["after"] == "cntr_prev"
    assert kwargs["limit"] == 2
    assert kwargs["user_api_key_dict"] == UserAPIKeyAuth(user_id="user-1")
    assert kwargs["custom_llm_provider"] == "openai"
    processor_cls.assert_not_called()

    assert await kwargs["fetch_page"]("cntr_page_cursor", 100) == upstream_page
    forwarded = processor_cls.call_args.kwargs["data"]
    assert forwarded["after"] == "cntr_page_cursor"
    assert forwarded["limit"] == 100
    assert forwarded["order"] == "desc"
    assert forwarded["custom_llm_provider"] == "openai"


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


@pytest.mark.asyncio
async def test_should_record_containers_from_responses_output_for_service_account(
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
    auth = UserAPIKeyAuth(team_id="team-1")
    encoded_container_id = (
        "cntr_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOmF6dXJlO21vZGVsX2lkOmR"
        "lZi0xMjM7Y29udGFpbmVyX2lkOmNudHJfbmF0aXZl"
    )
    responses_payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [
                            {
                                "type": "container_file_citation",
                                "container_id": encoded_container_id,
                                "file_id": "cfile_abc",
                            }
                        ],
                    }
                ],
            }
        ],
        "_hidden_params": {"custom_llm_provider": "azure"},
    }

    await ownership.record_container_owners_from_responses_response(
        response=responses_payload,
        user_api_key_dict=auth,
    )

    table.create.assert_awaited_once()
    created_data = table.create.await_args.kwargs["data"]
    assert created_data["created_by"] == "team:team-1"
    assert created_data["unified_object_id"] == encoded_container_id


@pytest.mark.asyncio
async def test_service_account_can_access_container_after_responses_tracking(
    monkeypatch,
):
    encoded_container_id = (
        "cntr_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOmF6dXJlO21vZGVsX2lkOmR"
        "lZi0xMjM7Y29udGFpbmVyX2lkOmNudHJfbmF0aXZl"
    )
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

    await ownership.record_container_owners_from_responses_response(
        response={
            "output": [
                {
                    "type": "code_interpreter_call",
                    "container_id": encoded_container_id,
                }
            ],
            "_hidden_params": {"custom_llm_provider": "azure"},
        },
        user_api_key_dict=auth,
    )

    original_id, provider = await ownership.assert_user_can_access_container(
        container_id=encoded_container_id,
        user_api_key_dict=auth,
        custom_llm_provider="azure",
    )
    assert original_id == "cntr_native"
    assert provider == "azure"


@pytest.mark.asyncio
async def test_should_record_container_ownership_after_streaming_responses_finish(
    monkeypatch,
):
    """Streaming /v1/responses calls return through the
    ``select_data_generator`` branch and never reach the non-streaming
    container-ownership tail. The wrapper must read
    ``completed_response`` off the upstream iterator once iteration
    finishes and write the row, otherwise code-interpreter containers
    created during the stream stay unregistered and follow-up file API
    calls 403.
    """
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    encoded_container_id = (
        "cntr_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOmF6dXJlO21vZGVsX2lkOmR"
        "lZi0xMjM7Y29udGFpbmVyX2lkOmNudHJfbmF0aXZl"
    )
    response_body = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="code_interpreter_call",
                container_id=encoded_container_id,
                code_interpreter_call=None,
            )
        ]
    )
    stream_response = SimpleNamespace(
        completed_response=SimpleNamespace(response=response_body),
        _hidden_params={"custom_llm_provider": "azure"},
    )

    async def fake_sse_generator():
        yield "data: chunk-1\n\n"
        yield "data: chunk-2\n\n"

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

    wrapped = (
        ProxyBaseLLMRequestProcessing._wrap_responses_stream_for_container_ownership(
            original_stream_response=stream_response,
            wrapped_generator=fake_sse_generator(),
            user_api_key_dict=auth,
        )
    )

    chunks = [chunk async for chunk in wrapped]
    assert chunks == ["data: chunk-1\n\n", "data: chunk-2\n\n"]

    table.create.assert_awaited_once()
    created_data = table.create.await_args.kwargs["data"]
    assert created_data["created_by"] == "team:team-1"
    assert created_data["unified_object_id"] == encoded_container_id


@pytest.mark.asyncio
async def test_streaming_ownership_wrap_no_op_when_stream_did_not_complete(
    monkeypatch,
):
    """If the stream errored before ``response.completed``,
    ``completed_response`` is ``None`` — we must skip the ownership
    write rather than crash the response generator."""
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    stream_response = SimpleNamespace(completed_response=None)

    async def fake_sse_generator():
        yield "data: chunk-1\n\n"

    record = AsyncMock()
    monkeypatch.setattr(
        ownership,
        "record_container_owners_from_responses_response",
        record,
    )

    wrapped = (
        ProxyBaseLLMRequestProcessing._wrap_responses_stream_for_container_ownership(
            original_stream_response=stream_response,
            wrapped_generator=fake_sse_generator(),
            user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
        )
    )
    chunks = [chunk async for chunk in wrapped]

    assert chunks == ["data: chunk-1\n\n"]
    record.assert_not_awaited()
