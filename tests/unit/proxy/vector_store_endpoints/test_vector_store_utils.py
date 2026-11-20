import base64
import json
from collections.abc import Mapping, Sequence
from typing import Final

import httpx
import pytest

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.vector_store_endpoints.utils import get_litellm_managed_vector_store, with_managed_search_file_ids
from litellm.types.utils import SpecialEnums
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry


def _registered_store(litellm_params: object) -> dict[str, object]:
    return {
        "vector_store_id": "vs_1",
        "custom_llm_provider": "bedrock",
        "vector_store_name": "kb",
        "vector_store_metadata": '{"source": "db"}',
        "team_id": None,
        "litellm_params": litellm_params,
    }


@pytest.mark.parametrize(
    ("stored_params", "expected_params"),
    [
        (
            '{"api_base": "https://example.com", "embedding": {"api_key": "sk-1", "dims": [1, null]}}',
            {"api_base": "https://example.com", "embedding": {"api_key": "sk-1", "dims": [1, None]}},
        ),
        (
            '{"top_k": 12, "score_threshold": 12.0, "api_version": "12", "stream": true, "verify": false,'
            ' "region": null, "max_tokens": 123456789012345678901234567890}',
            {
                "top_k": 12,
                "score_threshold": 12.0,
                "api_version": "12",
                "stream": True,
                "verify": False,
                "region": None,
                "max_tokens": 123456789012345678901234567890,
            },
        ),
        ('{"z": 1, "a": 2, "a": 3}', {"z": 1, "a": 3}),
        ("{}", {}),
    ],
)
async def test_registry_store_with_json_object_string_params_resolves_to_decoded_params(
    monkeypatch: pytest.MonkeyPatch, stored_params: str, expected_params: dict[str, object]
):
    store = _registered_store(stored_params)
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([store]))

    resolved = await get_litellm_managed_vector_store("vs_1")

    assert resolved == {**store, "litellm_params": expected_params}
    assert json.dumps(resolved["litellm_params"]) == json.dumps(expected_params)
    assert store["litellm_params"] == stored_params


@pytest.mark.parametrize(
    "stored_params",
    [
        "[1, 2]",
        '[["api_base", "https://example.com"]]',
        '"{\\"api_base\\": \\"https://example.com\\"}"',
        "12",
        "12.5",
        "true",
        "null",
        "{not json",
        "",
    ],
)
async def test_registry_store_with_non_object_or_invalid_json_string_params_resolves_to_empty_params(
    monkeypatch: pytest.MonkeyPatch, stored_params: str
):
    store = _registered_store(stored_params)
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([store]))

    resolved = await get_litellm_managed_vector_store("vs_1")

    assert resolved == {**store, "litellm_params": {}}
    assert store["litellm_params"] == stored_params


@pytest.mark.parametrize("stored_params", [{"api_base": "https://example.com"}, {}, None])
async def test_registry_store_with_non_string_params_is_returned_as_is(
    monkeypatch: pytest.MonkeyPatch, stored_params: object
):
    store = _registered_store(stored_params)
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([store]))

    resolved = await get_litellm_managed_vector_store("vs_1")

    assert resolved is store
    assert resolved["litellm_params"] is stored_params


def _managed_file_id(provider_file_id: str) -> str:
    unified_file_id: Final = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
        "text/plain", "unified-1", "openai/team-provider", provider_file_id, "team-deployment"
    )
    return base64.urlsafe_b64encode(unified_file_id.encode()).decode().rstrip("=")


class _ManagedFileIds:
    def __init__(self, unified_file_ids: Mapping[str, str]) -> None:
        self.unified_file_ids: Final = unified_file_ids

    async def get_unified_file_ids_for_provider_file_ids(
        self, provider_file_ids: Sequence[str], user_api_key_dict: UserAPIKeyAuth
    ) -> Mapping[str, str]:
        return {file_id: self.unified_file_ids[file_id] for file_id in provider_file_ids if file_id in self.unified_file_ids}


class _UnreachableManagedFilesTable:
    async def get_unified_file_ids_for_provider_file_ids(
        self, provider_file_ids: Sequence[str], user_api_key_dict: UserAPIKeyAuth
    ) -> Mapping[str, str]:
        raise httpx.ConnectError("database is unreachable")


def _search_hit(file_id: str) -> Mapping[str, object]:
    return {"file_id": file_id, "filename": f"{file_id}.txt", "score": 0.5, "content": []}


def _search_page(*hits: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "object": "vector_store.search_results.page",
        "search_query": "q",
        "data": list(hits),
        "has_more": False,
        "next_page": None,
    }


@pytest.mark.asyncio
async def test_search_hits_carry_the_managed_file_id_ingest_returned() -> None:
    managed_ids: Final = _ManagedFileIds(
        {"file-ingested": _managed_file_id("file-ingested"), "file-mismatched": _managed_file_id("file-other")}
    )

    mapped: Final = await with_managed_search_file_ids(
        _search_page(_search_hit("file-ingested"), _search_hit("file-unregistered"), _search_hit("file-mismatched")),
        managed_ids,
        UserAPIKeyAuth(api_key="team-a-member-key", team_id="team-a"),
    )

    assert mapped == _search_page(
        {**_search_hit("file-ingested"), "file_id": _managed_file_id("file-ingested")},
        _search_hit("file-unregistered"),
        _search_hit("file-mismatched"),
    )


@pytest.mark.asyncio
async def test_search_hits_keep_provider_file_ids_without_the_managed_files_hook() -> None:
    page: Final = _search_page(_search_hit("file-ingested"))

    assert await with_managed_search_file_ids(page, None, UserAPIKeyAuth(api_key="k", team_id="team-a")) == page


@pytest.mark.asyncio
async def test_search_hits_keep_provider_file_ids_when_the_managed_file_lookup_fails() -> None:
    page: Final = _search_page(_search_hit("file-ingested"))

    assert (
        await with_managed_search_file_ids(
            page, _UnreachableManagedFilesTable(), UserAPIKeyAuth(api_key="k", team_id="team-a")
        )
        == page
    )


@pytest.mark.asyncio
async def test_fallback_resolution_checks_backup_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from litellm.proxy.vector_store_endpoints.utils import resolve_vector_store_fallback_plan
    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore

    primary: Final = LiteLLM_ManagedVectorStore(
        is_config=True,
        vector_store_id="primary",
        custom_llm_provider="openai",
        team_id="one",
        litellm_params={"fallbacks": ["private-backup"]},
    )
    backup: Final = LiteLLM_ManagedVectorStore(
        is_config=True,
        vector_store_id="private-backup",
        custom_llm_provider="openai",
        team_id="two",
    )
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([primary, backup]))
    with pytest.raises(HTTPException) as denied:
        await resolve_vector_store_fallback_plan(primary, UserAPIKeyAuth(team_id="one"))
    assert denied.value.status_code == 403


@pytest.mark.asyncio
async def test_unified_fallback_allows_another_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.vector_store_endpoints.utils import resolve_vector_store_fallback_plan
    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore

    primary: Final = LiteLLM_ManagedVectorStore(
        is_config=True,
        vector_store_id="primary",
        custom_llm_provider="azure_ai",
        litellm_params={"fallbacks": ["backup"]},
    )
    backup: Final = LiteLLM_ManagedVectorStore(is_config=True, vector_store_id="backup", custom_llm_provider="openai")
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([primary, backup]))
    admin: Final = UserAPIKeyAuth(user_role="proxy_admin")
    plan: Final = await resolve_vector_store_fallback_plan(primary, admin)
    assert plan is not None
    assert tuple((target.store_id, target.provider) for target in plan.targets) == (
        ("primary", "azure_ai"),
        ("backup", "openai"),
    )


@pytest.mark.parametrize(
    "field",
    [
        "api_base",
        "api_key",
        "model",
        "extra_headers",
        "fallbacks",
        "_vector_store_fallback_plan",
        "timeout",
        "request_timeout",
    ],
)
def test_search_fallback_rejects_caller_routing_and_connection_overrides(field: str) -> None:
    from fastapi import HTTPException

    from litellm.proxy.vector_store_endpoints.endpoints import reject_caller_fallback_selection

    with pytest.raises(HTTPException) as denied:
        reject_caller_fallback_selection({field: "caller-controlled"}, enabled=True)
    assert denied.value.status_code == 400


def test_store_fallback_settings_are_not_sent_to_provider() -> None:
    from litellm.vector_stores.search_query import store_search_params

    assert store_search_params({"api_base": "https://store.example", "fallbacks": ["backup"]}) == {
        "api_base": "https://store.example",
    }


@pytest.mark.asyncio
async def test_unified_search_fallback_uses_its_own_native_resource_id(monkeypatch: pytest.MonkeyPatch):
    import respx

    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.vector_store_endpoints.utils import resolve_vector_store_fallback_plan
    from litellm.vector_stores.fallbacks import vector_store_fallback_context
    from litellm.vector_stores.main import asearch

    stores: Final = [
        {
            "is_config": True,
            "vector_store_id": store_id,
            "custom_llm_provider": "openai",
            "litellm_params": {
                "api_base": f"https://{host}.native-id.example/v1",
                "api_key": host,
                **({"fallbacks": ["vs_native_backup"]} if host == "primary" else {}),
            },
        }
        for store_id, host in (("vs_native_primary", "primary"), ("vs_native_backup", "backup"))
    ]
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry(stores))
    primary: Final = await get_litellm_managed_vector_store("vs_native_primary")
    assert primary is not None
    plan: Final = await resolve_vector_store_fallback_plan(
        primary, UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    )
    with respx.mock(assert_all_called=True) as upstream:
        upstream.post("https://primary.native-id.example/v1/vector_stores/vs_native_primary/search").respond(503)
        backup: Final = upstream.post(
            "https://backup.native-id.example/v1/vector_stores/vs_native_backup/search"
        ).respond(200, json={"object": "vector_store.search_results.page", "search_query": "q", "data": []})
        with vector_store_fallback_context(plan):
            response: Final = await asearch(
                "vs_native_primary", query="q", client=AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport())
            )
        assert backup.calls[0].request.headers["authorization"] == "Bearer backup"
        assert response == {
            "object": "vector_store.search_results.page",
            "search_query": "q",
            "data": [],
            "_hidden_params": {
                "api_base": "https://backup.native-id.example/v1",
                "additional_headers": {
                    "x-litellm-vector-store-id": "vs_native_backup",
                    "x-litellm-attempted-fallbacks": "1",
                    "x-litellm-model-api-base": "https://backup.native-id.example/v1",
                },
            },
        }


@pytest.mark.asyncio
async def test_unauthorized_primary_does_not_reveal_invalid_fallback_configuration():
    from fastapi import HTTPException

    from litellm.proxy.vector_store_endpoints.utils import resolve_vector_store_fallback_plan
    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore

    primary: Final = LiteLLM_ManagedVectorStore(
        is_config=True,
        vector_store_id="private-primary",
        custom_llm_provider="openai",
        team_id="private-team",
        litellm_params={"fallbacks": ["missing"]},
    )
    with pytest.raises(HTTPException) as denied:
        await resolve_vector_store_fallback_plan(primary, UserAPIKeyAuth(team_id="other-team"))
    assert denied.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("refs", [["missing"], ["primary"], ["backup", "backup"]])
async def test_invalid_registered_fallback_returns_client_error(monkeypatch: pytest.MonkeyPatch, refs: list[str]):
    from fastapi import HTTPException

    from litellm.proxy.vector_store_endpoints.utils import resolve_vector_store_fallback_plan
    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore

    primary: Final = LiteLLM_ManagedVectorStore(
        is_config=True, vector_store_id="primary", custom_llm_provider="openai", litellm_params={"fallbacks": refs}
    )
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([primary]))
    with pytest.raises(HTTPException) as invalid:
        await resolve_vector_store_fallback_plan(primary, UserAPIKeyAuth(user_role="proxy_admin"))
    assert invalid.value.status_code == 400


@pytest.mark.asyncio
async def test_search_rejects_caller_timeout_before_any_provider_attempt(monkeypatch: pytest.MonkeyPatch):
    from fastapi import HTTPException, Request, Response

    from litellm.proxy.vector_store_endpoints.endpoints import vector_store_search
    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore

    stores: Final = [
        LiteLLM_ManagedVectorStore(
            is_config=True,
            vector_store_id=store_id,
            custom_llm_provider="openai",
            litellm_params={"fallbacks": ["timeout-backup"]} if store_id == "timeout-primary" else {},
        )
        for store_id in ("timeout-primary", "timeout-backup")
    ]
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry(stores))

    async def receive():
        return {"type": "http.request", "body": b'{"query":"q","timeout":0.001}', "more_body": False}

    request: Final = Request(
        {"type": "http", "method": "POST", "path": "/v1/vector_stores/timeout-primary/search", "headers": []},
        receive=receive,
    )
    with pytest.raises(HTTPException) as rejected:
        await vector_store_search(
            request=request,
            vector_store_id="timeout-primary",
            fastapi_response=Response(),
            user_api_key_dict=UserAPIKeyAuth(user_role="proxy_admin"),
        )
    assert rejected.value.status_code == 400
    assert rejected.value.detail == "'timeout' is owned by the vector store registration"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key_stores,team_stores,group_stores,strict,allowed",
    [
        (["primary"], ["primary", "backup"], [], False, False),
        (["primary", "backup"], ["primary"], [], False, False),
        (["primary", "backup"], ["primary", "backup"], [], False, True),
        (["primary"], ["primary"], ["backup"], False, True),
        ([], [], [], False, True),
        ([], [], [], True, False),
        (["primary", "backup"], [], [], True, False),
        ([], [], ["primary", "backup"], True, True),
    ],
)
@pytest.mark.parametrize("role", ["internal_user", "proxy_admin"])
async def test_fallback_candidates_require_the_same_key_and_team_search_grants(
    monkeypatch: pytest.MonkeyPatch,
    key_stores: list[str],
    team_stores: list[str],
    group_stores: list[str],
    strict: bool,
    allowed: bool,
    role: str,
):
    from litellm.proxy._types import ProxyException
    from litellm.proxy.vector_store_endpoints.utils import resolve_vector_store_fallback_plan
    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
    from tests.unit.proxy.vector_store_access_group_state import TEAM_ID, ProxyState, make_key, make_team

    proxy: Final = ProxyState(monkeypatch)
    await proxy.add_group("search-grants", group_stores)
    await proxy.add_team(make_team(team_stores, ["search-grants"]))
    caller: Final = make_key(key_stores, ["search-grants"], team_id=TEAM_ID)
    caller.via_virtual_key = True
    caller.user_role = role
    primary: Final = LiteLLM_ManagedVectorStore(
        is_config=True,
        vector_store_id="primary",
        custom_llm_provider="openai",
        team_id=TEAM_ID,
        litellm_params={"fallbacks": ["backup"]},
    )
    backup: Final = LiteLLM_ManagedVectorStore(
        is_config=True, vector_store_id="backup", custom_llm_provider="openai", team_id=TEAM_ID
    )
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([primary, backup]))
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {"vector_store_deny_by_default": strict})
    if allowed:
        plan: Final = await resolve_vector_store_fallback_plan(primary, caller)
        assert plan is not None
        assert tuple(target.store_id for target in plan.targets) == ("primary", "backup")
    else:
        with pytest.raises(ProxyException) as refused:
            await resolve_vector_store_fallback_plan(primary, caller)
        assert refused.value.code == "401"


@pytest.mark.asyncio
@pytest.mark.parametrize("user_stores,allowed", [(["primary"], False), (["primary", "backup"], True)])
async def test_keyless_user_fallback_requires_personal_grants_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch, user_stores: list[str], allowed: bool
):
    from litellm.proxy._types import LiteLLM_UserTable, ProxyException
    from litellm.proxy.vector_store_endpoints.utils import resolve_vector_store_fallback_plan
    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
    from tests.unit.proxy.vector_store_access_group_state import ProxyState, own_permission

    proxy: Final = ProxyState(monkeypatch)
    user: Final = LiteLLM_UserTable(
        user_id="caller", object_permission_id="own", object_permission=own_permission(user_stores)
    )
    await proxy.cache.async_set_cache(key="caller", value=user)
    caller: Final = UserAPIKeyAuth(user_id="caller", user_role="internal_user", is_session_token=True)
    primary: Final = LiteLLM_ManagedVectorStore(
        is_config=True,
        vector_store_id="primary",
        custom_llm_provider="openai",
        litellm_params={"fallbacks": ["backup"]},
    )
    backup: Final = LiteLLM_ManagedVectorStore(is_config=True, vector_store_id="backup", custom_llm_provider="openai")
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([primary, backup]))
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {"vector_store_deny_by_default": True})
    if allowed:
        plan: Final = await resolve_vector_store_fallback_plan(primary, caller)
        assert plan is not None
        assert tuple(target.store_id for target in plan.targets) == ("primary", "backup")
    else:
        with pytest.raises(ProxyException) as refused:
            await resolve_vector_store_fallback_plan(primary, caller)
        assert refused.value.code == "401"


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed_ids,allowed", [(["primary"], False), (["primary", "backup"], True)])
async def test_config_only_fallback_enforces_loaded_caller_grants_without_database(
    monkeypatch: pytest.MonkeyPatch, allowed_ids: list[str], allowed: bool
):
    from litellm.proxy._types import ProxyException
    from litellm.proxy.vector_store_endpoints.utils import resolve_vector_store_fallback_plan
    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
    from tests.unit.proxy.vector_store_access_group_state import own_permission

    caller: Final = UserAPIKeyAuth(object_permission=own_permission(allowed_ids))
    primary: Final = LiteLLM_ManagedVectorStore(
        vector_store_id="primary",
        custom_llm_provider="openai",
        litellm_params={"fallbacks": ["backup"]},
        is_config=True,
    )
    backup: Final = LiteLLM_ManagedVectorStore(vector_store_id="backup", custom_llm_provider="openai", is_config=True)
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([primary, backup]))
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    if allowed:
        plan: Final = await resolve_vector_store_fallback_plan(primary, caller)
        assert plan is not None
        assert tuple(target.store_id for target in plan.targets) == ("primary", "backup")
    else:
        with pytest.raises(ProxyException) as refused:
            await resolve_vector_store_fallback_plan(primary, caller)
        assert refused.value.code == "401"


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("team_grants,allowed", [(["primary"], False), (["primary", "backup"], True)])
async def test_config_only_fallback_enforces_loaded_team_grants_without_database(
    monkeypatch: pytest.MonkeyPatch, team_grants: list[str], allowed: bool, strict: bool
):
    from litellm.proxy._types import ProxyException
    from litellm.proxy.vector_store_endpoints.utils import resolve_vector_store_fallback_plan
    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
    from tests.unit.proxy.vector_store_access_group_state import own_permission

    caller: Final = UserAPIKeyAuth(
        team_id="custom-auth-team",
        object_permission=own_permission(["primary", "backup"]),
        team_object_permission=own_permission(team_grants),
    )
    primary: Final = LiteLLM_ManagedVectorStore(
        vector_store_id="primary",
        custom_llm_provider="openai",
        team_id="custom-auth-team",
        litellm_params={"fallbacks": ["backup"]},
        is_config=True,
    )
    backup: Final = LiteLLM_ManagedVectorStore(
        vector_store_id="backup", custom_llm_provider="openai", team_id="custom-auth-team", is_config=True
    )
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([primary, backup]))
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {"vector_store_deny_by_default": strict})
    if allowed:
        plan: Final = await resolve_vector_store_fallback_plan(primary, caller)
        assert plan is not None
        assert tuple(target.store_id for target in plan.targets) == ("primary", "backup")
    else:
        with pytest.raises(ProxyException) as refused:
            await resolve_vector_store_fallback_plan(primary, caller)
        assert refused.value.code == "401"


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_config,backup_config", [(False, True), (True, False)])
async def test_fallback_resolution_ignores_database_primary_and_rejects_database_backup(
    monkeypatch: pytest.MonkeyPatch, primary_config: bool, backup_config: bool
) -> None:
    from fastapi import HTTPException

    from litellm.proxy.vector_store_endpoints.utils import resolve_vector_store_fallback_plan
    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore

    primary: Final = LiteLLM_ManagedVectorStore(
        vector_store_id="primary",
        custom_llm_provider="openai",
        is_config=primary_config,
        litellm_params={"fallbacks": ["backup"]},
    )
    backup: Final = LiteLLM_ManagedVectorStore(
        vector_store_id="backup",
        custom_llm_provider="openai",
        is_config=backup_config,
    )
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([primary, backup]))
    if not primary_config:
        assert await resolve_vector_store_fallback_plan(primary, UserAPIKeyAuth(user_role="proxy_admin")) is None
        return
    with pytest.raises(HTTPException) as refused:
        await resolve_vector_store_fallback_plan(primary, UserAPIKeyAuth(user_role="proxy_admin"))
    assert refused.value.status_code == 400
    assert "configured in YAML" in refused.value.detail
