"""
require_managed_files enforcement for litellm/proxy/vector_store_files_endpoints/endpoints.py

Every vector-store file route (create, retrieve, content, update, delete) resolves its
caller-supplied file id through _update_request_data_with_managed_file_id before the
provider call, so the guard lives there once and covers all five.

A raw or forged managed-looking file id has no ownership row, so without the guard it
is attached to a vector store or read back under shared provider credentials.
"""

import base64
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from fastapi import HTTPException

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.route_llm_request import route_request
from litellm.proxy.vector_store_files_endpoints.endpoints import (
    _update_request_data_with_litellm_managed_vector_store_registry,
    _update_request_data_with_managed_file_id,
    _with_managed_file_list_ids,
    _with_provider_file_id_cursors,
)
from litellm.types.utils import CredentialItem, SpecialEnums
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
from litellm.types.vector_store_files import (
    VectorStoreFileListResponse,
    VectorStoreFileObject,
    VectorStoreFileStatus,
)

RAW_FILE_ID = "file-victim-abc123"
CALLER = UserAPIKeyAuth(api_key="sk-test", user_id="attacker-user", team_id="team-b")


@dataclass(frozen=True)
class ManagedResourceAccessCheckerStub:
    file_access: Literal["allow", "deny", "missing"]

    async def can_user_call_unified_file_id(
        self,
        unified_file_id: str,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> bool:
        if self.file_access == "missing":
            raise HTTPException(status_code=404, detail=f"File not found: {unified_file_id}")
        return self.file_access == "allow"

    async def can_user_call_unified_object_id(
        self,
        unified_object_id: str,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> bool:
        return False


@dataclass(frozen=True)
class ManagedFileIdResolverStub:
    resolver: AsyncMock

    async def get_unified_file_ids_for_provider_file_ids(
        self,
        provider_file_ids: Sequence[str],
        user_api_key_dict: UserAPIKeyAuth,
    ) -> Mapping[str, str]:
        return await self.resolver(
            provider_file_ids=provider_file_ids,
            user_api_key_dict=user_api_key_dict,
        )


def _unified_file_id(provider_file_id: str = RAW_FILE_ID) -> str:
    unified = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
        "application/json",
        "victim-unified-id",
        "gpt-4o-mini",
        provider_file_id,
        "gpt-4o-mini-id",
    )
    return base64.urlsafe_b64encode(unified.encode()).decode().rstrip("=")


def _vector_store_file_row(file_id: str) -> VectorStoreFileObject:
    return {
        "id": file_id,
        "object": "vector_store.file",
        "created_at": 1700000000,
        "usage_bytes": 100,
        "vector_store_id": "vs-test",
        "status": VectorStoreFileStatus.COMPLETED,
        "last_error": None,
        "chunking_strategy": {"type": "auto"},
        "attributes": {"source": "test"},
    }


async def _resolve(
    file_id: str,
    file_access: Literal["allow", "deny", "missing"] = "allow",
):
    return await _update_request_data_with_managed_file_id(
        data={"vector_store_id": "vs-test", "file_id": file_id},
        file_id=file_id,
        request=MagicMock(headers={}, query_params={}),
        user_api_key_dict=CALLER,
        managed_files_obj=ManagedResourceAccessCheckerStub(file_access=file_access),
        llm_router=None,
    )


@pytest.mark.parametrize(
    "provider_ids",
    [
        (RAW_FILE_ID, "file-unmanaged-123"),
        ("file-unmanaged-123", RAW_FILE_ID),
    ],
)
@pytest.mark.asyncio
async def test_vector_store_file_list_maps_owned_ids_and_preserves_raw_ids(
    provider_ids: tuple[str, str],
) -> None:
    managed_file_id: Final = _unified_file_id()
    expected_provider_ids: Final = tuple(
        managed_file_id if provider_file_id == RAW_FILE_ID else provider_file_id
        for provider_file_id in provider_ids
    )
    provider_response: Final[VectorStoreFileListResponse] = {
        "object": "list",
        "data": [
            _vector_store_file_row(provider_file_id)
            for provider_file_id in provider_ids
        ],
        "first_id": provider_ids[0],
        "last_id": provider_ids[1],
        "has_more": True,
    }
    original_response: Final = deepcopy(provider_response)
    resolver: Final = AsyncMock(return_value={RAW_FILE_ID: managed_file_id})
    managed_files_obj: Final = ManagedFileIdResolverStub(resolver=resolver)

    response: Final = await _with_managed_file_list_ids(
        response=provider_response,
        managed_files_obj=managed_files_obj,
        user_api_key_dict=CALLER,
    )

    expected_response: Final[VectorStoreFileListResponse] = {
        "object": "list",
        "data": [
            _vector_store_file_row(provider_file_id)
            for provider_file_id in expected_provider_ids
        ],
        "first_id": expected_provider_ids[0],
        "last_id": expected_provider_ids[1],
        "has_more": True,
    }
    assert response == expected_response
    assert provider_response == original_response
    resolver.assert_awaited_once_with(
        provider_file_ids=tuple(dict.fromkeys(provider_ids)),
        user_api_key_dict=CALLER,
    )


@pytest.mark.asyncio
async def test_vector_store_file_list_only_maps_round_trippable_ids() -> None:
    managed_file_id: Final = _unified_file_id("file-model-a")
    provider_response: Final[VectorStoreFileListResponse] = {
        "object": "list",
        "data": [
            _vector_store_file_row("file-model-a"),
            _vector_store_file_row("file-model-b"),
        ],
        "first_id": "file-model-a",
        "last_id": "file-model-b",
        "has_more": False,
    }
    resolver: Final = AsyncMock(
        return_value={
            "file-model-a": managed_file_id,
            "file-model-b": managed_file_id,
        }
    )
    managed_files_obj: Final = ManagedFileIdResolverStub(resolver=resolver)

    response: Final = await _with_managed_file_list_ids(
        response=provider_response,
        managed_files_obj=managed_files_obj,
        user_api_key_dict=CALLER,
    )

    expected_response: Final[VectorStoreFileListResponse] = {
        "object": "list",
        "data": [
            _vector_store_file_row(managed_file_id),
            _vector_store_file_row("file-model-b"),
        ],
        "first_id": managed_file_id,
        "last_id": "file-model-b",
        "has_more": False,
    }
    assert response == expected_response


def test_vector_store_file_list_translates_managed_cursors_and_preserves_raw_after() -> (
    None
):
    managed_file_id: Final = _unified_file_id()

    assert _with_provider_file_id_cursors(
        {"after": managed_file_id, "before": managed_file_id}
    ) == {"after": RAW_FILE_ID, "before": RAW_FILE_ID}
    assert _with_provider_file_id_cursors({"after": RAW_FILE_ID}) == {
        "after": RAW_FILE_ID
    }


@pytest.mark.asyncio
async def test_raw_file_id_rejected_when_managed_files_required():
    with patch.object(litellm, "require_managed_files", True):
        with pytest.raises(HTTPException) as exc:
            await _resolve(RAW_FILE_ID)

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_model_encoded_file_id_rejected_when_managed_files_required():
    """encode_file_id_with_model output is client-forgeable and carries no ownership
    row, so it is not a managed file id."""
    from litellm.proxy.openai_files_endpoints.common_utils import encode_file_id_with_model

    encoded = encode_file_id_with_model(RAW_FILE_ID, "gpt-4o-mini", id_type="file")

    with patch.object(litellm, "require_managed_files", True):
        with pytest.raises(HTTPException) as exc:
            await _resolve(encoded)

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_forged_unified_file_id_rejected_without_ownership_record():
    forged_id = _unified_file_id()
    data = {"vector_store_id": "vs-test", "file_id": forged_id}

    with patch.object(litellm, "require_managed_files", True):
        with pytest.raises(HTTPException) as exc:
            await _update_request_data_with_managed_file_id(
                data=data,
                file_id=forged_id,
                request=MagicMock(headers={}, query_params={}),
                user_api_key_dict=CALLER,
                managed_files_obj=ManagedResourceAccessCheckerStub(file_access="missing"),
                llm_router=None,
            )

    assert exc.value.status_code == 404
    assert data["file_id"] == forged_id


@pytest.mark.asyncio
async def test_other_teams_unified_file_id_rejected():
    with patch.object(litellm, "require_managed_files", True):
        with pytest.raises(HTTPException) as exc:
            await _resolve(_unified_file_id(), file_access="deny")

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_owned_unified_file_id_allowed_when_managed_files_required():
    with patch.object(litellm, "require_managed_files", True):
        data, original = await _resolve(_unified_file_id())

    assert original == _unified_file_id()
    assert data["file_id"] == RAW_FILE_ID


@pytest.mark.asyncio
async def test_raw_file_id_allowed_when_managed_files_not_required():
    with patch.object(litellm, "require_managed_files", False):
        data, original = await _resolve(RAW_FILE_ID)

    assert original is None
    assert data["file_id"] == RAW_FILE_ID


_TEAM_OPENAI_BASE: Final = "https://team-openai.example/v1"
_PROXY_OPENAI_BASE: Final = "https://proxy-openai.example/v1"


def _proxy_and_team_b_openai_router(
    team_litellm_params: Mapping[str, object] = MappingProxyType(
        {"api_key": "team-openai-key", "api_base": _TEAM_OPENAI_BASE}
    ),
) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "proxy-openai-key",
                    "api_base": _PROXY_OPENAI_BASE,
                },
                "model_info": {"id": "proxy-openai"},
            },
            {
                "model_name": "model_name_team-b_6f0c8a52",
                "litellm_params": {"model": "openai/*", **team_litellm_params},
                "model_info": {
                    "id": "team-b-openai",
                    "team_id": "team-b",
                    "team_public_model_name": "openai/*",
                    "db_model": True,
                },
            },
        ]
    )


@dataclass(frozen=True)
class ManagedFileCopiesStub:
    copies: Mapping[str, str]

    async def can_user_call_unified_file_id(self, unified_file_id: str, user_api_key_dict: UserAPIKeyAuth) -> bool:
        return True

    async def can_user_call_unified_object_id(self, unified_object_id: str, user_api_key_dict: UserAPIKeyAuth) -> bool:
        return False

    async def get_model_file_id_mapping(
        self, file_ids: Sequence[str], litellm_parent_otel_span: object
    ) -> Mapping[str, Mapping[str, str]]:
        return {file_id: self.copies for file_id in file_ids}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller_team_id", "expected_base", "expected_authorization"),
    [
        ("team-b", _TEAM_OPENAI_BASE, "Bearer team-openai-key"),
        ("team-c", _PROXY_OPENAI_BASE, "Bearer proxy-openai-key"),
    ],
)
async def test_managed_file_id_reaches_its_upload_deployment_only_for_that_team(
    monkeypatch: pytest.MonkeyPatch, caller_team_id: str, expected_base: str, expected_authorization: str
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    caller: Final = UserAPIKeyAuth(api_key="sk-test", user_id="caller-user", team_id=caller_team_id)
    router: Final = _proxy_and_team_b_openai_router()
    managed_file_id: Final = (
        base64.urlsafe_b64encode(
            SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
                "text/plain", "ingested-unified-id", "openai/*", RAW_FILE_ID, "team-b-openai"
            ).encode()
        )
        .decode()
        .rstrip("=")
    )

    data, _ = await _update_request_data_with_managed_file_id(
        data={"vector_store_id": "vs-test", "file_id": managed_file_id},
        file_id=managed_file_id,
        request=MagicMock(headers={}, query_params={}),
        user_api_key_dict=caller,
        managed_files_obj=ManagedResourceAccessCheckerStub(file_access="allow"),
        llm_router=router,
    )
    with respx.mock(assert_all_called=True) as respx_mock:
        upstream: Final = respx_mock.get(f"{expected_base}/vector_stores/vs-test/files/{RAW_FILE_ID}").mock(
            return_value=httpx.Response(200, json=dict(_vector_store_file_row(RAW_FILE_ID)))
        )
        await (
            await route_request(
                {**data, "custom_llm_provider": "openai"},
                router,
                None,
                "avector_store_file_retrieve",
                user_api_key_dict=caller,
            )
        )

    assert upstream.calls.last.request.headers["Authorization"] == expected_authorization


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller_team_id", "expected_url", "expected_authorization"),
    [
        ("team-b", f"{_TEAM_OPENAI_BASE}/vector_stores/vs-test/files/file-team-copy", "Bearer team-openai-key"),
        ("team-c", f"{_PROXY_OPENAI_BASE}/vector_stores/vs-test/files/file-proxy-copy", "Bearer proxy-openai-key"),
    ],
)
async def test_managed_file_id_with_a_copy_on_the_callers_team_deployment_reaches_that_copy(
    monkeypatch: pytest.MonkeyPatch, caller_team_id: str, expected_url: str, expected_authorization: str
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    caller: Final = UserAPIKeyAuth(api_key="sk-test", user_id="caller-user", team_id=caller_team_id)
    router: Final = _proxy_and_team_b_openai_router()
    managed_file_id: Final = (
        base64.urlsafe_b64encode(
            SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
                "text/plain", "two-copy-unified-id", "openai/*", "file-proxy-copy", "proxy-openai"
            ).encode()
        )
        .decode()
        .rstrip("=")
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        upstream: Final = respx_mock.get(expected_url).mock(
            return_value=httpx.Response(200, json=dict(_vector_store_file_row(expected_url.rsplit("/", 1)[-1])))
        )
        data, _ = await _update_request_data_with_managed_file_id(
            data={"vector_store_id": "vs-test", "file_id": managed_file_id},
            file_id=managed_file_id,
            request=MagicMock(headers={}, query_params={}),
            user_api_key_dict=caller,
            managed_files_obj=ManagedFileCopiesStub(
                copies={"proxy-openai": "file-proxy-copy", "team-b-openai": "file-team-copy"}
            ),
            llm_router=router,
        )
        await (
            await route_request(
                {**data, "custom_llm_provider": "openai"},
                router,
                None,
                "avector_store_file_retrieve",
                user_api_key_dict=caller,
            )
        )

    assert upstream.calls.last.request.headers["Authorization"] == expected_authorization


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "registered_store",
    [
        LiteLLM_ManagedVectorStore(
            vector_store_id="vs-test", custom_llm_provider="openai", litellm_params={"api_key": "store-openai-key"}
        ),
        LiteLLM_ManagedVectorStore(
            vector_store_id="vs-test", custom_llm_provider="openai", litellm_credential_name="store-openai"
        ),
    ],
)
async def test_managed_file_id_on_a_store_with_its_own_credentials_keeps_the_upload_copy(
    monkeypatch: pytest.MonkeyPatch, registered_store: LiteLLM_ManagedVectorStore
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="store-openai", credential_values={"api_key": "store-openai-key"}, credential_info={}
            )
        ],
    )
    caller: Final = UserAPIKeyAuth(api_key="sk-test", user_id="caller-user", team_id="team-b")
    router: Final = _proxy_and_team_b_openai_router()
    managed_file_id: Final = (
        base64.urlsafe_b64encode(
            SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
                "text/plain", "two-copy-unified-id", "openai/*", "file-proxy-copy", "proxy-openai"
            ).encode()
        )
        .decode()
        .rstrip("=")
    )

    data, _ = await _update_request_data_with_managed_file_id(
        data={"vector_store_id": "vs-test", "file_id": managed_file_id},
        file_id=managed_file_id,
        request=MagicMock(headers={}, query_params={}),
        user_api_key_dict=caller,
        managed_files_obj=ManagedFileCopiesStub(
            copies={"proxy-openai": "file-proxy-copy", "team-b-openai": "file-team-copy"}
        ),
        llm_router=router,
        managed_vector_store=registered_store,
    )
    with respx.mock(assert_all_called=True) as respx_mock:
        upstream: Final = respx_mock.get(f"{_PROXY_OPENAI_BASE}/vector_stores/vs-test/files/file-proxy-copy").mock(
            return_value=httpx.Response(200, json=dict(_vector_store_file_row("file-proxy-copy")))
        )
        await (
            await route_request(
                _update_request_data_with_litellm_managed_vector_store_registry(
                    data=data,
                    vector_store_id="vs-test",
                    llm_router=router,
                    managed_vector_store=registered_store,
                    should_lookup_registry=False,
                ),
                router,
                None,
                "avector_store_file_retrieve",
                user_api_key_dict=caller,
            )
        )

    assert upstream.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_type", "method", "reply"),
    [
        ("avector_store_file_retrieve", "GET", dict(_vector_store_file_row("file-proxy-copy"))),
        (
            "avector_store_file_delete",
            "DELETE",
            {"id": "file-proxy-copy", "object": "vector_store.file.deleted", "deleted": True},
        ),
    ],
)
async def test_managed_file_id_on_a_store_holding_only_the_upload_copy_reaches_that_copy(
    monkeypatch: pytest.MonkeyPatch, route_type: str, method: str, reply: Mapping[str, object]
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    caller: Final = UserAPIKeyAuth(api_key="sk-test", user_id="caller-user", team_id="team-b")
    router: Final = _proxy_and_team_b_openai_router()
    managed_file_id: Final = (
        base64.urlsafe_b64encode(
            SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
                "text/plain", "two-copy-unified-id", "openai/*", "file-proxy-copy", "proxy-openai"
            ).encode()
        )
        .decode()
        .rstrip("=")
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        team_copy_lookup: Final = respx_mock.get(f"{_TEAM_OPENAI_BASE}/vector_stores/vs-test/files/file-team-copy").mock(
            return_value=httpx.Response(404, json={"error": {"message": "No file found", "type": "invalid_request_error"}})
        )
        upstream: Final = respx_mock.request(
            method, f"{_PROXY_OPENAI_BASE}/vector_stores/vs-test/files/file-proxy-copy"
        ).mock(return_value=httpx.Response(200, json=reply))
        data, _ = await _update_request_data_with_managed_file_id(
            data={"vector_store_id": "vs-test", "file_id": managed_file_id},
            file_id=managed_file_id,
            request=MagicMock(headers={}, query_params={}),
            user_api_key_dict=caller,
            managed_files_obj=ManagedFileCopiesStub(
                copies={"proxy-openai": "file-proxy-copy", "team-b-openai": "file-team-copy"}
            ),
            llm_router=router,
        )
        await (await route_request({**data, "custom_llm_provider": "openai"}, router, None, route_type, user_api_key_dict=caller))

    assert team_copy_lookup.call_count == 1
    assert upstream.call_count == 1
    assert upstream.calls.last.request.headers["Authorization"] == "Bearer proxy-openai-key"


@pytest.mark.asyncio
async def test_managed_file_id_attached_by_a_team_key_attaches_the_team_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    caller: Final = UserAPIKeyAuth(api_key="sk-test", user_id="caller-user", team_id="team-b")
    router: Final = _proxy_and_team_b_openai_router()
    managed_file_id: Final = (
        base64.urlsafe_b64encode(
            SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
                "text/plain", "two-copy-unified-id", "openai/*", "file-proxy-copy", "proxy-openai"
            ).encode()
        )
        .decode()
        .rstrip("=")
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        upstream: Final = respx_mock.post(f"{_TEAM_OPENAI_BASE}/vector_stores/vs-test/files").mock(
            return_value=httpx.Response(200, json=dict(_vector_store_file_row("file-team-copy")))
        )
        data, _ = await _update_request_data_with_managed_file_id(
            data={"vector_store_id": "vs-test", "file_id": managed_file_id},
            file_id=managed_file_id,
            request=MagicMock(headers={}, query_params={}),
            user_api_key_dict=caller,
            managed_files_obj=ManagedFileCopiesStub(
                copies={"proxy-openai": "file-proxy-copy", "team-b-openai": "file-team-copy"}
            ),
            llm_router=router,
            attaching=True,
        )
        await (
            await route_request(
                {**data, "custom_llm_provider": "openai"},
                router,
                None,
                "avector_store_file_create",
                user_api_key_dict=caller,
            )
        )

    assert upstream.call_count == 1
    assert json.loads(upstream.calls.last.request.content)["file_id"] == "file-team-copy"
    assert upstream.calls.last.request.headers["Authorization"] == "Bearer team-openai-key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store_deployment", "upstream_base", "attached_copy", "upstream_key"),
    [
        ("proxy-openai", _PROXY_OPENAI_BASE, "file-proxy-copy", "proxy-openai-key"),
        ("team-b-openai", _TEAM_OPENAI_BASE, "file-team-copy", "team-openai-key"),
    ],
)
async def test_managed_file_id_attached_to_a_managed_store_sends_the_copy_on_the_store_deployment(
    monkeypatch: pytest.MonkeyPatch,
    store_deployment: str,
    upstream_base: str,
    attached_copy: str,
    upstream_key: str,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    caller: Final = UserAPIKeyAuth(api_key="sk-test", user_id="caller-user", team_id="team-b")
    router: Final = _proxy_and_team_b_openai_router()
    managed_file_id: Final = (
        base64.urlsafe_b64encode(
            SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
                "text/plain", "two-copy-unified-id", "openai/*", "file-proxy-copy", "proxy-openai"
            ).encode()
        )
        .decode()
        .rstrip("=")
    )
    managed_store_id: Final = (
        base64.urlsafe_b64encode(
            f"litellm_proxy:vector_store;unified_id,store-uuid;target_model_names,openai/*;"
            f"resource_id,vs-managed;model_id,{store_deployment}".encode()
        )
        .decode()
        .rstrip("=")
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        upstream: Final = respx_mock.post(f"{upstream_base}/vector_stores/vs-managed/files").mock(
            return_value=httpx.Response(200, json=dict(_vector_store_file_row(attached_copy)))
        )
        data, _ = await _update_request_data_with_managed_file_id(
            data={"vector_store_id": managed_store_id, "file_id": managed_file_id},
            file_id=managed_file_id,
            request=MagicMock(headers={}, query_params={}),
            user_api_key_dict=caller,
            managed_files_obj=ManagedFileCopiesStub(
                copies={"proxy-openai": "file-proxy-copy", "team-b-openai": "file-team-copy"}
            ),
            llm_router=router,
            attaching=True,
        )
        await (
            await route_request(
                {
                    **_update_request_data_with_litellm_managed_vector_store_registry(
                        data=data, vector_store_id=managed_store_id, llm_router=router
                    ),
                    "custom_llm_provider": "openai",
                    "metadata": {"user_api_key_team_id": "team-b"},
                },
                router,
                None,
                "avector_store_file_create",
                user_api_key_dict=caller,
            )
        )

    assert upstream.call_count == 1
    assert json.loads(upstream.calls.last.request.content)["file_id"] == attached_copy
    assert upstream.calls.last.request.headers["Authorization"] == f"Bearer {upstream_key}"


@pytest.mark.asyncio
@pytest.mark.parametrize("credential_setting", [{"timeout": 30}, {"extra_headers": {"X-Team": "team-b"}}])
async def test_team_copy_lookup_works_with_a_named_credential_that_sets_request_options(
    monkeypatch: pytest.MonkeyPatch, credential_setting: Mapping[str, object]
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="team-b-openai",
                credential_values={"api_key": "team-openai-key", "api_base": _TEAM_OPENAI_BASE, **credential_setting},
                credential_info={},
            )
        ],
    )
    caller: Final = UserAPIKeyAuth(api_key="sk-test", user_id="caller-user", team_id="team-b")
    router: Final = _proxy_and_team_b_openai_router(MappingProxyType({"litellm_credential_name": "team-b-openai"}))
    managed_file_id: Final = (
        base64.urlsafe_b64encode(
            SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
                "text/plain", "two-copy-unified-id", "openai/*", "file-proxy-copy", "proxy-openai"
            ).encode()
        )
        .decode()
        .rstrip("=")
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        team_copy: Final = respx_mock.get(f"{_TEAM_OPENAI_BASE}/vector_stores/vs-test/files/file-team-copy").mock(
            return_value=httpx.Response(200, json=dict(_vector_store_file_row("file-team-copy")))
        )
        data, _ = await _update_request_data_with_managed_file_id(
            data={"vector_store_id": "vs-test", "file_id": managed_file_id},
            file_id=managed_file_id,
            request=MagicMock(headers={}, query_params={}),
            user_api_key_dict=caller,
            managed_files_obj=ManagedFileCopiesStub(
                copies={"proxy-openai": "file-proxy-copy", "team-b-openai": "file-team-copy"}
            ),
            llm_router=router,
        )
        await (
            await route_request(
                {**data, "custom_llm_provider": "openai"},
                router,
                None,
                "avector_store_file_retrieve",
                user_api_key_dict=caller,
            )
        )

    assert team_copy.call_count == 2
    assert team_copy.calls.last.request.headers["Authorization"] == "Bearer team-openai-key"
