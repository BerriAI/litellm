"""
require_managed_files enforcement for litellm/proxy/vector_store_files_endpoints/endpoints.py

Every vector-store file route (create, retrieve, content, update, delete) resolves its
caller-supplied file id through _update_request_data_with_managed_file_id before the
provider call, so the guard lives there once and covers all five.

A raw or forged managed-looking file id has no ownership row, so without the guard it
is attached to a vector store or read back under shared provider credentials.
"""

import asyncio
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
from litellm.integrations.custom_logger import CustomLogger
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
    proxy_litellm_params: Mapping[str, object] = MappingProxyType(
        {"api_key": "proxy-openai-key", "api_base": _PROXY_OPENAI_BASE}
    ),
) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*", **proxy_litellm_params},
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
                {**data, "custom_llm_provider": "openai", "metadata": {"user_api_key_team_id": caller.team_id}},
                router,
                None,
                "avector_store_file_retrieve",
                user_api_key_dict=caller,
            )
        )

    assert upstream.calls.last.request.headers["Authorization"] == expected_authorization


_CALLER_OPENAI_BASE: Final = "https://caller-openai.example/v1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller_credentials", "expected_base", "expected_authorization"),
    [
        ({"api_base": _CALLER_OPENAI_BASE}, _TEAM_OPENAI_BASE, "Bearer team-openai-key"),
        ({"api_base": _CALLER_OPENAI_BASE, "api_key": "caller-key"}, _CALLER_OPENAI_BASE, "Bearer caller-key"),
    ],
)
async def test_managed_file_id_sends_its_deployment_key_only_to_that_deployment_base(
    monkeypatch: pytest.MonkeyPatch,
    caller_credentials: Mapping[str, str],
    expected_base: str,
    expected_authorization: str,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    caller: Final = UserAPIKeyAuth(api_key="sk-test", user_id="caller-user", team_id="team-b")
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
        data={"vector_store_id": "vs-test", "file_id": managed_file_id, **caller_credentials},
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
                {**data, "custom_llm_provider": "openai", "metadata": {"user_api_key_team_id": caller.team_id}},
                router,
                None,
                "avector_store_file_retrieve",
                user_api_key_dict=caller,
            )
        )

    assert upstream.calls.last.request.headers["Authorization"] == expected_authorization


class _SuccessRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logged: Final = asyncio.Event()
        self.standard_logging_objects: Final[list[object]] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.standard_logging_objects.append(kwargs.get("standard_logging_object"))
        self.logged.set()


def _gpt_4o_mini_deployment(
    deployment_id: str, api_base: str, order: int, team_id: str | None = None
) -> Mapping[str, object]:
    return {
        "model_name": "gpt-4o-mini" if team_id is None else f"model_name_{team_id}_{deployment_id}",
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": f"{deployment_id}-key",
            "api_base": api_base,
            "order": order,
        },
        "model_info": {"id": deployment_id}
        if team_id is None
        else {"id": deployment_id, "team_id": team_id, "team_public_model_name": "gpt-4o-mini", "db_model": True},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller_team_id", "upload_deployment_id", "upload_base"),
    [
        (None, "upload-openai", "https://upload-openai.example/v1"),
        ("team-b", "team-b-upload", "https://team-b-upload.example/v1"),
    ],
)
async def test_managed_file_id_on_its_upload_deployment_logs_that_deployment_under_its_public_model_name(
    monkeypatch: pytest.MonkeyPatch, caller_team_id: str | None, upload_deployment_id: str, upload_base: str
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    recorder: Final = _SuccessRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    caller: Final = UserAPIKeyAuth(api_key="sk-test", user_id="caller-user", team_id=caller_team_id)
    router: Final = litellm.Router(
        model_list=[
            dict(_gpt_4o_mini_deployment("sibling-openai", "https://sibling-openai.example/v1", order=1)),
            dict(_gpt_4o_mini_deployment("upload-openai", "https://upload-openai.example/v1", order=2)),
            dict(_gpt_4o_mini_deployment("team-b-sibling", "https://team-b-sibling.example/v1", 1, "team-b")),
            dict(_gpt_4o_mini_deployment("team-b-upload", "https://team-b-upload.example/v1", 2, "team-b")),
        ]
    )
    managed_file_id: Final = (
        base64.urlsafe_b64encode(
            SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
                "text/plain", "ingested-unified-id", "gpt-4o-mini", RAW_FILE_ID, upload_deployment_id
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
    with respx.mock(assert_all_called=False) as respx_mock:
        upstream: Final = respx_mock.get(f"{upload_base}/vector_stores/vs-test/files/{RAW_FILE_ID}").mock(
            return_value=httpx.Response(200, json=dict(_vector_store_file_row(RAW_FILE_ID)))
        )
        sibling: Final = respx_mock.route().mock(
            return_value=httpx.Response(200, json=dict(_vector_store_file_row(RAW_FILE_ID)))
        )
        await (
            await route_request(
                {**data, "custom_llm_provider": "openai", "metadata": {"user_api_key_team_id": caller.team_id}},
                router,
                None,
                "avector_store_file_retrieve",
                user_api_key_dict=caller,
            )
        )
        await asyncio.wait_for(recorder.logged.wait(), timeout=10)

    assert (upstream.call_count, sibling.call_count) == (1, 0)
    assert upstream.calls.last.request.headers["Authorization"] == f"Bearer {upload_deployment_id}-key"
    logged: Final = recorder.standard_logging_objects[-1]
    assert isinstance(logged, dict), logged
    assert (logged["model_group"], logged["model_id"]) == ("gpt-4o-mini", upload_deployment_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key_models", "expected_base", "expected_authorization"),
    [
        (["gpt-4o-mini"], "https://allowed-openai.example/v1", "Bearer allowed-key"),
        ([], "https://restricted-openai.example/v1", "Bearer restricted-key"),
    ],
)
async def test_managed_file_id_reaches_its_upload_deployment_only_when_the_key_may_use_it(
    monkeypatch: pytest.MonkeyPatch, key_models: list[str], expected_base: str, expected_authorization: str
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    caller: Final = UserAPIKeyAuth(api_key="sk-test", user_id="caller-user", models=key_models)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "allowed-key",
                    "api_base": "https://allowed-openai.example/v1",
                },
                "model_info": {"id": "gpt-4o-mini-id"},
            },
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "restricted-key",
                    "api_base": "https://restricted-openai.example/v1",
                },
                "model_info": {"id": "restricted-openai"},
            },
        ]
    )
    managed_file_id: Final = (
        base64.urlsafe_b64encode(
            SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
                "text/plain", "forged-unified-id", "gpt-4o-mini", RAW_FILE_ID, "restricted-openai"
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
    ("caller_team_id", "upload_copy_status", "expected_url", "expected_authorization"),
    [
        ("team-b", 404, f"{_TEAM_OPENAI_BASE}/vector_stores/vs-test/files/file-team-copy", "Bearer team-openai-key"),
        ("team-c", 200, f"{_PROXY_OPENAI_BASE}/vector_stores/vs-test/files/file-proxy-copy", "Bearer proxy-openai-key"),
    ],
)
async def test_managed_file_id_with_a_copy_on_the_callers_team_deployment_reaches_that_copy(
    monkeypatch: pytest.MonkeyPatch,
    caller_team_id: str,
    upload_copy_status: int,
    expected_url: str,
    expected_authorization: str,
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
        respx_mock.get(f"{_PROXY_OPENAI_BASE}/vector_stores/vs-test/files/file-proxy-copy").mock(
            return_value=httpx.Response(upload_copy_status, json=dict(_vector_store_file_row("file-proxy-copy")))
        )
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
                {**data, "custom_llm_provider": "openai", "metadata": {"user_api_key_team_id": caller.team_id}},
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

    with respx.mock(assert_all_called=False) as respx_mock:
        team_deployment: Final = respx_mock.route(host="team-openai.example")
        upstream: Final = respx_mock.route(url=f"{_PROXY_OPENAI_BASE}/vector_stores/vs-test/files/file-proxy-copy").mock(
            side_effect=lambda request: httpx.Response(
                200, json=dict(reply) if request.method == method else dict(_vector_store_file_row("file-proxy-copy"))
            )
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
        await (await route_request({**data, "custom_llm_provider": "openai", "metadata": {"user_api_key_team_id": caller.team_id}}, router, None, route_type, user_api_key_dict=caller))

    assert team_deployment.call_count == 0
    assert upstream.calls.last.request.method == method
    assert upstream.calls.last.request.headers["Authorization"] == "Bearer proxy-openai-key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store_holds_upload_copy", "upstream_base", "attached_copy", "upstream_key"),
    [
        (False, _TEAM_OPENAI_BASE, "file-team-copy", "team-openai-key"),
        (True, _PROXY_OPENAI_BASE, "file-proxy-copy", "proxy-openai-key"),
    ],
)
async def test_managed_file_id_attached_by_a_team_key_attaches_the_copy_the_store_can_hold(
    monkeypatch: pytest.MonkeyPatch,
    store_holds_upload_copy: bool,
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

    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.get(f"{_PROXY_OPENAI_BASE}/vector_stores/vs-test/files/file-proxy-copy").mock(
            return_value=(
                httpx.Response(200, json=dict(_vector_store_file_row("file-proxy-copy")))
                if store_holds_upload_copy
                else httpx.Response(404, json={"error": {"message": "No file found", "type": "invalid_request_error"}})
            )
        )
        upstream: Final = respx_mock.post(f"{upstream_base}/vector_stores/vs-test/files").mock(
            return_value=httpx.Response(200, json=dict(_vector_store_file_row(attached_copy)))
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
                {**data, "custom_llm_provider": "openai", "metadata": {"user_api_key_team_id": caller.team_id}},
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
@pytest.mark.parametrize("check_status", [429, 500])
async def test_managed_file_id_whose_store_check_fails_without_a_404_still_reaches_the_upload_copy(
    monkeypatch: pytest.MonkeyPatch, check_status: int
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

    with respx.mock(assert_all_called=False) as respx_mock:
        team_deployment: Final = respx_mock.route(host="team-openai.example")
        upstream: Final = respx_mock.get(f"{_PROXY_OPENAI_BASE}/vector_stores/vs-test/files/file-proxy-copy").mock(
            side_effect=[
                httpx.Response(check_status, json={"error": {"message": "try again", "type": "server_error"}}),
                httpx.Response(200, json=dict(_vector_store_file_row("file-proxy-copy"))),
            ]
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
        response: Final = await (
            await route_request(
                {**data, "custom_llm_provider": "openai", "metadata": {"user_api_key_team_id": caller.team_id}},
                router,
                None,
                "avector_store_file_retrieve",
                user_api_key_dict=caller,
            )
        )

    assert response["id"] == "file-proxy-copy"
    assert upstream.call_count == 2
    assert upstream.calls.last.request.headers["Authorization"] == "Bearer proxy-openai-key"
    assert team_deployment.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_type", "method", "path"),
    [
        ("avector_store_file_create", "POST", "/vector_stores/vs-managed/files"),
        ("avector_store_file_retrieve", "GET", "/vector_stores/vs-managed/files/file-team-copy"),
        ("avector_store_file_delete", "DELETE", "/vector_stores/vs-managed/files/file-team-copy"),
    ],
)
async def test_managed_file_id_on_a_managed_store_reaches_the_store_deployment(
    monkeypatch: pytest.MonkeyPatch, route_type: str, method: str, path: str
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    caller: Final = UserAPIKeyAuth(api_key="sk-test", user_id="caller-user", team_id="team-b")
    router: Final = _proxy_and_team_b_openai_router()
    managed_file_id: Final = (
        base64.urlsafe_b64encode(
            SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
                "text/plain", "team-upload-unified-id", "openai/*", "file-team-copy", "team-b-openai"
            ).encode()
        )
        .decode()
        .rstrip("=")
    )
    managed_store_id: Final = (
        base64.urlsafe_b64encode(
            b"litellm_proxy:vector_store;unified_id,store-uuid;target_model_names,openai/*;"
            b"resource_id,vs-managed;model_id,proxy-openai"
        )
        .decode()
        .rstrip("=")
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        upstream: Final = respx_mock.request(method, f"{_PROXY_OPENAI_BASE}{path}").mock(
            return_value=httpx.Response(200, json=dict(_vector_store_file_row("file-team-copy")))
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
                route_type,
                user_api_key_dict=caller,
            )
        )

    assert upstream.call_count == 1
    assert upstream.calls.last.request.headers["Authorization"] == "Bearer proxy-openai-key"


@pytest.mark.asyncio
async def test_managed_file_id_on_its_upload_deployment_carries_no_options_from_another_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="team-b-openai",
                credential_values={
                    "api_key": "team-openai-key",
                    "api_base": _TEAM_OPENAI_BASE,
                    "extra_headers": {"X-Team": "team-b"},
                },
                credential_info={},
            )
        ],
    )
    caller: Final = UserAPIKeyAuth(api_key="sk-test", user_id="caller-user", team_id="team-b")
    upload_base: Final = "https://team-upload.example/v1"
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "model_name_team-b_6f0c8a52",
                "litellm_params": {"model": "openai/*", "litellm_credential_name": "team-b-openai"},
                "model_info": {
                    "id": "team-b-openai",
                    "team_id": "team-b",
                    "team_public_model_name": "openai/*",
                    "db_model": True,
                },
            },
            {
                "model_name": "model_name_team-b_upload",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "upload-key", "api_base": upload_base},
                "model_info": {
                    "id": "team-b-upload",
                    "team_id": "team-b",
                    "team_public_model_name": "team-upload",
                    "db_model": True,
                },
            },
        ]
    )
    managed_file_id: Final = (
        base64.urlsafe_b64encode(
            SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
                "text/plain", "upload-unified-id", "team-upload", "file-upload-copy", "team-b-upload"
            ).encode()
        )
        .decode()
        .rstrip("=")
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        upstream: Final = respx_mock.get(f"{upload_base}/vector_stores/vs-test/files/file-upload-copy").mock(
            return_value=httpx.Response(200, json=dict(_vector_store_file_row("file-upload-copy")))
        )
        data, _ = await _update_request_data_with_managed_file_id(
            data={"vector_store_id": "vs-test", "file_id": managed_file_id},
            file_id=managed_file_id,
            request=MagicMock(headers={}, query_params={}),
            user_api_key_dict=caller,
            managed_files_obj=ManagedFileCopiesStub(copies={"team-b-upload": "file-upload-copy"}),
            llm_router=router,
        )
        await (
            await route_request(
                {**data, "custom_llm_provider": "openai", "metadata": {"user_api_key_team_id": caller.team_id}},
                router,
                None,
                "avector_store_file_retrieve",
                user_api_key_dict=caller,
            )
        )

    assert upstream.calls.last.request.headers["Authorization"] == "Bearer upload-key"
    assert "X-Team" not in upstream.calls.last.request.headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "credential_setting", [{"timeout": 30}, {"extra_headers": {"X-Proxy": "on"}}, {"no-log": True}]
)
async def test_store_check_sends_the_upload_deployment_named_credential_options(
    monkeypatch: pytest.MonkeyPatch, credential_setting: Mapping[str, object]
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="proxy-openai",
                credential_values={"api_key": "proxy-openai-key", "api_base": _PROXY_OPENAI_BASE, **credential_setting},
                credential_info={},
            )
        ],
    )
    caller: Final = UserAPIKeyAuth(api_key="sk-test", user_id="caller-user", team_id="team-b")
    router: Final = _proxy_and_team_b_openai_router(
        proxy_litellm_params=MappingProxyType({"litellm_credential_name": "proxy-openai"})
    )
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
        upload_copy: Final = respx_mock.get(f"{_PROXY_OPENAI_BASE}/vector_stores/vs-test/files/file-proxy-copy").mock(
            return_value=httpx.Response(200, json=dict(_vector_store_file_row("file-proxy-copy")))
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
                {**data, "custom_llm_provider": "openai", "metadata": {"user_api_key_team_id": caller.team_id}},
                router,
                None,
                "avector_store_file_retrieve",
                user_api_key_dict=caller,
            )
        )

    store_check, call = upload_copy.calls
    assert call.request.headers["Authorization"] == "Bearer proxy-openai-key"
    assert store_check.request.headers.get("X-Proxy") == call.request.headers.get("X-Proxy")
