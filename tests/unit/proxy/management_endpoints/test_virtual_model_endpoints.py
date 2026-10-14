import json
from collections.abc import Sequence
from datetime import datetime
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from prisma.models import (
    LiteLLM_ProxyModelTable,
    LiteLLM_VirtualModelMemberTable,
    LiteLLM_VirtualModelTable,
)
from pydantic import ValidationError

import litellm
from litellm._uuid import uuid4
from litellm.constants import VIRTUAL_MODEL_PREFIX
from litellm.proxy._types import (
    LitellmUserRoles,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.router_utils.auto_router_model_naming import AUTO_ROUTER_MODEL_PREFIX
from litellm.types.proxy.management_endpoints.virtual_model_endpoints import (
    VirtualModelCreateRequest,
    VirtualModelMemberRequest,
    VirtualModelPatchRequest,
)
from litellm.types.router import VirtualModelRouterState


def _make_admin_user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_id="admin-user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )


def _make_viewer_user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_id="viewer-user",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    )


def _make_internal_user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_id="internal-user",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )


def _make_deployment(model_id: str, model_name: str, team_id: str | None = None) -> LiteLLM_ProxyModelTable:
    return LiteLLM_ProxyModelTable(
        model_id=model_id,
        model_name=model_name,
        litellm_params=json.dumps({"model": f"openai/{model_name}"}),
        model_info=json.dumps({"team_id": team_id}) if team_id else json.dumps({}),
        blocked=False,
        created_at=datetime.now(),
        created_by="test-user",
        updated_at=datetime.now(),
        updated_by="test-user",
    )


def _make_vm(
    virtual_model_id: str,
    created_by: str = "admin",
    team_id: str | None = None,
    members: Sequence[LiteLLM_VirtualModelMemberTable] | None = None,
) -> LiteLLM_VirtualModelTable:
    return LiteLLM_VirtualModelTable(
        virtual_model_id=virtual_model_id,
        virtual_model_name="test-vm",
        description=None,
        routing_strategy="ordered",
        num_retries=None,
        timeout=None,
        stream_timeout=None,
        allowed_fails=None,
        cooldown_time=None,
        team_id=team_id,
        created_by=created_by,
        updated_by=created_by,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        members=list(members) if members is not None else [],
    )


class MockPrismaDB:
    def __init__(self):
        self.litellm_virtualmodeltable = AsyncMock()
        self.litellm_virtualmodelmembertable = AsyncMock()
        self.litellm_proxymodeltable = AsyncMock()
        self.litellm_usertable = AsyncMock()


class MockTransaction:
    def __init__(self):
        self.query_raw = AsyncMock()
        self.litellm_virtualmodeltable = AsyncMock()
        self.litellm_virtualmodelmembertable = AsyncMock()
        self.litellm_virtualmodelmembertable.delete_many = AsyncMock()


class MockTransactionManager:
    def __init__(self, tx: MockTransaction):
        self._tx = tx

    async def __aenter__(self):
        return self._tx

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None


class MockPrismaClient:
    def __init__(self):
        self.db = MockPrismaDB()
        self.writer_db = self.db
        self._tx_manager = None

    def tx(self):
        if self._tx_manager is None:
            tx = MockTransaction()
            self._tx_manager = MockTransactionManager(tx)
        return self._tx_manager


class TestVirtualModelCreate:
    @pytest.mark.asyncio
    async def test_create_persists_rows_and_returns_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        model_id_1: Final = str(uuid4())
        model_id_2: Final = str(uuid4())
        vm_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.create = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
                created_by="admin-user",
                updated_by="admin-user",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                members=[],
            )
        )
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[
                _make_deployment(model_id_1, "model-1"),
                _make_deployment(model_id_2, "model-2"),
            ]
        )
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(return_value=None)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", MagicMock())

        with patch(
            "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
            new_callable=AsyncMock,
        ) as mock_clear_cache:
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            result = await create_virtual_model(
                request_body=VirtualModelCreateRequest(
                    virtual_model_name="test-vm",
                    members=(
                        VirtualModelMemberRequest(model_id=model_id_1, priority=1),
                        VirtualModelMemberRequest(model_id=model_id_2, priority=2),
                    ),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert result.virtual_model_id == vm_id
        prisma.db.litellm_virtualmodeltable.create.assert_called_once()

        create_call = prisma.db.litellm_virtualmodeltable.create.call_args
        create_data = create_call[1]["data"]
        assert create_data["virtual_model_name"] == "test-vm"
        assert "members" in create_data
        assert "create" in create_data["members"]
        member_creates = create_data["members"]["create"]
        assert len(member_creates) == 2
        assert member_creates[0] == {"model_id": model_id_1, "priority": 1}
        assert member_creates[1] == {"model_id": model_id_2, "priority": 2}

        assert len(result.members) == 2
        assert result.members[0].model_id == model_id_1
        assert result.members[0].model_name == "model-1"
        assert result.members[0].priority == 1
        assert result.members[1].model_id == model_id_2
        assert result.members[1].model_name == "model-2"
        assert result.members[1].priority == 2

    @pytest.mark.asyncio
    async def test_internal_user_creates_own_virtual_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm import Router
        from litellm.models.user import LiteLLM_UserTable
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        model_id: Final = str(uuid4())
        vm_id: Final = str(uuid4())

        router: Final = Router(
            model_list=[
                {
                    "model_name": "model-1",
                    "litellm_params": {"model": "gpt-4o", "api_key": "key1"},
                    "model_info": {"id": model_id},
                },
            ],
        )

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.create = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
                created_by="internal-user",
                updated_by="internal-user",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                members=[],
            )
        )
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[_make_deployment(model_id, "model-1")])
        prisma.db.litellm_usertable.find_unique = AsyncMock(
            return_value=LiteLLM_UserTable(user_id="internal-user", models=[], teams=[])
        )

        mock_cache: Final = MagicMock()
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_cache.async_set_cache = AsyncMock()

        internal_user_with_model_access: Final = UserAPIKeyAuth(
            user_id="internal-user",
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=["model-1"],
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", router)
        monkeypatch.setattr(proxy_server, "user_api_key_cache", mock_cache)

        with patch(
            "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
            new_callable=AsyncMock,
        ) as mock_clear_cache:
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            result = await create_virtual_model(
                request_body=VirtualModelCreateRequest(
                    virtual_model_name="test-vm",
                    members=(VirtualModelMemberRequest(model_id=model_id, priority=1),),
                ),
                user_api_key_dict=internal_user_with_model_access,
            )

        assert result.virtual_model_id == vm_id
        create_data = prisma.db.litellm_virtualmodeltable.create.call_args[1]["data"]
        assert create_data["created_by"] == "internal-user"
        assert "team_id" not in create_data

    @pytest.mark.asyncio
    async def test_internal_viewer_returns_403(self) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        with pytest.raises(ProxyException) as exc_info:
            await create_virtual_model(
                request_body=VirtualModelCreateRequest(
                    virtual_model_name="test-vm",
                    members=(VirtualModelMemberRequest(model_id=str(uuid4()), priority=1),),
                ),
                user_api_key_dict=UserAPIKeyAuth(
                    user_id="internal-viewer",
                    user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
                ),
            )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)
        assert "View-only" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_viewer_returns_403(self) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        with pytest.raises(ProxyException) as exc_info:
            await create_virtual_model(
                request_body=VirtualModelCreateRequest(
                    virtual_model_name="test-vm",
                    members=(VirtualModelMemberRequest(model_id=str(uuid4()), priority=1),),
                ),
                user_api_key_dict=_make_viewer_user(),
            )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)

    @pytest.mark.asyncio
    async def test_nonexistent_model_id_returns_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        model_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(return_value=None)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with pytest.raises(ProxyException) as exc_info:
            await create_virtual_model(
                request_body=VirtualModelCreateRequest(
                    virtual_model_name="test-vm",
                    members=(VirtualModelMemberRequest(model_id=model_id, priority=1),),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
        assert "DB-managed" in exc_info.value.message


class TestVirtualModelAuthorization:
    @pytest.mark.asyncio
    async def test_caller_with_virtual_model_name_authorized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Team/key authorization at unit level:
        a caller whose models list holds "virtual_model/<name>" is authorized for that name.
        """
        from litellm import Router
        from litellm.proxy.auth.auth_checks import model_in_access_group

        vm_name = f"{VIRTUAL_MODEL_PREFIX}test-vm"

        router = Router(
            model_list=[
                {
                    "model_name": "dep1-model",
                    "litellm_params": {"model": "gpt-4o", "api_key": "key1"},
                    "model_info": {"id": "dep1"},
                },
            ],
        )

        router.set_virtual_models(
            {
                vm_name: VirtualModelRouterState(
                    members={"dep1": 1},
                )
            }
        )

        team_models = [vm_name]

        is_authorized = model_in_access_group(
            model=vm_name,
            team_models=team_models,
            llm_router=router,
        )

        assert is_authorized is True

    @pytest.mark.asyncio
    async def test_caller_without_virtual_model_name_denied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Team/key authorization at unit level:
        one lacking it is denied.
        """
        from litellm import Router
        from litellm.proxy.auth.auth_checks import model_in_access_group

        vm_name = f"{VIRTUAL_MODEL_PREFIX}test-vm"

        router = Router(
            model_list=[
                {
                    "model_name": "dep1-model",
                    "litellm_params": {"model": "gpt-4o", "api_key": "key1"},
                    "model_info": {"id": "dep1"},
                },
            ],
        )

        router.set_virtual_models(
            {
                vm_name: VirtualModelRouterState(
                    members={"dep1": 1},
                )
            }
        )

        team_models = ["some-other-model"]

        is_authorized = model_in_access_group(
            model=vm_name,
            team_models=team_models,
            llm_router=router,
        )

        assert is_authorized is False

    @pytest.mark.asyncio
    async def test_duplicate_member_ids_returns_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        model_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[_make_deployment(model_id, "model-1")])
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(return_value=None)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with pytest.raises(ProxyException) as exc_info:
            await create_virtual_model(
                request_body=VirtualModelCreateRequest(
                    virtual_model_name="test-vm",
                    members=(
                        VirtualModelMemberRequest(model_id=model_id, priority=1),
                        VirtualModelMemberRequest(model_id=model_id, priority=2),
                    ),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
        assert "duplicate" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_zero_members_returns_400(self) -> None:
        with pytest.raises(ValidationError, match="members"):
            VirtualModelCreateRequest(
                virtual_model_name="test-vm",
                members=(),
            )

    @pytest.mark.asyncio
    async def test_51_members_returns_400(self) -> None:
        members = tuple(VirtualModelMemberRequest(model_id=str(uuid4()), priority=1) for _ in range(51))

        with pytest.raises(ValidationError, match="members"):
            VirtualModelCreateRequest(
                virtual_model_name="test-vm",
                members=members,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("priority", [0, -1, 1001])
    async def test_invalid_priority_returns_400(self, priority: int) -> None:
        with pytest.raises(ValidationError, match="priority"):
            VirtualModelMemberRequest(model_id=str(uuid4()), priority=priority)

    @pytest.mark.asyncio
    async def test_duplicate_name_returns_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from prisma.errors import UniqueViolationError

        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        model_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[_make_deployment(model_id, "model-1")])
        prisma.db.litellm_virtualmodeltable.create = AsyncMock(
            side_effect=UniqueViolationError({"meta": {"target": ["virtual_model_name"]}})
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", MagicMock())

        with patch(
            "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
            new_callable=AsyncMock,
        ) as mock_clear_cache:
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)

            with pytest.raises(ProxyException) as exc_info:
                await create_virtual_model(
                    request_body=VirtualModelCreateRequest(
                        virtual_model_name="existing-vm",
                        members=(VirtualModelMemberRequest(model_id=model_id, priority=1),),
                    ),
                    user_api_key_dict=_make_admin_user(),
                )

        assert exc_info.value.code in (str(status.HTTP_400_BAD_REQUEST), str(status.HTTP_409_CONFLICT))

    @pytest.mark.asyncio
    async def test_name_with_slash_returns_400(self) -> None:
        with pytest.raises(ValidationError, match="virtual_model_name"):
            VirtualModelCreateRequest(
                virtual_model_name="invalid/name",
                members=(VirtualModelMemberRequest(model_id=str(uuid4()), priority=1),),
            )

    @pytest.mark.asyncio
    async def test_name_starting_with_prefix_returns_400(self) -> None:
        with pytest.raises(ValidationError, match="virtual_model_name"):
            VirtualModelCreateRequest(
                virtual_model_name=f"{VIRTUAL_MODEL_PREFIX}test",
                members=(VirtualModelMemberRequest(model_id=str(uuid4()), priority=1),),
            )

    @pytest.mark.asyncio
    async def test_team_scoped_member_returns_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        model_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[_make_deployment(model_id, "model-1", team_id="team-1")]
        )
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(return_value=None)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with pytest.raises(ProxyException) as exc_info:
            await create_virtual_model(
                request_body=VirtualModelCreateRequest(
                    virtual_model_name="test-vm",
                    members=(VirtualModelMemberRequest(model_id=model_id, priority=1),),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
        assert "team" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_auto_router_marker_member_returns_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        model_id: Final = str(uuid4())

        deployment = _make_deployment(model_id, "model-1")
        deployment.litellm_params["model"] = f"{AUTO_ROUTER_MODEL_PREFIX}complexity_router"

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[deployment])
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(return_value=None)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with pytest.raises(ProxyException) as exc_info:
            await create_virtual_model(
                request_body=VirtualModelCreateRequest(
                    virtual_model_name="test-vm",
                    members=(VirtualModelMemberRequest(model_id=model_id, priority=1),),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)

    @pytest.mark.asyncio
    async def test_num_retries_above_3_returns_400(self) -> None:
        with pytest.raises(ValidationError, match="num_retries"):
            VirtualModelCreateRequest(
                virtual_model_name="test-vm",
                members=(VirtualModelMemberRequest(model_id=str(uuid4()), priority=1),),
                num_retries=4,
            )

    @pytest.mark.asyncio
    async def test_unknown_routing_strategy_returns_400(self) -> None:
        with pytest.raises(ValidationError, match="routing_strategy"):
            VirtualModelCreateRequest(
                virtual_model_name="test-vm",
                members=(VirtualModelMemberRequest(model_id=str(uuid4()), priority=1),),
                routing_strategy="unknown-strategy",
            )


class TestVirtualModelList:
    @pytest.mark.asyncio
    async def test_list_returns_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import list_virtual_models

        vm_id: Final = str(uuid4())
        model_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_many = AsyncMock(
            return_value=[
                LiteLLM_VirtualModelTable(
                    virtual_model_id=vm_id,
                    virtual_model_name="test-vm",
                    description=None,
                    routing_strategy="ordered",
                    members=[
                        LiteLLM_VirtualModelMemberTable(
                            virtual_model_id=vm_id,
                            model_id=model_id,
                            priority=1,
                        ),
                    ],
                    created_at=datetime.now(),
                    created_by="admin",
                    updated_at=datetime.now(),
                    updated_by="admin",
                    num_retries=None,
                    timeout=None,
                    stream_timeout=None,
                    allowed_fails=None,
                    cooldown_time=None,
                    team_id=None,
                )
            ]
        )
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[_make_deployment(model_id, "model-1")])

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        result = await list_virtual_models(user_api_key_dict=_make_admin_user())

        assert len(result) == 1
        assert result[0].virtual_model_id == vm_id
        assert result[0].virtual_model_name == "test-vm"
        assert len(result[0].members) == 1
        assert result[0].members[0].model_id == model_id
        assert result[0].members[0].model_name == "model-1"
        assert result[0].members[0].priority == 1
        assert result[0].live_member_count == 1

    @pytest.mark.asyncio
    async def test_list_with_search_passes_where_clause_to_prisma(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import list_virtual_models

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_many = AsyncMock(return_value=[])
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        await list_virtual_models(search="foo", user_api_key_dict=_make_admin_user())

        find_many_call = prisma.db.litellm_virtualmodeltable.find_many.call_args
        where_clause = find_many_call[1].get("where")
        assert where_clause == {"virtual_model_name": {"contains": "foo", "mode": "insensitive"}}

    @pytest.mark.asyncio
    async def test_list_with_search_filters_by_name_substring(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import list_virtual_models

        vm_id_matching: Final = str(uuid4())
        vm_id_other: Final = str(uuid4())

        prisma: Final = MockPrismaClient()

        def _fake_find_many(**kwargs) -> list[LiteLLM_VirtualModelTable]:
            where = kwargs.get("where", {})
            name_filter = where.get("virtual_model_name", {})
            if name_filter:
                contains = name_filter.get("contains", "").lower()
                if contains == "deepseek":
                    return [
                        LiteLLM_VirtualModelTable(
                            virtual_model_id=vm_id_matching,
                            virtual_model_name="DeepSeek-V4",
                            description=None,
                            routing_strategy="ordered",
                            members=[],
                            created_at=datetime.now(),
                            created_by="admin",
                            updated_at=datetime.now(),
                            updated_by="admin",
                            num_retries=None,
                            timeout=None,
                            stream_timeout=None,
                            allowed_fails=None,
                            cooldown_time=None,
                            team_id=None,
                        )
                    ]
            return [
                LiteLLM_VirtualModelTable(
                    virtual_model_id=vm_id_matching,
                    virtual_model_name="DeepSeek-V4",
                    description=None,
                    routing_strategy="ordered",
                    members=[],
                    created_at=datetime.now(),
                    created_by="admin",
                    updated_at=datetime.now(),
                    updated_by="admin",
                    num_retries=None,
                    timeout=None,
                    stream_timeout=None,
                    allowed_fails=None,
                    cooldown_time=None,
                    team_id=None,
                ),
                LiteLLM_VirtualModelTable(
                    virtual_model_id=vm_id_other,
                    virtual_model_name="OtherModel",
                    description=None,
                    routing_strategy="ordered",
                    members=[],
                    created_at=datetime.now(),
                    created_by="admin",
                    updated_at=datetime.now(),
                    updated_by="admin",
                    num_retries=None,
                    timeout=None,
                    stream_timeout=None,
                    allowed_fails=None,
                    cooldown_time=None,
                    team_id=None,
                ),
            ]

        prisma.db.litellm_virtualmodeltable.find_many = AsyncMock(side_effect=_fake_find_many)
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        result = await list_virtual_models(search="deepseek", user_api_key_dict=_make_admin_user())

        assert len(result) == 1
        assert result[0].virtual_model_id == vm_id_matching
        assert result[0].virtual_model_name == "DeepSeek-V4"

    @pytest.mark.asyncio
    async def test_list_with_search_is_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import list_virtual_models

        vm_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_many = AsyncMock(
            return_value=[
                LiteLLM_VirtualModelTable(
                    virtual_model_id=vm_id,
                    virtual_model_name="DeepSeek-V4",
                    description=None,
                    routing_strategy="ordered",
                    members=[],
                    created_at=datetime.now(),
                    created_by="admin",
                    updated_at=datetime.now(),
                    updated_by="admin",
                    num_retries=None,
                    timeout=None,
                    stream_timeout=None,
                    allowed_fails=None,
                    cooldown_time=None,
                    team_id=None,
                )
            ]
        )
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        result = await list_virtual_models(search="deepseek", user_api_key_dict=_make_admin_user())

        find_many_call = prisma.db.litellm_virtualmodeltable.find_many.call_args
        where_clause = find_many_call[1].get("where")
        assert where_clause["virtual_model_name"]["mode"] == "insensitive"
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_with_search_no_matches_returns_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import list_virtual_models

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_many = AsyncMock(return_value=[])
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        result = await list_virtual_models(search="nonexistent", user_api_key_dict=_make_admin_user())

        assert result == ()

    @pytest.mark.asyncio
    async def test_list_with_search_passes_sql_wildcards_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import list_virtual_models

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_many = AsyncMock(return_value=[])
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        await list_virtual_models(search="test%model_name", user_api_key_dict=_make_admin_user())

        find_many_call = prisma.db.litellm_virtualmodeltable.find_many.call_args
        where_clause = find_many_call[1].get("where")
        assert where_clause == {"virtual_model_name": {"contains": "test%model_name", "mode": "insensitive"}}

    @pytest.mark.asyncio
    async def test_list_without_search_has_no_name_filter_in_where(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import list_virtual_models

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_many = AsyncMock(return_value=[])
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        await list_virtual_models(user_api_key_dict=_make_admin_user())

        find_many_call = prisma.db.litellm_virtualmodeltable.find_many.call_args
        where_clause = find_many_call[1].get("where")
        assert where_clause is None

    @pytest.mark.asyncio
    async def test_list_with_empty_search_has_no_name_filter_in_where(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import list_virtual_models

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_many = AsyncMock(return_value=[])
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        await list_virtual_models(search="", user_api_key_dict=_make_admin_user())

        find_many_call = prisma.db.litellm_virtualmodeltable.find_many.call_args
        where_clause = find_many_call[1].get("where")
        assert where_clause is None

    @pytest.mark.asyncio
    async def test_viewer_can_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import list_virtual_models

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_many = AsyncMock(return_value=[])
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        result = await list_virtual_models(user_api_key_dict=_make_viewer_user())

        assert result == ()

    @pytest.mark.asyncio
    async def test_internal_user_lists_only_own_virtual_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import list_virtual_models

        vm_id: Final = str(uuid4())
        model_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_many = AsyncMock(
            return_value=[
                LiteLLM_VirtualModelTable(
                    virtual_model_id=vm_id,
                    virtual_model_name="my-vm",
                    description=None,
                    routing_strategy="ordered",
                    members=[
                        LiteLLM_VirtualModelMemberTable(
                            virtual_model_id=vm_id,
                            model_id=model_id,
                            priority=1,
                        ),
                    ],
                    created_at=datetime.now(),
                    created_by="internal-user",
                    updated_at=datetime.now(),
                    updated_by="internal-user",
                    num_retries=None,
                    timeout=None,
                    stream_timeout=None,
                    allowed_fails=None,
                    cooldown_time=None,
                    team_id=None,
                ),
            ]
        )
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[_make_deployment(model_id, "model-1")]
        )
        prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        result = await list_virtual_models(user_api_key_dict=_make_internal_user())

        assert len(result) == 1
        assert result[0].virtual_model_id == vm_id
        assert result[0].virtual_model_name == "my-vm"
        assert len(result[0].members) == 1
        assert result[0].members[0].model_id == model_id
        assert result[0].members[0].model_name == "model-1"

    @pytest.mark.asyncio
    async def test_internal_user_list_includes_member_teams(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import list_virtual_models

        vm_id_creator: Final = str(uuid4())
        vm_id_team: Final = str(uuid4())
        model_id_creator: Final = str(uuid4())
        model_id_team: Final = str(uuid4())

        prisma: Final = MockPrismaClient()

        call_count = [0]

        def _fake_find_many(**kwargs) -> list[LiteLLM_VirtualModelTable]:
            call_count[0] += 1
            where = kwargs.get("where", {})
            if "created_by" in where:
                return [
                    LiteLLM_VirtualModelTable(
                        virtual_model_id=vm_id_creator,
                        virtual_model_name="creator-vm",
                        description=None,
                        routing_strategy="ordered",
                        members=[
                            LiteLLM_VirtualModelMemberTable(
                                virtual_model_id=vm_id_creator,
                                model_id=model_id_creator,
                                priority=1,
                            ),
                        ],
                        created_at=datetime.now(),
                        created_by="internal-user",
                        updated_at=datetime.now(),
                        updated_by="internal-user",
                        num_retries=None,
                        timeout=None,
                        stream_timeout=None,
                        allowed_fails=None,
                        cooldown_time=None,
                        team_id=None,
                    )
                ]
            elif "team_id" in where:
                return [
                    LiteLLM_VirtualModelTable(
                        virtual_model_id=vm_id_team,
                        virtual_model_name="team-vm",
                        description=None,
                        routing_strategy="ordered",
                        members=[
                            LiteLLM_VirtualModelMemberTable(
                                virtual_model_id=vm_id_team,
                                model_id=model_id_team,
                                priority=1,
                            ),
                        ],
                        created_at=datetime.now(),
                        created_by="other-user",
                        updated_at=datetime.now(),
                        updated_by="other-user",
                        num_retries=None,
                        timeout=None,
                        stream_timeout=None,
                        allowed_fails=None,
                        cooldown_time=None,
                        team_id="team-a",
                    )
                ]
            return []

        prisma.db.litellm_virtualmodeltable.find_many = AsyncMock(side_effect=_fake_find_many)
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[
                _make_deployment(model_id_creator, "model-creator"),
                _make_deployment(model_id_team, "model-team"),
            ]
        )
        user_row: Final = MagicMock(teams=["team-a", "team-b"])
        prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=user_row)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        result = await list_virtual_models(user_api_key_dict=_make_internal_user())

        assert len(result) == 2
        result_ids = frozenset(r.virtual_model_id for r in result)
        assert vm_id_creator in result_ids
        assert vm_id_team in result_ids

        creator_vm = next(r for r in result if r.virtual_model_id == vm_id_creator)
        assert len(creator_vm.members) == 1
        assert creator_vm.members[0].model_id == model_id_creator
        assert creator_vm.members[0].model_name == "model-creator"

        team_vm = next(r for r in result if r.virtual_model_id == vm_id_team)
        assert len(team_vm.members) == 1
        assert team_vm.members[0].model_id == model_id_team
        assert team_vm.members[0].model_name == "model-team"

    @pytest.mark.asyncio
    async def test_internal_user_list_combines_search_with_visibility(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import list_virtual_models

        vm_id: Final = str(uuid4())
        model_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()

        def _fake_find_many(**kwargs) -> list[LiteLLM_VirtualModelTable]:
            where = kwargs.get("where", {})
            if "AND" in where:
                assert "created_by" in where["AND"][0]
                assert "virtual_model_name" in where["AND"][1]
                return [
                    LiteLLM_VirtualModelTable(
                        virtual_model_id=vm_id,
                        virtual_model_name="needle-vm",
                        description=None,
                        routing_strategy="ordered",
                        members=[
                            LiteLLM_VirtualModelMemberTable(
                                virtual_model_id=vm_id,
                                model_id=model_id,
                                priority=1,
                            ),
                        ],
                        created_at=datetime.now(),
                        created_by="internal-user",
                        updated_at=datetime.now(),
                        updated_by="internal-user",
                        num_retries=None,
                        timeout=None,
                        stream_timeout=None,
                        allowed_fails=None,
                        cooldown_time=None,
                        team_id=None,
                    )
                ]
            return []

        prisma.db.litellm_virtualmodeltable.find_many = AsyncMock(side_effect=_fake_find_many)
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[_make_deployment(model_id, "needle-model")]
        )
        prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        result = await list_virtual_models(user_api_key_dict=_make_internal_user(), search="needle")

        assert len(result) == 1
        assert result[0].virtual_model_id == vm_id
        assert result[0].virtual_model_name == "needle-vm"
        assert len(result[0].members) == 1
        assert result[0].members[0].model_id == model_id
        assert result[0].members[0].model_name == "needle-model"


class TestVirtualModelGet:
    @pytest.mark.asyncio
    async def test_get_returns_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import get_virtual_model

        vm_id: Final = str(uuid4())
        model_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id,
                        priority=1,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[_make_deployment(model_id, "model-1")])

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        result = await get_virtual_model(
            virtual_model_id=vm_id,
            user_api_key_dict=_make_admin_user(),
        )

        assert result.virtual_model_id == vm_id

    @pytest.mark.asyncio
    async def test_viewer_can_get(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import get_virtual_model

        vm_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        result = await get_virtual_model(
            virtual_model_id=vm_id,
            user_api_key_dict=_make_viewer_user(),
        )

        assert result.virtual_model_id == vm_id

    @pytest.mark.asyncio
    async def test_internal_user_cannot_get_strangers_virtual_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import get_virtual_model

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=_make_vm(virtual_model_id=str(uuid4()), created_by="someone-else")
        )
        prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with pytest.raises(ProxyException) as exc_info:
            await get_virtual_model(
                virtual_model_id=str(uuid4()),
                user_api_key_dict=_make_internal_user(),
            )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)
        assert "do not have access" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_owner_can_get_own_virtual_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import get_virtual_model

        vm_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=_make_vm(virtual_model_id=vm_id, created_by="internal-user")
        )
        prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        result = await get_virtual_model(
            virtual_model_id=vm_id,
            user_api_key_dict=_make_internal_user(),
        )

        assert result.virtual_model_id == vm_id

    @pytest.mark.asyncio
    async def test_missing_id_returns_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import get_virtual_model

        vm_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(return_value=None)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with pytest.raises(ProxyException) as exc_info:
            await get_virtual_model(
                virtual_model_id=vm_id,
                user_api_key_dict=_make_admin_user(),
            )

        assert exc_info.value.code == str(status.HTTP_404_NOT_FOUND)


class TestVirtualModelPatch:
    @pytest.mark.asyncio
    async def test_patch_add_members_uses_transaction_and_locks_parent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from prisma.errors import UniqueViolationError

        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        model_id_1: Final = str(uuid4())
        model_id_2: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[
                _make_deployment(model_id_1, "model-1"),
                _make_deployment(model_id_2, "model-2"),
            ]
        )
        prisma.db.litellm_virtualmodelmembertable.create_many = AsyncMock(
            side_effect=UniqueViolationError({"meta": {"target": ["model_id"]}})
        )

        mock_tx = MockTransaction()
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        mock_tx.litellm_virtualmodelmembertable.create_many = AsyncMock(
            side_effect=UniqueViolationError({"meta": {"target": ["model_id"]}})
        )
        prisma._tx_manager = MockTransactionManager(mock_tx)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with pytest.raises(ProxyException) as exc_info:
            await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(
                    add_members=(
                        VirtualModelMemberRequest(model_id=model_id_1, priority=1),
                        VirtualModelMemberRequest(model_id=model_id_2, priority=2),
                    )
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
        assert "already member" in exc_info.value.message.lower()
        mock_tx.litellm_virtualmodelmembertable.create_many.assert_called_once()
        create_many_call = mock_tx.litellm_virtualmodelmembertable.create_many.call_args
        assert len(create_many_call[1]["data"]) == 2
        mock_tx.query_raw.assert_called_once()
        lock_call = mock_tx.query_raw.call_args
        assert "FOR UPDATE" in str(lock_call[0][0])

    @pytest.mark.asyncio
    async def test_patch_add_members_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        existing_model_id: Final = str(uuid4())
        new_model_id: Final = str(uuid4())

        updated_vm_with_member: Final = LiteLLM_VirtualModelTable(
            virtual_model_id=vm_id,
            virtual_model_name="test-vm",
            description=None,
            routing_strategy="ordered",
            members=[
                LiteLLM_VirtualModelMemberTable(
                    virtual_model_id=vm_id,
                    model_id=existing_model_id,
                    priority=1,
                ),
                LiteLLM_VirtualModelMemberTable(
                    virtual_model_id=vm_id,
                    model_id=new_model_id,
                    priority=2,
                ),
            ],
            created_at=datetime.now(),
            created_by="admin",
            updated_at=datetime.now(),
            updated_by="admin",
            num_retries=None,
            timeout=None,
            stream_timeout=None,
            allowed_fails=None,
            cooldown_time=None,
            team_id=None,
        )

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            side_effect=[
                updated_vm_with_member,
                updated_vm_with_member,
                updated_vm_with_member,
            ]
        )
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            side_effect=[
                [_make_deployment(new_model_id, "model-2")],
                [_make_deployment(existing_model_id, "model-1"), _make_deployment(new_model_id, "model-2")],
            ]
        )

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=existing_model_id,
                        priority=1,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        mock_tx.litellm_virtualmodelmembertable.create_many = AsyncMock(return_value=1)
        mock_tx.litellm_virtualmodeltable.update = AsyncMock(return_value=updated_vm_with_member)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", MagicMock())

        with (
            patch(
                "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
                new_callable=AsyncMock,
            ) as mock_clear_cache,
            patch(
                "litellm.proxy.common_utils.config_sync_pubsub.publish_config_change_for_object_type",
                new_callable=AsyncMock,
            ) as mock_publish,
        ):
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            result = await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(
                    add_members=(VirtualModelMemberRequest(model_id=new_model_id, priority=2),),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert result.virtual_model_id == vm_id
        assert len(result.members) == 2
        assert result.members[1].model_id == new_model_id
        assert result.members[1].priority == 2

        mock_tx.litellm_virtualmodelmembertable.create_many.assert_called_once()
        create_many_call = mock_tx.litellm_virtualmodelmembertable.create_many.call_args
        assert len(create_many_call[1]["data"]) == 1
        member_data = create_many_call[1]["data"][0]
        assert member_data["model_id"] == new_model_id
        assert member_data["priority"] == 2
        assert member_data["virtual_model_id"] == vm_id

        assert mock_publish.call_count == 2
        published_tables = [call[0][0] for call in mock_publish.call_args_list]
        assert "litellm_virtualmodeltable" in published_tables
        assert "litellm_virtualmodelmembertable" in published_tables

    @pytest.mark.asyncio
    async def test_patch_with_remove_members_accepted_by_schema(self) -> None:
        req = VirtualModelPatchRequest(remove_members=("model-id-1", "model-id-2"))
        assert req.remove_members == ("model-id-1", "model-id-2")

    @pytest.mark.asyncio
    async def test_patch_remove_members_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        model_id_1: Final = str(uuid4())
        model_id_2: Final = str(uuid4())
        model_id_3: Final = str(uuid4())

        updated_vm: Final = LiteLLM_VirtualModelTable(
            virtual_model_id=vm_id,
            virtual_model_name="test-vm",
            description=None,
            routing_strategy="ordered",
            members=[
                LiteLLM_VirtualModelMemberTable(
                    virtual_model_id=vm_id,
                    model_id=model_id_1,
                    priority=1,
                ),
                LiteLLM_VirtualModelMemberTable(
                    virtual_model_id=vm_id,
                    model_id=model_id_3,
                    priority=3,
                ),
            ],
            created_at=datetime.now(),
            created_by="admin",
            updated_at=datetime.now(),
            updated_by="admin-user",
            num_retries=None,
            timeout=None,
            stream_timeout=None,
            allowed_fails=None,
            cooldown_time=None,
            team_id=None,
        )

        prisma: Final = MockPrismaClient()

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_1,
                        priority=1,
                    ),
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_2,
                        priority=2,
                    ),
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_3,
                        priority=3,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_1,
                        priority=1,
                    ),
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_2,
                        priority=2,
                    ),
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_3,
                        priority=3,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        mock_tx.litellm_virtualmodelmembertable.delete_many = AsyncMock(return_value=1)
        mock_tx.litellm_virtualmodeltable.update = AsyncMock(return_value=updated_vm)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", MagicMock())

        with (
            patch(
                "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
                new_callable=AsyncMock,
            ) as mock_clear_cache,
            patch(
                "litellm.proxy.common_utils.config_sync_pubsub.publish_config_change_for_object_type",
                new_callable=AsyncMock,
            ) as mock_publish,
        ):
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            result = await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(
                    remove_members=(model_id_2,),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert result.virtual_model_id == vm_id
        assert len(result.members) == 2
        assert result.members[0].model_id == model_id_1
        assert result.members[0].priority == 1
        assert result.members[1].model_id == model_id_3
        assert result.members[1].priority == 3

        mock_tx.litellm_virtualmodelmembertable.delete_many.assert_called_once()
        delete_many_call = mock_tx.litellm_virtualmodelmembertable.delete_many.call_args
        where_clause = delete_many_call[1]["where"]
        assert where_clause["AND"][0]["virtual_model_id"] == vm_id
        assert model_id_2 in where_clause["AND"][1]["model_id"]["in"]

        assert mock_publish.call_count == 2
        published_tables = [call[0][0] for call in mock_publish.call_args_list]
        assert "litellm_virtualmodeltable" in published_tables
        assert "litellm_virtualmodelmembertable" in published_tables

    @pytest.mark.asyncio
    async def test_patch_remove_members_non_member_returns_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        model_id_existing: Final = str(uuid4())
        model_id_nonmember: Final = str(uuid4())

        prisma: Final = MockPrismaClient()

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_existing,
                        priority=1,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_existing,
                        priority=1,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with pytest.raises(ProxyException) as exc_info:
            await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(
                    remove_members=(model_id_nonmember,),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
        assert "not a member" in exc_info.value.message.lower()
        mock_tx.litellm_virtualmodelmembertable.delete_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_patch_remove_members_same_id_in_add_and_remove_returns_400(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        model_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[_make_deployment(model_id, "model-1")])

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id,
                        priority=1,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id,
                        priority=1,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with pytest.raises(ProxyException) as exc_info:
            await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(
                    add_members=(VirtualModelMemberRequest(model_id=model_id, priority=2),),
                    remove_members=(model_id,),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
        assert "add and remove" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_patch_remove_members_to_zero_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        model_id: Final = str(uuid4())

        updated_vm: Final = LiteLLM_VirtualModelTable(
            virtual_model_id=vm_id,
            virtual_model_name="test-vm",
            description=None,
            routing_strategy="ordered",
            members=[],
            created_at=datetime.now(),
            created_by="admin",
            updated_at=datetime.now(),
            updated_by="admin-user",
            num_retries=None,
            timeout=None,
            stream_timeout=None,
            allowed_fails=None,
            cooldown_time=None,
            team_id=None,
        )

        prisma: Final = MockPrismaClient()

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id,
                        priority=1,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id,
                        priority=1,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        mock_tx.litellm_virtualmodelmembertable.delete_many = AsyncMock(return_value=1)
        mock_tx.litellm_virtualmodeltable.update = AsyncMock(return_value=updated_vm)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", MagicMock())

        with patch(
            "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
            new_callable=AsyncMock,
        ) as mock_clear_cache:
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            result = await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(
                    remove_members=(model_id,),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert result.virtual_model_id == vm_id
        assert len(result.members) == 0
        mock_tx.litellm_virtualmodelmembertable.delete_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_patch_remove_members_does_not_renumber_priorities(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        model_id_1: Final = str(uuid4())
        model_id_2: Final = str(uuid4())
        model_id_3: Final = str(uuid4())

        updated_vm: Final = LiteLLM_VirtualModelTable(
            virtual_model_id=vm_id,
            virtual_model_name="test-vm",
            description=None,
            routing_strategy="ordered",
            members=[
                LiteLLM_VirtualModelMemberTable(
                    virtual_model_id=vm_id,
                    model_id=model_id_1,
                    priority=1,
                ),
                LiteLLM_VirtualModelMemberTable(
                    virtual_model_id=vm_id,
                    model_id=model_id_3,
                    priority=3,
                ),
            ],
            created_at=datetime.now(),
            created_by="admin",
            updated_at=datetime.now(),
            updated_by="admin-user",
            num_retries=None,
            timeout=None,
            stream_timeout=None,
            allowed_fails=None,
            cooldown_time=None,
            team_id=None,
        )

        prisma: Final = MockPrismaClient()

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_1,
                        priority=1,
                    ),
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_2,
                        priority=2,
                    ),
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_3,
                        priority=3,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_1,
                        priority=1,
                    ),
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_2,
                        priority=2,
                    ),
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_3,
                        priority=3,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        mock_tx.litellm_virtualmodelmembertable.delete_many = AsyncMock(return_value=1)
        mock_tx.litellm_virtualmodeltable.update = AsyncMock(return_value=updated_vm)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", MagicMock())

        with patch(
            "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
            new_callable=AsyncMock,
        ) as mock_clear_cache:
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            result = await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(
                    remove_members=(model_id_2,),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert len(result.members) == 2
        assert result.members[0].priority == 1
        assert result.members[1].priority == 3

    @pytest.mark.asyncio
    async def test_patch_remove_members_publishes_member_table_change(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        model_id: Final = str(uuid4())

        updated_vm: Final = LiteLLM_VirtualModelTable(
            virtual_model_id=vm_id,
            virtual_model_name="test-vm",
            description=None,
            routing_strategy="ordered",
            members=[],
            created_at=datetime.now(),
            created_by="admin",
            updated_at=datetime.now(),
            updated_by="admin-user",
            num_retries=None,
            timeout=None,
            stream_timeout=None,
            allowed_fails=None,
            cooldown_time=None,
            team_id=None,
        )

        prisma: Final = MockPrismaClient()

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id,
                        priority=1,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id,
                        priority=1,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        mock_tx.litellm_virtualmodelmembertable.delete_many = AsyncMock(return_value=1)
        mock_tx.litellm_virtualmodeltable.update = AsyncMock(return_value=updated_vm)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", MagicMock())

        with (
            patch(
                "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
                new_callable=AsyncMock,
            ) as mock_clear_cache,
            patch(
                "litellm.proxy.common_utils.config_sync_pubsub.publish_config_change_for_object_type",
                new_callable=AsyncMock,
            ) as mock_publish,
        ):
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(
                    remove_members=(model_id,),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert mock_publish.call_count == 2
        published_tables = [call[0][0] for call in mock_publish.call_args_list]
        assert "litellm_virtualmodeltable" in published_tables
        assert "litellm_virtualmodelmembertable" in published_tables

    @pytest.mark.asyncio
    async def test_patch_remove_members_validation_failure_no_writes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        model_id_existing: Final = str(uuid4())
        model_id_nonmember: Final = str(uuid4())

        prisma: Final = MockPrismaClient()

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_existing,
                        priority=1,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id_existing,
                        priority=1,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with pytest.raises(ProxyException):
            await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(
                    remove_members=(model_id_nonmember,),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        mock_tx.litellm_virtualmodelmembertable.delete_many.assert_not_called()
        mock_tx.litellm_virtualmodeltable.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_patch_re_adding_existing_member_returns_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        model_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[_make_deployment(model_id, "model-1")])

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id,
                        priority=1,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[
                    LiteLLM_VirtualModelMemberTable(
                        virtual_model_id=vm_id,
                        model_id=model_id,
                        priority=1,
                    ),
                ],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with pytest.raises(ProxyException) as exc_info:
            await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(
                    add_members=(VirtualModelMemberRequest(model_id=model_id, priority=2),),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
        assert "already a member" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_patch_tuning_fields_persist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())

        updated_vm: Final = LiteLLM_VirtualModelTable(
            virtual_model_id=vm_id,
            virtual_model_name="test-vm",
            description=None,
            routing_strategy="ordered",
            members=[],
            created_at=datetime.now(),
            created_by="admin",
            updated_at=datetime.now(),
            updated_by="admin-user",
            num_retries=2,
            timeout=30,
            stream_timeout=None,
            allowed_fails=None,
            cooldown_time=None,
            team_id=None,
        )

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        mock_tx.litellm_virtualmodeltable.update = AsyncMock(return_value=updated_vm)

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            side_effect=[
                updated_vm,
                updated_vm,
            ]
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", MagicMock())

        with (
            patch(
                "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
                new_callable=AsyncMock,
            ) as mock_clear_cache,
            patch(
                "litellm.proxy.common_utils.config_sync_pubsub.publish_config_change_for_object_type",
                new_callable=AsyncMock,
            ),
        ):
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(
                    num_retries=2,
                    timeout=30,
                ),
                user_api_key_dict=_make_admin_user(),
            )

        mock_tx.litellm_virtualmodeltable.update.assert_called_once()
        update_call = mock_tx.litellm_virtualmodeltable.update.call_args
        assert update_call[1]["where"]["virtual_model_id"] == vm_id
        assert update_call[1]["data"]["num_retries"] == 2
        assert update_call[1]["data"]["timeout"] == 30

    @pytest.mark.asyncio
    async def test_patch_audit_log_has_after_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())

        updated_vm: Final = LiteLLM_VirtualModelTable(
            virtual_model_id=vm_id,
            virtual_model_name="test-vm",
            description="updated description",
            routing_strategy="ordered",
            members=[],
            created_at=datetime.now(),
            created_by="admin",
            updated_at=datetime.now(),
            updated_by="admin-user",
            num_retries=2,
            timeout=30,
            stream_timeout=None,
            allowed_fails=None,
            cooldown_time=None,
            team_id=None,
        )

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        mock_tx.litellm_virtualmodeltable.update = AsyncMock(return_value=updated_vm)

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            side_effect=[
                updated_vm,
                updated_vm,
            ]
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", MagicMock())

        with (
            patch(
                "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
                new_callable=AsyncMock,
            ) as mock_clear_cache,
            patch(
                "litellm.proxy.common_utils.config_sync_pubsub.publish_config_change_for_object_type",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy.management_endpoints.virtual_model_endpoints.create_object_audit_log",
                new_callable=AsyncMock,
            ) as mock_audit_log,
        ):
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)

            await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(description="updated description"),
                user_api_key_dict=_make_admin_user(),
            )

            mock_audit_log.assert_called_once()
            call_kwargs = mock_audit_log.call_args[1]
            assert call_kwargs["after_value"] is not None
            assert "updated description" in call_kwargs["after_value"]

    @pytest.mark.asyncio
    async def test_patch_name_conflict_race_returns_409(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from prisma.errors import UniqueViolationError

        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        mock_tx.litellm_virtualmodeltable.update = AsyncMock(
            side_effect=UniqueViolationError({"meta": {"target": ["virtual_model_name"]}})
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with pytest.raises(ProxyException) as exc_info:
            await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(virtual_model_name="conflicting-name"),
                user_api_key_dict=_make_admin_user(),
            )

        assert exc_info.value.code == str(status.HTTP_409_CONFLICT)

    @pytest.mark.asyncio
    async def test_non_admin_patch_returns_403(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=_make_vm(virtual_model_id=vm_id, created_by="someone-else")
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with pytest.raises(ProxyException) as exc_info:
            await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(description="updated"),
                user_api_key_dict=_make_internal_user(),
            )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)
        assert "owner, a team admin, or a proxy admin" in exc_info.value.message
        prisma.db.litellm_virtualmodeltable.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_failure_after_member_insert_rolls_back_no_publish_no_audit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        model_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[_make_deployment(model_id, "model-1")])

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        mock_tx.litellm_virtualmodelmembertable.create_many = AsyncMock(return_value=1)
        mock_tx.litellm_virtualmodeltable.update = AsyncMock(side_effect=RuntimeError("DB error"))

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with (
            patch(
                "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy.common_utils.config_sync_pubsub.publish_config_change_for_object_type",
                new_callable=AsyncMock,
            ) as mock_publish,
            patch(
                "litellm.proxy.management_endpoints.virtual_model_endpoints.create_object_audit_log",
                new_callable=AsyncMock,
            ) as mock_audit,
        ):
            with pytest.raises(RuntimeError, match="DB error"):
                await patch_virtual_model(
                    virtual_model_id=vm_id,
                    request_body=VirtualModelPatchRequest(
                        add_members=(VirtualModelMemberRequest(model_id=model_id, priority=1),),
                    ),
                    user_api_key_dict=_make_admin_user(),
                )

        mock_publish.assert_not_called()
        mock_audit.assert_not_called()

    @pytest.mark.asyncio
    async def test_member_conflict_skips_parent_update_and_publish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from prisma.errors import UniqueViolationError

        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        model_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[_make_deployment(model_id, "model-1")])

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        mock_tx.litellm_virtualmodelmembertable.create_many = AsyncMock(
            side_effect=UniqueViolationError({"meta": {"target": ["model_id"]}})
        )
        mock_tx.litellm_virtualmodeltable.update = AsyncMock()

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with patch(
            "litellm.proxy.common_utils.config_sync_pubsub.publish_config_change_for_object_type",
            new_callable=AsyncMock,
        ) as mock_publish:
            with pytest.raises(ProxyException) as exc_info:
                await patch_virtual_model(
                    virtual_model_id=vm_id,
                    request_body=VirtualModelPatchRequest(
                        add_members=(VirtualModelMemberRequest(model_id=model_id, priority=1),),
                    ),
                    user_api_key_dict=_make_admin_user(),
                )

        assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
        mock_tx.litellm_virtualmodeltable.update.assert_not_called()
        mock_publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_cap_uses_in_transaction_membership_not_stale_pre_tx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        model_ids: Final = [str(uuid4()) for _ in range(50)]

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[_make_deployment(mid, f"model-{i}") for i, mid in enumerate(model_ids)]
        )

        existing_members = [
            LiteLLM_VirtualModelMemberTable(
                virtual_model_id=vm_id,
                model_id=model_ids[i],
                priority=i,
            )
            for i in range(50)
        ]

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=existing_members,
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=existing_members,
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        extra_model_id: Final = str(uuid4())
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[_make_deployment(mid, f"model-{i}") for i, mid in enumerate(model_ids)]
            + [_make_deployment(extra_model_id, "model-extra")]
        )

        with pytest.raises(ProxyException) as exc_info:
            await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(
                    add_members=(VirtualModelMemberRequest(model_id=extra_model_id, priority=50),),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
        assert "cannot have more than 50 members" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_patch_swap_at_cap_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        model_ids: Final = [str(uuid4()) for _ in range(50)]

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[_make_deployment(mid, f"model-{i}") for i, mid in enumerate(model_ids)]
        )

        existing_members = [
            LiteLLM_VirtualModelMemberTable(
                virtual_model_id=vm_id,
                model_id=model_ids[i],
                priority=i,
            )
            for i in range(50)
        ]

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=existing_members,
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=existing_members,
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        mock_tx.litellm_virtualmodelmembertable.delete_many = AsyncMock(return_value=1)
        mock_tx.litellm_virtualmodelmembertable.create_many = AsyncMock(return_value=1)
        mock_tx.litellm_virtualmodeltable.update = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=existing_members,
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        new_model_id: Final = str(uuid4())
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[_make_deployment(mid, f"model-{i}") for i, mid in enumerate(model_ids)]
            + [_make_deployment(new_model_id, "model-new")]
        )

        result = await patch_virtual_model(
            virtual_model_id=vm_id,
            request_body=VirtualModelPatchRequest(
                remove_members=(model_ids[0],),
                add_members=(VirtualModelMemberRequest(model_id=new_model_id, priority=50),),
            ),
            user_api_key_dict=_make_admin_user(),
        )

        assert result.members is not None
        assert len(result.members) == 50

    @pytest.mark.asyncio
    async def test_lock_taken_only_when_adding_members(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())

        updated_vm: Final = LiteLLM_VirtualModelTable(
            virtual_model_id=vm_id,
            virtual_model_name="test-vm",
            description="updated",
            routing_strategy="ordered",
            members=[],
            created_at=datetime.now(),
            created_by="admin",
            updated_at=datetime.now(),
            updated_by="admin-user",
            num_retries=None,
            timeout=None,
            stream_timeout=None,
            allowed_fails=None,
            cooldown_time=None,
            team_id=None,
        )

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

        existing_vm: Final = LiteLLM_VirtualModelTable(
            virtual_model_id=vm_id,
            virtual_model_name="test-vm",
            description=None,
            routing_strategy="ordered",
            members=[],
            created_at=datetime.now(),
            created_by="admin",
            updated_at=datetime.now(),
            updated_by="admin",
            num_retries=None,
            timeout=None,
            stream_timeout=None,
            allowed_fails=None,
            cooldown_time=None,
            team_id=None,
        )

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(return_value=existing_vm)
        mock_tx.litellm_virtualmodeltable.update = AsyncMock(return_value=updated_vm)

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            side_effect=[
                updated_vm,
                updated_vm,
            ]
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", MagicMock())

        with (
            patch(
                "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
                new_callable=AsyncMock,
            ) as mock_clear_cache,
            patch(
                "litellm.proxy.common_utils.config_sync_pubsub.publish_config_change_for_object_type",
                new_callable=AsyncMock,
            ) as mock_publish,
        ):
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            result = await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(description="updated"),
                user_api_key_dict=_make_admin_user(),
            )

        assert result.virtual_model_id == vm_id
        assert result.description == "updated"
        mock_tx.query_raw.assert_not_called()
        mock_publish.assert_called_once_with("litellm_virtualmodeltable")

    @pytest.mark.asyncio
    async def test_both_writes_use_same_tx_no_repository_methods(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        model_id: Final = str(uuid4())

        updated_vm: Final = LiteLLM_VirtualModelTable(
            virtual_model_id=vm_id,
            virtual_model_name="test-vm",
            description=None,
            routing_strategy="ordered",
            members=[
                LiteLLM_VirtualModelMemberTable(
                    virtual_model_id=vm_id,
                    model_id=model_id,
                    priority=1,
                ),
            ],
            created_at=datetime.now(),
            created_by="admin",
            updated_at=datetime.now(),
            updated_by="admin-user",
            num_retries=None,
            timeout=None,
            stream_timeout=None,
            allowed_fails=None,
            cooldown_time=None,
            team_id=None,
        )

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[_make_deployment(model_id, "model-1")])

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        mock_tx.litellm_virtualmodelmembertable.create_many = AsyncMock(return_value=1)
        mock_tx.litellm_virtualmodeltable.update = AsyncMock(return_value=updated_vm)

        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(return_value=updated_vm)
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[_make_deployment(model_id, "model-1")])

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", MagicMock())

        with (
            patch(
                "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
                new_callable=AsyncMock,
            ) as mock_clear_cache,
            patch(
                "litellm.proxy.common_utils.config_sync_pubsub.publish_config_change_for_object_type",
                new_callable=AsyncMock,
            ),
        ):
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            result = await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(
                    add_members=(VirtualModelMemberRequest(model_id=model_id, priority=1),),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert result.virtual_model_id == vm_id
        assert len(result.members) == 1
        assert result.members[0].model_id == model_id
        assert mock_tx.litellm_virtualmodelmembertable.create_many.call_count == 1
        assert mock_tx.litellm_virtualmodeltable.update.call_count == 1
        assert prisma.db.litellm_virtualmodeltable.update.call_count == 0
        assert prisma.db.litellm_virtualmodelmembertable.create_many.call_count == 0


class TestVirtualModelDelete:
    @pytest.mark.asyncio
    async def test_delete_removes_row_and_leaves_deployments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import delete_virtual_model

        vm_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                members=[],
                created_at=datetime.now(),
                created_by="admin",
                updated_at=datetime.now(),
                updated_by="admin",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
            )
        )
        prisma.db.litellm_virtualmodeltable.delete = AsyncMock(
            return_value=LiteLLM_VirtualModelTable(
                virtual_model_id=vm_id,
                virtual_model_name="test-vm",
                description=None,
                routing_strategy="ordered",
                num_retries=None,
                timeout=None,
                stream_timeout=None,
                allowed_fails=None,
                cooldown_time=None,
                team_id=None,
                created_by="admin",
                updated_by="admin",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                members=[],
            )
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", MagicMock())

        with patch(
            "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
            new_callable=AsyncMock,
        ) as mock_clear_cache:
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            result = await delete_virtual_model(
                virtual_model_id=vm_id,
                user_api_key_dict=_make_admin_user(),
            )

        assert result.virtual_model_id == vm_id
        prisma.db.litellm_virtualmodeltable.delete.assert_called_once()
        prisma.db.litellm_proxymodeltable.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_admin_delete_returns_403(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import delete_virtual_model

        vm_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=_make_vm(virtual_model_id=vm_id, created_by="someone-else")
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with pytest.raises(ProxyException) as exc_info:
            await delete_virtual_model(
                virtual_model_id=vm_id,
                user_api_key_dict=_make_internal_user(),
            )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)
        assert "owner, a team admin, or a proxy admin" in exc_info.value.message
        prisma.db.litellm_virtualmodeltable.delete.assert_not_called()


class TestVirtualModelOwnershipAuthz:
    def _patch_harness(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vm_id: str,
        row: LiteLLM_VirtualModelTable,
    ):
        from litellm.proxy import proxy_server

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(return_value=row)

        mock_tx = MockTransaction()
        prisma._tx_manager = MockTransactionManager(mock_tx)
        mock_tx.litellm_virtualmodeltable.find_unique = AsyncMock(return_value=row)
        mock_tx.litellm_virtualmodeltable.update = AsyncMock(return_value=row)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
        return prisma

    def _delete_harness(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vm_id: str,
        row: LiteLLM_VirtualModelTable,
    ):
        from litellm.proxy import proxy_server

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(return_value=row)
        prisma.db.litellm_virtualmodeltable.delete = AsyncMock(return_value=row)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
        return prisma

    def _team(self, team_id: str, user_id: str, role: str):
        from litellm.proxy._types import LiteLLM_TeamTable, Member

        return LiteLLM_TeamTable(
            team_id=team_id,
            members_with_roles=[Member(user_id=user_id, role=role)],
        )

    @pytest.mark.asyncio
    async def test_owner_can_patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        prisma: Final = self._patch_harness(
            monkeypatch, vm_id, _make_vm(virtual_model_id=vm_id, created_by="internal-user")
        )

        with (
            patch(
                "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
                new_callable=AsyncMock,
            ) as mock_clear_cache,
            patch(
                "litellm.proxy.common_utils.config_sync_pubsub.publish_config_change_for_object_type",
                new_callable=AsyncMock,
            ),
        ):
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            result = await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(description="owner update"),
                user_api_key_dict=_make_internal_user(),
            )

        assert result.virtual_model_id == vm_id
        prisma._tx_manager._tx.litellm_virtualmodeltable.update.assert_called_once()
        assert (
            prisma._tx_manager._tx.litellm_virtualmodeltable.update.call_args[1]["data"]["updated_by"]
            == "internal-user"
        )

    @pytest.mark.asyncio
    async def test_team_admin_can_patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        self._patch_harness(
            monkeypatch, vm_id, _make_vm(virtual_model_id=vm_id, created_by="someone-else", team_id="team-a")
        )

        with (
            patch(
                "litellm.proxy.auth.auth_checks.get_team_object",
                new_callable=AsyncMock,
                return_value=self._team("team-a", "internal-user", "admin"),
            ),
            patch(
                "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
                new_callable=AsyncMock,
            ) as mock_clear_cache,
            patch(
                "litellm.proxy.common_utils.config_sync_pubsub.publish_config_change_for_object_type",
                new_callable=AsyncMock,
            ),
        ):
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            result = await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(description="team admin update"),
                user_api_key_dict=_make_internal_user(),
            )

        assert result.virtual_model_id == vm_id

    @pytest.mark.asyncio
    async def test_team_member_user_role_cannot_patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        prisma: Final = self._patch_harness(
            monkeypatch, vm_id, _make_vm(virtual_model_id=vm_id, created_by="someone-else", team_id="team-a")
        )

        with patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new_callable=AsyncMock,
            return_value=self._team("team-a", "internal-user", "user"),
        ):
            with pytest.raises(ProxyException) as exc_info:
                await patch_virtual_model(
                    virtual_model_id=vm_id,
                    request_body=VirtualModelPatchRequest(description="member update"),
                    user_api_key_dict=_make_internal_user(),
                )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)
        prisma._tx_manager._tx.litellm_virtualmodeltable.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_other_team_admin_cannot_patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        self._patch_harness(
            monkeypatch, vm_id, _make_vm(virtual_model_id=vm_id, created_by="someone-else", team_id="team-a")
        )

        with patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new_callable=AsyncMock,
            return_value=self._team("team-a", "someone-unrelated", "admin"),
        ):
            with pytest.raises(ProxyException) as exc_info:
                await patch_virtual_model(
                    virtual_model_id=vm_id,
                    request_body=VirtualModelPatchRequest(description="other team admin update"),
                    user_api_key_dict=_make_internal_user(),
                )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)

    @pytest.mark.asyncio
    async def test_owner_cannot_transfer_team(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        prisma: Final = self._patch_harness(
            monkeypatch, vm_id, _make_vm(virtual_model_id=vm_id, created_by="internal-user")
        )

        with pytest.raises(ProxyException) as exc_info:
            await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(team_id="team-a"),
                user_api_key_dict=_make_internal_user(),
            )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)
        assert "Only proxy admins can change" in exc_info.value.message
        prisma._tx_manager._tx.litellm_virtualmodeltable.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_admin_transfers_team(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        prisma: Final = self._patch_harness(
            monkeypatch, vm_id, _make_vm(virtual_model_id=vm_id, created_by="internal-user")
        )

        with (
            patch(
                "litellm.proxy.auth.auth_checks.get_team_object",
                new_callable=AsyncMock,
                return_value=self._team("team-a", "someone-else", "user"),
            ),
            patch(
                "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
                new_callable=AsyncMock,
            ) as mock_clear_cache,
            patch(
                "litellm.proxy.common_utils.config_sync_pubsub.publish_config_change_for_object_type",
                new_callable=AsyncMock,
            ),
        ):
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(team_id="team-a"),
                user_api_key_dict=_make_admin_user(),
            )

        update_call = prisma._tx_manager._tx.litellm_virtualmodeltable.update.call_args
        assert update_call[1]["data"]["team_id"] == "team-a"

    @pytest.mark.asyncio
    async def test_admin_transfer_to_missing_team_returns_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        prisma: Final = self._patch_harness(
            monkeypatch, vm_id, _make_vm(virtual_model_id=vm_id, created_by="internal-user")
        )

        with patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(ProxyException) as exc_info:
                await patch_virtual_model(
                    virtual_model_id=vm_id,
                    request_body=VirtualModelPatchRequest(team_id="missing-team"),
                    user_api_key_dict=_make_admin_user(),
                )

        assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
        prisma._tx_manager._tx.litellm_virtualmodeltable.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_team_denies_non_owner_modify(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        prisma: Final = self._patch_harness(
            monkeypatch, vm_id, _make_vm(virtual_model_id=vm_id, created_by="someone-else", team_id="deleted-team")
        )

        with patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(ProxyException) as exc_info:
                await patch_virtual_model(
                    virtual_model_id=vm_id,
                    request_body=VirtualModelPatchRequest(description="orphaned team update"),
                    user_api_key_dict=_make_internal_user(),
                )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)
        prisma._tx_manager._tx.litellm_virtualmodeltable.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_owner_can_delete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import delete_virtual_model

        vm_id: Final = str(uuid4())
        prisma: Final = self._delete_harness(
            monkeypatch, vm_id, _make_vm(virtual_model_id=vm_id, created_by="internal-user")
        )

        with patch(
            "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
            new_callable=AsyncMock,
        ) as mock_clear_cache:
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            result = await delete_virtual_model(
                virtual_model_id=vm_id,
                user_api_key_dict=_make_internal_user(),
            )

        assert result.virtual_model_id == vm_id
        prisma.db.litellm_virtualmodeltable.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_team_admin_can_delete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import delete_virtual_model

        vm_id: Final = str(uuid4())
        prisma: Final = self._delete_harness(
            monkeypatch, vm_id, _make_vm(virtual_model_id=vm_id, created_by="someone-else", team_id="team-a")
        )

        with (
            patch(
                "litellm.proxy.auth.auth_checks.get_team_object",
                new_callable=AsyncMock,
                return_value=self._team("team-a", "internal-user", "admin"),
            ),
            patch(
                "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
                new_callable=AsyncMock,
            ) as mock_clear_cache,
        ):
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            result = await delete_virtual_model(
                virtual_model_id=vm_id,
                user_api_key_dict=_make_internal_user(),
            )

        assert result.virtual_model_id == vm_id
        prisma.db.litellm_virtualmodeltable.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_team_member_user_role_cannot_delete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import delete_virtual_model

        vm_id: Final = str(uuid4())
        prisma: Final = self._delete_harness(
            monkeypatch, vm_id, _make_vm(virtual_model_id=vm_id, created_by="someone-else", team_id="team-a")
        )

        with patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new_callable=AsyncMock,
            return_value=self._team("team-a", "internal-user", "user"),
        ):
            with pytest.raises(ProxyException) as exc_info:
                await delete_virtual_model(
                    virtual_model_id=vm_id,
                    user_api_key_dict=_make_internal_user(),
                )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)
        prisma.db.litellm_virtualmodeltable.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_team_member_can_get_team_virtual_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import get_virtual_model

        vm_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=_make_vm(virtual_model_id=vm_id, created_by="someone-else", team_id="team-a")
        )
        prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=MagicMock(teams=["team-a"]))

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        result = await get_virtual_model(
            virtual_model_id=vm_id,
            user_api_key_dict=_make_internal_user(),
        )

        assert result.virtual_model_id == vm_id

    @pytest.mark.asyncio
    async def test_team_member_creates_team_virtual_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm import Router
        from litellm.models.user import LiteLLM_UserTable
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        model_id: Final = str(uuid4())
        vm_id: Final = str(uuid4())

        router: Final = Router(
            model_list=[
                {
                    "model_name": "model-1",
                    "litellm_params": {"model": "gpt-4o", "api_key": "key1"},
                    "model_info": {"id": model_id},
                },
            ],
        )

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.create = AsyncMock(
            return_value=_make_vm(virtual_model_id=vm_id, created_by="internal-user", team_id="team-a")
        )
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[_make_deployment(model_id, "model-1")])
        prisma.db.litellm_usertable.find_unique = AsyncMock(
            return_value=LiteLLM_UserTable(user_id="internal-user", models=[], teams=["team-a"])
        )

        mock_cache: Final = MagicMock()
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_cache.async_set_cache = AsyncMock()

        internal_user_with_model_access: Final = UserAPIKeyAuth(
            user_id="internal-user",
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=["model-1"],
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", router)
        monkeypatch.setattr(proxy_server, "user_api_key_cache", mock_cache)

        with (
            patch(
                "litellm.proxy.auth.auth_checks.get_team_object",
                new_callable=AsyncMock,
                return_value=self._team("team-a", "internal-user", "user"),
            ),
            patch(
                "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
                new_callable=AsyncMock,
            ) as mock_clear_cache,
        ):
            mock_clear_cache.return_value = MagicMock(still_desired=None, live_after=None)
            result = await create_virtual_model(
                request_body=VirtualModelCreateRequest(
                    virtual_model_name="team-vm",
                    team_id="team-a",
                    members=(VirtualModelMemberRequest(model_id=model_id, priority=1),),
                ),
                user_api_key_dict=internal_user_with_model_access,
            )

        assert result.virtual_model_id == vm_id
        create_data = prisma.db.litellm_virtualmodeltable.create.call_args[1]["data"]
        assert create_data["team_id"] == "team-a"
        assert create_data["created_by"] == "internal-user"

    @pytest.mark.asyncio
    async def test_non_member_create_with_team_returns_403(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        prisma: Final = MockPrismaClient()
        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new_callable=AsyncMock,
            return_value=self._team("team-a", "someone-unrelated", "user"),
        ):
            with pytest.raises(ProxyException) as exc_info:
                await create_virtual_model(
                    request_body=VirtualModelCreateRequest(
                        virtual_model_name="team-vm",
                        team_id="team-a",
                        members=(VirtualModelMemberRequest(model_id=str(uuid4()), priority=1),),
                    ),
                    user_api_key_dict=_make_internal_user(),
                )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)
        assert "Only team members" in exc_info.value.message
        prisma.db.litellm_virtualmodeltable.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_with_missing_team_returns_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        prisma: Final = MockPrismaClient()
        monkeypatch.setattr(proxy_server, "prisma_client", prisma)

        with patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(ProxyException) as exc_info:
                await create_virtual_model(
                    request_body=VirtualModelCreateRequest(
                        virtual_model_name="team-vm",
                        team_id="missing-team",
                        members=(VirtualModelMemberRequest(model_id=str(uuid4()), priority=1),),
                    ),
                    user_api_key_dict=_make_internal_user(),
                )

        assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
        prisma.db.litellm_virtualmodeltable.create.assert_not_called()


class TestVirtualModelMemberAuthorization:
    def _router_with_models(self, *model_names: str) -> litellm.Router:
        from litellm import Router

        return Router(
            model_list=[
                {
                    "model_name": name,
                    "litellm_params": {"model": "gpt-4o", "api_key": f"key-{name}"},
                    "model_info": {"id": str(uuid4())},
                }
                for name in model_names
            ]
        )

    def _mock_cache(self) -> MagicMock:
        cache: Final = MagicMock()
        cache.async_get_cache = AsyncMock(return_value=None)
        cache.async_set_cache = AsyncMock()
        return cache

    @pytest.mark.asyncio
    async def test_key_with_virtual_model_wildcard_denies_restricted_deployment_create(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from litellm.models.user import LiteLLM_UserTable
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        restricted_model_id: Final = str(uuid4())

        router: Final = self._router_with_models("restricted-model")
        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[_make_deployment(restricted_model_id, "restricted-model")]
        )
        prisma.db.litellm_usertable.find_unique = AsyncMock(
            return_value=LiteLLM_UserTable(user_id="user-with-vm-wildcard", models=[], teams=[])
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", router)
        monkeypatch.setattr(proxy_server, "user_api_key_cache", self._mock_cache())

        user_key: Final = UserAPIKeyAuth(
            user_id="user-with-vm-wildcard",
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=["virtual_model/*"],
        )

        with pytest.raises(ProxyException) as exc_info:
            await create_virtual_model(
                request_body=VirtualModelCreateRequest(
                    virtual_model_name="test-vm",
                    members=(VirtualModelMemberRequest(model_id=restricted_model_id, priority=1),),
                ),
                user_api_key_dict=user_key,
            )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)
        prisma.db.litellm_virtualmodeltable.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_key_with_virtual_model_wildcard_denies_restricted_deployment_patch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from litellm.models.user import LiteLLM_UserTable
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        restricted_model_id: Final = str(uuid4())

        router: Final = self._router_with_models("restricted-model")
        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.find_unique = AsyncMock(
            return_value=_make_vm(virtual_model_id=vm_id, created_by="user-with-vm-wildcard")
        )
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[_make_deployment(restricted_model_id, "restricted-model")]
        )
        prisma.db.litellm_usertable.find_unique = AsyncMock(
            return_value=LiteLLM_UserTable(user_id="user-with-vm-wildcard", models=[], teams=[])
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", router)
        monkeypatch.setattr(proxy_server, "user_api_key_cache", self._mock_cache())

        user_key: Final = UserAPIKeyAuth(
            user_id="user-with-vm-wildcard",
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=["virtual_model/*"],
        )

        with pytest.raises(ProxyException) as exc_info:
            await patch_virtual_model(
                virtual_model_id=vm_id,
                request_body=VirtualModelPatchRequest(
                    add_members=(VirtualModelMemberRequest(model_id=restricted_model_id, priority=1),),
                ),
                user_api_key_dict=user_key,
            )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)

    @pytest.mark.asyncio
    async def test_key_with_allowed_model_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.models.user import LiteLLM_UserTable
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        allowed_model_id: Final = str(uuid4())
        vm_id: Final = str(uuid4())

        router: Final = self._router_with_models("allowed-model")
        prisma: Final = MockPrismaClient()
        prisma.db.litellm_virtualmodeltable.create = AsyncMock(
            return_value=_make_vm(virtual_model_id=vm_id, created_by="user-with-allowed-key")
        )
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[_make_deployment(allowed_model_id, "allowed-model")]
        )
        prisma.db.litellm_usertable.find_unique = AsyncMock(
            return_value=LiteLLM_UserTable(user_id="user-with-allowed-key", models=[], teams=[])
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", router)
        monkeypatch.setattr(proxy_server, "user_api_key_cache", self._mock_cache())

        user_key: Final = UserAPIKeyAuth(
            user_id="user-with-allowed-key",
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=["allowed-model"],
        )

        with patch(
            "litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache",
            new_callable=AsyncMock,
            return_value=MagicMock(still_desired=None, live_after=None),
        ):
            result = await create_virtual_model(
                request_body=VirtualModelCreateRequest(
                    virtual_model_name="test-vm",
                    members=(VirtualModelMemberRequest(model_id=allowed_model_id, priority=1),),
                ),
                user_api_key_dict=user_key,
            )

        assert result.virtual_model_id == vm_id

    @pytest.mark.asyncio
    async def test_user_models_with_empty_key_models_denies_restricted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from litellm.models.user import LiteLLM_UserTable
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        restricted_model_id: Final = str(uuid4())

        router: Final = self._router_with_models("restricted-model")
        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[_make_deployment(restricted_model_id, "restricted-model")]
        )
        prisma.db.litellm_usertable.find_unique = AsyncMock(
            return_value=LiteLLM_UserTable(user_id="user-with-allowed-model", models=["allowed-model"], teams=[])
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", router)
        monkeypatch.setattr(proxy_server, "user_api_key_cache", self._mock_cache())

        user_key: Final = UserAPIKeyAuth(
            user_id="user-with-allowed-model",
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=[],
            team_id=None,
        )

        with pytest.raises(ProxyException) as exc_info:
            await create_virtual_model(
                request_body=VirtualModelCreateRequest(
                    virtual_model_name="test-vm",
                    members=(VirtualModelMemberRequest(model_id=restricted_model_id, priority=1),),
                ),
                user_api_key_dict=user_key,
            )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)
        prisma.db.litellm_virtualmodeltable.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_team_bound_key_with_unrestricted_key_denies_restricted_via_team_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from litellm.models.team import LiteLLM_TeamTable
        from litellm.models.user import LiteLLM_UserTable
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        restricted_model_id: Final = str(uuid4())

        router: Final = self._router_with_models("restricted-model")
        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[_make_deployment(restricted_model_id, "restricted-model")]
        )
        prisma.db.litellm_usertable.find_unique = AsyncMock(
            return_value=LiteLLM_UserTable(user_id="team-user", models=[], teams=["team-a"])
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", router)
        monkeypatch.setattr(proxy_server, "user_api_key_cache", self._mock_cache())

        team_user_key: Final = UserAPIKeyAuth(
            user_id="team-user",
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=[],
            team_id="team-a",
            team_models=["allowed-model"],
        )

        with patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new_callable=AsyncMock,
            return_value=LiteLLM_TeamTable(
                team_id="team-a",
                models=["allowed-model"],
                members_with_roles=[],
            ),
        ):
            with pytest.raises(ProxyException) as exc_info:
                await create_virtual_model(
                    request_body=VirtualModelCreateRequest(
                        virtual_model_name="test-vm",
                        members=(VirtualModelMemberRequest(model_id=restricted_model_id, priority=1),),
                    ),
                    user_api_key_dict=team_user_key,
                )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)
        prisma.db.litellm_virtualmodeltable.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_router_fails_closed_on_create(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.models.user import LiteLLM_UserTable
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        model_id: Final = str(uuid4())

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[_make_deployment(model_id, "some-model")]
        )
        prisma.db.litellm_usertable.find_unique = AsyncMock(
            return_value=LiteLLM_UserTable(user_id="user-no-router", models=[], teams=[])
        )

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "proxy_config", MagicMock())
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock())
        monkeypatch.setattr(proxy_server, "llm_router", None)
        monkeypatch.setattr(proxy_server, "user_api_key_cache", self._mock_cache())

        user_key: Final = UserAPIKeyAuth(
            user_id="user-no-router",
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=[],
        )

        with pytest.raises(ProxyException) as exc_info:
            await create_virtual_model(
                request_body=VirtualModelCreateRequest(
                    virtual_model_name="test-vm",
                    members=(VirtualModelMemberRequest(model_id=model_id, priority=1),),
                ),
                user_api_key_dict=user_key,
            )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)
        assert "router" in exc_info.value.message.lower()


class TestModelNamePrefixReservation:
    @pytest.mark.asyncio
    async def test_model_new_rejects_virtual_model_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.model_management_endpoints import add_new_model
        from litellm.types.router import Deployment

        prisma: Final = MockPrismaClient()
        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "general_settings", {})
        monkeypatch.setattr(proxy_server, "premium_user", True)
        monkeypatch.setattr(proxy_server, "store_model_in_db", True)

        with pytest.raises(ProxyException) as exc_info:
            await add_new_model(
                model_params=Deployment(
                    model_name=f"{VIRTUAL_MODEL_PREFIX}test",
                    litellm_params={"model": "openai/gpt-4"},
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
        assert "reserved" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_patch_model_update_rejects_virtual_model_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.model_management_endpoints import patch_model
        from litellm.types.router import Deployment, updateDeployment

        model_id: Final = str(uuid4())
        db_model: Final = Deployment(
            model_name="existing-model",
            litellm_params={"model": "openai/gpt-4"},
            model_info={"id": model_id},
        )

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(return_value=db_model)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "store_model_in_db", True)
        monkeypatch.setattr(proxy_server, "premium_user", True)

        with pytest.raises(ProxyException) as exc_info:
            await patch_model(
                model_id=model_id,
                patch_data=updateDeployment(model_name=f"{VIRTUAL_MODEL_PREFIX}renamed"),
                user_api_key_dict=_make_admin_user(),
            )

        assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
        assert "reserved" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_update_model_endpoint_rejects_virtual_model_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.model_management_endpoints import update_model
        from litellm.types.router import Deployment, ModelInfo, updateDeployment

        model_id: Final = str(uuid4())
        db_model: Final = Deployment(
            model_name="existing-model",
            litellm_params={"model": "openai/gpt-4"},
            model_info={"id": model_id},
        )

        prisma: Final = MockPrismaClient()
        prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(return_value=db_model)

        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        monkeypatch.setattr(proxy_server, "store_model_in_db", True)
        monkeypatch.setattr(proxy_server, "premium_user", True)

        with pytest.raises(ProxyException) as exc_info:
            await update_model(
                model_params=updateDeployment(
                    model_id=model_id,
                    model_name=f"{VIRTUAL_MODEL_PREFIX}renamed",
                    model_info=ModelInfo(id=model_id),
                ),
                user_api_key_dict=_make_admin_user(),
            )

        assert exc_info.value.code == str(status.HTTP_400_BAD_REQUEST)
        assert "reserved" in exc_info.value.message.lower()


class TestConfigLoadGuard:
    @pytest.mark.asyncio
    async def test_config_load_skips_model_list_with_virtual_model_prefix(self, tmp_path, monkeypatch) -> None:
        from litellm.proxy.proxy_server import ProxyConfig

        config_content = """
model_list:
  - model_name: virtual_model/test
    litellm_params:
      model: openai/gpt-4
  - model_name: normal-model
    litellm_params:
      model: openai/gpt-3.5-turbo
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        proxy_config = ProxyConfig()
        router, model_list, _ = await proxy_config.load_config(router=None, config_file_path=str(config_file))

        assert len(model_list) == 1
        assert model_list[0]["model_name"] == "normal-model"

    @pytest.mark.asyncio
    async def test_config_load_skips_model_group_alias_with_virtual_model_prefix(self, tmp_path, monkeypatch) -> None:
        from litellm.proxy.proxy_server import ProxyConfig

        config_content = """
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4

router_settings:
  model_group_alias:
    virtual_model/alias: gpt-4
    normal-alias: gpt-4
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        proxy_config = ProxyConfig()
        router, _, _ = await proxy_config.load_config(router=None, config_file_path=str(config_file))

        assert "virtual_model/alias" not in router.model_group_alias
        assert "normal-alias" in router.model_group_alias

    @pytest.mark.asyncio
    async def test_config_load_skips_routing_groups_with_virtual_model_prefix(self, tmp_path, monkeypatch) -> None:
        from litellm.proxy.proxy_server import ProxyConfig

        config_content = """
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4

router_settings:
  routing_groups:
    - group_name: virtual_model/group
      routing_strategy: simple-shuffle
      models: [gpt-4]
    - group_name: normal-group
      routing_strategy: simple-shuffle
      models: [gpt-4]
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        proxy_config = ProxyConfig()
        router, _, _ = await proxy_config.load_config(router=None, config_file_path=str(config_file))

        assert len(router.routing_groups) == 1
        assert "normal-group" in router.routing_groups
        assert "virtual_model/group" not in router.routing_groups


class TestVirtualModelRoutingStrategyValidation:
    def test_lar1_strategy_rejected_as_invalid(self) -> None:
        with pytest.raises(ValidationError, match="routing_strategy"):
            VirtualModelCreateRequest(
                virtual_model_name="test-vm",
                members=(VirtualModelMemberRequest(model_id="model-a"),),
                routing_strategy="lar1",
            )


@pytest.mark.asyncio
async def test_get_virtual_model_404_not_found():
    """Test GET /virtual_model/<id> returns 404 when virtual model doesn't exist."""
    from litellm.proxy import proxy_server
    from litellm.proxy.management_endpoints.virtual_model_endpoints import get_virtual_model

    prisma_client = MockPrismaClient()
    prisma_client.db.litellm_virtualmodeltable.find_unique = AsyncMock(return_value=None)

    user = _make_admin_user()

    with patch.object(proxy_server, "prisma_client", prisma_client):
        with pytest.raises(ProxyException) as exc:
            await get_virtual_model(
                virtual_model_id="nonexistent-id",
                user_api_key_dict=user,
            )

    assert exc.value.code in (status.HTTP_404_NOT_FOUND, str(status.HTTP_404_NOT_FOUND))
    assert "not found" in str(exc.value.message).lower()


@pytest.mark.asyncio
async def test_delete_virtual_model_404_not_found():
    """Test DELETE /virtual_model/<id> returns 404 when virtual model doesn't exist."""
    from litellm.proxy import proxy_server
    from litellm.proxy.management_endpoints.virtual_model_endpoints import delete_virtual_model

    prisma_client = MockPrismaClient()
    prisma_client.db.litellm_virtualmodeltable.find_unique = AsyncMock(return_value=None)

    user = _make_admin_user()

    with patch.object(proxy_server, "prisma_client", prisma_client):
        with pytest.raises(ProxyException) as exc:
            await delete_virtual_model(
                virtual_model_id="nonexistent-id",
                user_api_key_dict=user,
            )

    assert exc.value.code in (status.HTTP_404_NOT_FOUND, str(status.HTTP_404_NOT_FOUND))
    assert "not found" in str(exc.value.message).lower()


@pytest.mark.asyncio
async def test_delete_virtual_model_already_deleted():
    """Test DELETE /virtual_model/<id> returns 404 when virtual model was deleted during operation."""
    from litellm.proxy import proxy_server
    from litellm.proxy.management_endpoints.virtual_model_endpoints import delete_virtual_model

    prisma_client = MockPrismaClient()

    existing_vm = LiteLLM_VirtualModelTable(
        virtual_model_id="vm-to-delete",
        virtual_model_name="test-vm",
        created_at=datetime.now(),
        created_by="test-user",
        updated_at=datetime.now(),
        updated_by="test-user",
        routing_strategy="ordered",
    )

    prisma_client.db.litellm_virtualmodeltable.find_unique = AsyncMock(return_value=existing_vm)
    prisma_client.db.litellm_virtualmodeltable.delete = AsyncMock(return_value=None)

    user = _make_admin_user()

    with patch.object(proxy_server, "prisma_client", prisma_client):
        with pytest.raises(ProxyException) as exc:
            await delete_virtual_model(
                virtual_model_id="vm-to-delete",
                user_api_key_dict=user,
            )

    assert exc.value.code in (status.HTTP_404_NOT_FOUND, str(status.HTTP_404_NOT_FOUND))
    assert "already deleted" in str(exc.value.message).lower()


@pytest.mark.asyncio
async def test_create_virtual_model_no_db_connection():
    """Test POST /virtual_model returns 500 when database is not connected."""
    from litellm.proxy import proxy_server
    from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

    user = _make_admin_user()

    request_body = VirtualModelCreateRequest(
        virtual_model_name="test-vm",
        members=(VirtualModelMemberRequest(model_id="dep-1", priority=1),),
    )

    with patch.object(proxy_server, "prisma_client", None):
        with pytest.raises(ProxyException) as exc:
            await create_virtual_model(
                request_body=request_body,
                user_api_key_dict=user,
            )

    assert exc.value.code in (status.HTTP_500_INTERNAL_SERVER_ERROR, str(status.HTTP_500_INTERNAL_SERVER_ERROR))
    assert "db" in str(exc.value.message).lower()


@pytest.mark.asyncio
async def test_list_virtual_models_no_db_connection():
    """Test GET /virtual_model returns 500 when database is not connected."""
    from litellm.proxy import proxy_server
    from litellm.proxy.management_endpoints.virtual_model_endpoints import list_virtual_models

    user = _make_admin_user()

    with patch.object(proxy_server, "prisma_client", None):
        with pytest.raises(ProxyException) as exc:
            await list_virtual_models(
                search=None,
                user_api_key_dict=user,
            )

    assert exc.value.code in (status.HTTP_500_INTERNAL_SERVER_ERROR, str(status.HTTP_500_INTERNAL_SERVER_ERROR))
    assert "db" in str(exc.value.message).lower()


@pytest.mark.asyncio
async def test_reconcile_failure_logs_warning():
    """Test that reconcile failure after write logs warning but doesn't fail."""
    from unittest.mock import patch

    from litellm.proxy.management_endpoints.virtual_model_endpoints import (
        _reconcile_after_write,
    )

    with patch("litellm.proxy.management_endpoints.virtual_model_endpoints.clear_cache") as mock_clear:
        mock_clear.side_effect = RuntimeError("Cache clear failed")

        result = await _reconcile_after_write()

        assert result.still_desired is None
        assert result.live_after is None


@pytest.mark.asyncio
async def test_validate_virtual_model_rejects_none():
    """Test that _validate_virtual_model raises ValueError for None input."""
    from litellm.proxy.management_endpoints.virtual_model_endpoints import (
        _validate_virtual_model,
    )

    with pytest.raises(ValueError, match="non-None"):
        _validate_virtual_model(None)


class TestVirtualModelNameValidation:
    def test_rejects_empty_name(self):
        """Test that empty virtual_model_name is rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            VirtualModelCreateRequest(
                virtual_model_name="",
                members=(VirtualModelMemberRequest(model_id="dep-1", priority=1),),
            )

    def test_rejects_slash_in_name(self):
        """Test that virtual_model_name with slash is rejected."""
        with pytest.raises(ValueError, match="must not contain '/'"):
            VirtualModelCreateRequest(
                virtual_model_name="test/model",
                members=(VirtualModelMemberRequest(model_id="dep-1", priority=1),),
            )

    def test_patch_accepts_none_name(self):
        """Test that VirtualModelPatchRequest accepts None for virtual_model_name."""
        request = VirtualModelPatchRequest(virtual_model_name=None)
        assert request.virtual_model_name is None
