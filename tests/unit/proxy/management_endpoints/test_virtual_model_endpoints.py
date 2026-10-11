import json
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


class MockPrismaDB:
    def __init__(self):
        self.litellm_virtualmodeltable = AsyncMock()
        self.litellm_virtualmodelmembertable = AsyncMock()
        self.litellm_proxymodeltable = AsyncMock()


class MockTransaction:
    def __init__(self):
        self.query_raw = AsyncMock()
        self.litellm_virtualmodeltable = AsyncMock()
        self.litellm_virtualmodelmembertable = AsyncMock()


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
    async def test_non_admin_returns_403(self) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import create_virtual_model

        with pytest.raises(ProxyException) as exc_info:
            await create_virtual_model(
                request_body=VirtualModelCreateRequest(
                    virtual_model_name="test-vm",
                    members=(VirtualModelMemberRequest(model_id=str(uuid4()), priority=1),),
                ),
                user_api_key_dict=_make_internal_user(),
            )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)

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
    async def test_internal_user_cannot_list(self) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import list_virtual_models

        with pytest.raises(ProxyException) as exc_info:
            await list_virtual_models(user_api_key_dict=_make_internal_user())

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)


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
    async def test_internal_user_cannot_get(self) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import get_virtual_model

        with pytest.raises(ProxyException) as exc_info:
            await get_virtual_model(
                virtual_model_id=str(uuid4()),
                user_api_key_dict=_make_internal_user(),
            )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)

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
    async def test_patch_with_remove_members_rejected_by_schema(self) -> None:
        with pytest.raises(ValidationError, match="remove_members"):
            VirtualModelPatchRequest(
                remove_members=["some-id"],
            )

    @pytest.mark.asyncio
    async def test_patch_re_adding_existing_member_returns_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        vm_id: Final = str(uuid4())
        model_id: Final = str(uuid4())

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
    async def test_non_admin_patch_returns_403(self) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import patch_virtual_model

        with pytest.raises(ProxyException) as exc_info:
            await patch_virtual_model(
                virtual_model_id=str(uuid4()),
                request_body=VirtualModelPatchRequest(description="updated"),
                user_api_key_dict=_make_internal_user(),
            )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)

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
    async def test_non_admin_delete_returns_403(self) -> None:
        from litellm.proxy.management_endpoints.virtual_model_endpoints import delete_virtual_model

        with pytest.raises(ProxyException) as exc_info:
            await delete_virtual_model(
                virtual_model_id=str(uuid4()),
                user_api_key_dict=_make_internal_user(),
            )

        assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)


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
