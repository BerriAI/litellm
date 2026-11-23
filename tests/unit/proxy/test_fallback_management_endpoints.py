"""
Tests for fallback management endpoints

Tests:
1. Create fallback configuration
2. Get fallback configuration
3. Delete fallback configuration
4. Validation tests (invalid models, duplicate fallbacks, etc.)
"""

import asyncio
import copy
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from types import MappingProxyType, SimpleNamespace
from typing import Final, Literal, TypeAlias, get_args
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import JsonValue, TypeAdapter
from redis.exceptions import NoPermissionError

import litellm
from litellm import Router
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.config_resolvers.settings_rules import Section
from litellm.proxy.deps.database import get_database
from litellm.proxy.management_endpoints.fallback_management_endpoints import (
    FallbackCreateRequest,
    FallbackDeleteResponse,
    FallbackGetResponse,
    FallbackResponse,
    create_fallback,
    delete_fallback,
    get_fallback,
)
from litellm.proxy.proxy_server import ProxyConfig, app
from litellm.proxy.utils import evict_config_param, get_config_param, litellm_config_cache
from litellm.repositories.config_repository import ConfigRepository
from litellm.repositories.database import Database
from litellm.router_api import ChangeSet
from litellm.router_api import RouterSettings as RouterSettingsSnapshot
from litellm.types.router import DeploymentTypedDict, LiteLLMParamsTypedDict
from tests._support.native_database import native_database


@pytest.fixture(autouse=True)
async def empty_config_cache() -> AsyncIterator[None]:
    for section in get_args(Section):
        await evict_config_param(section)
    yield
    for section in get_args(Section):
        await evict_config_param(section)


class TestFallbackCreateRequest:
    """Test the FallbackCreateRequest validation"""

    def test_valid_request(self):
        """Test valid fallback request"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4", "claude-3-haiku"],
            fallback_type="general",
        )
        assert request.model == "gpt-3.5-turbo"
        assert request.fallback_models == ["gpt-4", "claude-3-haiku"]
        assert request.fallback_type == "general"

    def test_default_fallback_type(self):
        """Test default fallback type is 'general'"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4"],
        )
        assert request.fallback_type == "general"

    def test_empty_fallback_models(self):
        """Test that empty fallback_models raises validation error"""
        with pytest.raises(ValueError, match="at least 1 item"):
            FallbackCreateRequest(
                model="gpt-3.5-turbo",
                fallback_models=[],
            )

    def test_duplicate_fallback_models(self):
        """Test that duplicate fallback models raise validation error"""
        with pytest.raises(ValueError, match="fallback_models must not contain duplicates"):
            FallbackCreateRequest(
                model="gpt-3.5-turbo",
                fallback_models=["gpt-4", "gpt-4"],
            )

    def test_empty_model_name(self):
        """Test that empty model name raises validation error"""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            FallbackCreateRequest(
                model="",
                fallback_models=["gpt-4"],
            )

    def test_whitespace_model_name(self):
        """Test that whitespace-only model name raises validation error"""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            FallbackCreateRequest(
                model="   ",
                fallback_models=["gpt-4"],
            )

    def test_model_name_trimmed(self):
        """Test that model name is trimmed"""
        request = FallbackCreateRequest(
            model="  gpt-3.5-turbo  ",
            fallback_models=["gpt-4"],
        )
        assert request.model == "gpt-3.5-turbo"

    def test_context_window_fallback_type(self):
        """Test context_window fallback type"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4-32k"],
            fallback_type="context_window",
        )
        assert request.fallback_type == "context_window"

    def test_content_policy_fallback_type(self):
        """Test content_policy fallback type"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4"],
            fallback_type="content_policy",
        )
        assert request.fallback_type == "content_policy"


TEAM_ID: Final = "team-a"
PRIMARY_INTERNAL_NAME: Final = f"model_name_{TEAM_ID}_1a6437cb-4cab-432c-8099-1d7411731a8b"
FALLBACK_INTERNAL_NAME: Final = f"model_name_{TEAM_ID}_83151607-5556-4bbf-ac65-c474dbc64eba"
PRIMARY_DEPLOYMENT_ID: Final = "team-primary-id"
FALLBACK_DEPLOYMENT_ID: Final = "team-fallback-id"

FallbackRules = list[dict[str, list[str]] | dict[str, object] | str]
RouterSettings = dict[str, FallbackRules | str | None]
_ROUTER_SETTINGS_JSON: Final = TypeAdapter(dict[str, JsonValue])


def _litellm_params(mock_response: str | None) -> LiteLLMParamsTypedDict:
    if mock_response is None:
        return {"model": "openai/gpt-5.4-mini", "api_key": "fake"}
    return {"model": "openai/gpt-5.4-mini", "api_key": "fake", "mock_response": mock_response}


def _team_scoped_deployment(
    internal_name: str, public_name: str, deployment_id: str, mock_response: str | None = None
) -> DeploymentTypedDict:
    return {
        "model_name": internal_name,
        "litellm_params": _litellm_params(mock_response),
        "model_info": {"id": deployment_id, "team_id": TEAM_ID, "team_public_model_name": public_name},
    }


def _team_router(primary_mock_response: str | None = None, fallback_mock_response: str | None = None) -> Router:
    return Router(
        model_list=[
            {"model_name": "gpt-5.4-mini", "litellm_params": _litellm_params(None)},
            _team_scoped_deployment(
                PRIMARY_INTERNAL_NAME, "team-primary", PRIMARY_DEPLOYMENT_ID, primary_mock_response
            ),
            _team_scoped_deployment(
                FALLBACK_INTERNAL_NAME, "team-fallback", FALLBACK_DEPLOYMENT_ID, fallback_mock_response
            ),
        ],
        num_retries=0,
    )


class TestCreateFallbackForTeamScopedModels:
    """POST /fallback takes the public name a caller invokes a team-scoped model by, not only the stored internal one"""

    @pytest.fixture
    def router(self) -> Router:
        return _team_router()

    @pytest.fixture
    def stored(self) -> "_StoredRouterSettings":
        return _StoredRouterSettings(None)

    @pytest.fixture
    def prisma_client(self, stored: "_StoredRouterSettings") -> MagicMock:
        return stored.prisma_client

    @pytest.fixture
    def proxy_config(self) -> ProxyConfig:
        return _ConfigFileProxyConfig()

    async def _create(
        self, request: FallbackCreateRequest, router: Router, prisma_client: MagicMock, proxy_config: ProxyConfig
    ) -> FallbackResponse:
        with (
            patch("litellm.proxy.proxy_server.llm_router", router),
            patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            return await create_fallback(request, MagicMock(), fallbacks=ConfigRepository(prisma_client))

    async def test_public_names_create_a_rule_keyed_on_the_public_name(
        self, router: Router, stored: "_StoredRouterSettings", prisma_client: MagicMock, proxy_config: MagicMock
    ) -> None:
        request: Final = FallbackCreateRequest(model="team-primary", fallback_models=["team-fallback"])

        response: Final = await self._create(request, router, prisma_client, proxy_config)

        assert response.model == "team-primary"
        assert response.fallback_models == ["team-fallback"]
        assert router.settings().fallbacks == [{"team-primary": ["team-fallback"]}]
        assert stored.stored_fallbacks() == [{"team-primary": ["team-fallback"]}]

    async def test_internal_names_keep_working(
        self, router: Router, prisma_client: MagicMock, proxy_config: MagicMock
    ) -> None:
        request: Final = FallbackCreateRequest(model=PRIMARY_INTERNAL_NAME, fallback_models=[FALLBACK_INTERNAL_NAME])

        response: Final = await self._create(request, router, prisma_client, proxy_config)

        assert router.settings().fallbacks == [{PRIMARY_INTERNAL_NAME: [FALLBACK_INTERNAL_NAME]}]
        assert response.model == PRIMARY_INTERNAL_NAME

    async def test_public_fallback_target_behind_a_gateway_primary(
        self, router: Router, prisma_client: MagicMock, proxy_config: MagicMock
    ) -> None:
        request: Final = FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"])

        await self._create(request, router, prisma_client, proxy_config)

        assert router.settings().fallbacks == [{"gpt-5.4-mini": ["team-fallback"]}]

    async def test_unknown_name_is_rejected_and_the_error_names_the_public_names(
        self, router: Router, prisma_client: MagicMock, proxy_config: MagicMock
    ) -> None:
        request: Final = FallbackCreateRequest(model="team-missing", fallback_models=["team-fallback"])

        with pytest.raises(HTTPException) as exc_info:
            await self._create(request, router, prisma_client, proxy_config)

        assert exc_info.value.status_code == 404
        assert {"team-primary", "team-fallback", "gpt-5.4-mini"} <= set(exc_info.value.detail["available_models"])

    async def test_a_team_request_fails_over_to_the_public_name_fallback(
        self, prisma_client: MagicMock, proxy_config: MagicMock
    ) -> None:
        router: Final = _team_router(primary_mock_response="litellm.RateLimitError", fallback_mock_response="pong")
        request: Final = FallbackCreateRequest(model="team-primary", fallback_models=["team-fallback"])
        await self._create(request, router, prisma_client, proxy_config)

        response: Final = await router.acompletion(
            model="team-primary",
            messages=[{"role": "user", "content": "ping"}],
            metadata={"user_api_key_team_id": TEAM_ID},
        )

        assert response.choices[0].message.content == "pong"
        assert response._hidden_params["model_id"] == FALLBACK_DEPLOYMENT_ID


def _config_row(router_settings: RouterSettings) -> SimpleNamespace:
    return SimpleNamespace(param_name="router_settings", param_value=router_settings)


class _ConfigFileProxyConfig(ProxyConfig):
    def __init__(self, config_file: RouterSettings | None = None) -> None:
        super().__init__()
        self.config_file: RouterSettings = config_file or {}
        self.router_settings.load_yaml(_ROUTER_SETTINGS_JSON.validate_python(self.config_file))

    async def _get_config_from_file(self, config_file_path: str | None = None) -> dict[str, RouterSettings]:
        return {"router_settings": copy.deepcopy(self.config_file)}


NATIVE: Final = native_database()


class _ConfigTransaction:
    def __init__(self, stored: "_StoredRouterSettings") -> None:
        self._stored: Final = stored
        self.staged: SimpleNamespace | None = None
        self.holds_lock = False

    @property
    def litellm_config(self) -> "_ConfigTransaction":
        return self

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, int]]:
        if "router_settings" not in args or self.holds_lock:
            return ()
        if self._stored.lock.locked():
            self._stored.lock_contended.set()
        await self._stored.lock.acquire()
        self.holds_lock = True
        return ({"locked": 1},)

    async def find_unique(self, where: Mapping[str, str]) -> SimpleNamespace | None:
        visible: Final = self.staged or copy.deepcopy(self._stored.row)
        await self._stored.after_read()
        return visible

    async def find_first(self, where: Mapping[str, str]) -> SimpleNamespace | None:
        return await self.find_unique(where)

    async def upsert(self, where: Mapping[str, str], data: Mapping[str, Mapping[str, str]]) -> None:
        if self._stored.fail_writes:
            raise RuntimeError("the database refused the write")
        self.staged = _config_row(json.loads(data["update"]["param_value"]))


class _StoredRouterSettings:
    """The LiteLLM_Config router_settings row behind a writer with PostgreSQL's transaction semantics: a
    transaction's write lands only when it commits, and the advisory lock waits for the transaction holding it"""

    def __init__(
        self,
        router_settings: RouterSettings | None,
        config_file: RouterSettings | None = None,
        *,
        lagging_replica: bool = False,
        hold_first_read: bool = False,
        fail_writes: bool = False,
    ) -> None:
        self.row: SimpleNamespace | None = None if router_settings is None else _config_row(router_settings)
        self.writes = 0
        self.fail_writes: Final = fail_writes
        self.lock: Final = asyncio.Lock()
        self.lock_contended: Final = asyncio.Event()
        self._holding_next_read = hold_first_read
        self.read_held: Final = asyncio.Event()
        self.held_read_released: Final = asyncio.Event()
        self.prisma_client: Final = MagicMock()
        self.prisma_client.get_generic_data = AsyncMock(
            side_effect=lambda value, **_: self.row if value == "router_settings" else None
        )
        self.prisma_client.writer_db.tx = self._transaction
        self.prisma_client.tx = self._transaction
        self.prisma_client.db.litellm_config.find_unique = AsyncMock(
            side_effect=lambda **_: None if lagging_replica else copy.deepcopy(self.row)
        )
        self.proxy_config: Final = _ConfigFileProxyConfig(config_file)
        self.prisma_client.writer_db.litellm_config.find_unique = AsyncMock(
            side_effect=lambda **_: copy.deepcopy(self.row)
        )
        self.fallbacks: Final = ConfigRepository(self.prisma_client, use_writer=True)

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[_ConfigTransaction]:
        transaction: Final = _ConfigTransaction(self)
        try:
            yield transaction
            if transaction.staged is not None:
                self.row = transaction.staged
                self.writes += 1
        finally:
            if transaction.holds_lock:
                self.lock.release()

    async def after_read(self) -> None:
        if not self._holding_next_read:
            return
        self._holding_next_read = False
        self.read_held.set()
        await self.held_read_released.wait()

    def written_by_another_instance(self, router_settings: RouterSettings) -> None:
        self.row = _config_row(router_settings)

    def stored_fallbacks(self) -> FallbackRules:
        assert self.row is not None
        return self.row.param_value["fallbacks"]

    async def cached_fallbacks(self) -> FallbackRules:
        row: Final = await get_config_param(self.prisma_client, "router_settings")
        return row.param_value["fallbacks"]


FallbackWrite: TypeAlias = Callable[[], Awaitable[object]]
TEAM_RULE: Final = {"team-primary": ["team-fallback"]}
GATEWAY_RULE: Final = {"gpt-5.4-mini": ["team-fallback"]}
MALFORMED_GATEWAY_RULE: Final = {"gpt-5.4-mini": "team-fallback"}
NON_STANDARD_RULES: Final = [
    "claude-3-haiku",
    {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "retry"}]},
]


class _RedisThatRefusesDel:
    def __init__(self) -> None:
        self.values: Mapping[str, object] = MappingProxyType({})

    async def async_get_cache(self, key: str, parent_otel_span: object = None) -> object:
        return self.values.get(key)

    async def async_set_cache(self, key: str, value: object, **_: object) -> None:
        self.values = MappingProxyType({**self.values, key: value})

    async def async_delete_cache(self, key: str) -> None:
        raise NoPermissionError("this user has no permissions to run the 'del' command")


class TestFallbackWritesSeeTheLatestStoredRules:
    """A write reads the rules the database holds now and leaves no stale copy in the config cache behind"""

    async def _create(self, request: FallbackCreateRequest, stored: _StoredRouterSettings) -> FallbackResponse:
        with (
            patch("litellm.proxy.proxy_server.llm_router", _team_router()),
            patch("litellm.proxy.proxy_server.proxy_config", stored.proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            return await create_fallback(request, MagicMock(), fallbacks=stored.fallbacks)

    async def _delete(self, model: str, stored: _StoredRouterSettings) -> FallbackDeleteResponse:
        with (
            patch("litellm.proxy.proxy_server.llm_router", _team_router()),
            patch("litellm.proxy.proxy_server.proxy_config", stored.proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            return await delete_fallback(model, "general", MagicMock(), fallbacks=stored.fallbacks)

    async def _get(
        self,
        model: str,
        stored: _StoredRouterSettings,
        fallback_type: Literal["general", "context_window", "content_policy"] = "general",
        router: Router | None = None,
    ) -> FallbackGetResponse:
        with (
            patch("litellm.proxy.proxy_server.llm_router", router or _team_router()),
            patch("litellm.proxy.proxy_server.proxy_config", stored.proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            return await get_fallback(model, fallback_type, MagicMock(), fallbacks=stored.fallbacks)

    async def _read_status(self, model: str, stored: _StoredRouterSettings, router: Router | None = None) -> int:
        try:
            await self._get(model, stored, router=router)
        except HTTPException as refused:
            return refused.status_code
        return 200

    async def test_a_replica_that_has_not_reloaded_reads_each_write_as_saved(self) -> None:
        stored: Final = _StoredRouterSettings(None, lagging_replica=True)
        stale_router: Final = _team_router()
        stale_router.apply(ChangeSet(assign={"fallbacks": [{"team-primary": ["gpt-5.4-mini"]}]}))

        await self._create(FallbackCreateRequest(model="team-primary", fallback_models=["team-fallback"]), stored)
        created: Final = await self._get("team-primary", stored, router=stale_router)
        await self._create(
            FallbackCreateRequest(model="team-primary", fallback_models=["gpt-5.4-mini", "team-fallback"]), stored
        )
        updated: Final = await self._get("team-primary", stored, router=stale_router)
        await self._delete("team-primary", stored)

        assert created.fallback_models == ["team-fallback"]
        assert updated.fallback_models == ["gpt-5.4-mini", "team-fallback"]
        assert await self._read_status("team-primary", stored, router=stale_router) == 404
        assert stale_router.settings().fallbacks == [{"team-primary": ["gpt-5.4-mini"]}]

    @pytest.mark.parametrize(
        ("fallback_type", "fallback_key"),
        [
            ("general", "fallbacks"),
            ("context_window", "context_window_fallbacks"),
            ("content_policy", "content_policy_fallbacks"),
        ],
    )
    async def test_a_read_names_every_saved_target_of_each_fallback_type(
        self, fallback_type: Literal["general", "context_window", "content_policy"], fallback_key: str
    ) -> None:
        stored: Final = _StoredRouterSettings(
            {fallback_key: [{"team-primary": [{"model": "gpt-5.4-mini", "api_key": "sk-test"}, "team-fallback"]}]}
        )

        response: Final = await self._get("team-primary", stored, fallback_type)

        assert response.fallback_models == ["gpt-5.4-mini", "team-fallback"]
        assert response.fallback_type == fallback_type

    @pytest.mark.parametrize(
        ("fallback_type", "fallback_key"),
        [("general", "fallbacks"), ("content_policy", "content_policy_fallbacks")],
    )
    async def test_a_read_skips_a_malformed_saved_rule_and_serves_the_valid_ones(
        self, fallback_type: Literal["general", "content_policy"], fallback_key: str
    ) -> None:
        stored: Final = _StoredRouterSettings(None)
        stored.row = SimpleNamespace(param_name="router_settings", param_value={fallback_key: [42, TEAM_RULE, None]})

        response: Final = await self._get("team-primary", stored, fallback_type)

        assert response.fallback_models == ["team-fallback"]

    async def test_a_read_resolves_the_saved_wildcard_rule(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [GATEWAY_RULE, {"*": ["gpt-5.4-mini"]}]})

        response: Final = await self._get("team-primary", stored)

        assert response.fallback_models == ["gpt-5.4-mini"]

    @pytest.mark.parametrize(
        "router_settings",
        [None, {"routing_strategy": "latency-based-routing"}, {"fallbacks": None}, {"fallbacks": "team-primary"}],
    )
    async def test_a_read_does_not_answer_with_rules_only_the_routers_memory_holds(
        self, router_settings: RouterSettings | None
    ) -> None:
        stored: Final = _StoredRouterSettings(router_settings)
        stale_router: Final = _team_router()
        stale_router.apply(ChangeSet(assign={"fallbacks": [TEAM_RULE]}))

        assert await self._read_status("team-primary", stored, router=stale_router) == 404

    async def test_a_content_policy_read_layers_the_saved_rules_over_the_litellm_settings_rules(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "content_policy_fallbacks", [TEAM_RULE, {"gpt-5.4-mini": ["team-primary"]}])
        stored: Final = _StoredRouterSettings({"content_policy_fallbacks": [GATEWAY_RULE]})
        emptied: Final = _StoredRouterSettings({"content_policy_fallbacks": []})

        saved_rule: Final = await self._get("gpt-5.4-mini", stored, "content_policy")
        file_rule: Final = await self._get("team-primary", stored, "content_policy")
        after_delete: Final = await self._get("gpt-5.4-mini", emptied, "content_policy")

        assert saved_rule.fallback_models == ["team-fallback"]
        assert file_rule.fallback_models == ["team-fallback"]
        assert after_delete.fallback_models == ["team-primary"]

    async def test_a_read_outside_database_mode_answers_from_the_router(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [TEAM_RULE]})
        router: Final = _team_router()
        router.apply(ChangeSet(assign={"fallbacks": [GATEWAY_RULE]}))

        with (
            patch("litellm.proxy.proxy_server.llm_router", router),
            patch("litellm.proxy.proxy_server.store_model_in_db", False),
        ):
            response: Final = await get_fallback("gpt-5.4-mini", "general", MagicMock(), fallbacks=stored.fallbacks)

        assert response.fallback_models == ["team-fallback"]

    async def test_a_read_of_a_router_settings_row_that_fails_answers_500(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [TEAM_RULE]})
        stored.prisma_client.writer_db.litellm_config.find_unique = AsyncMock(side_effect=RuntimeError("db down"))

        with pytest.raises(HTTPException) as failed:
            await self._get("team-primary", stored)

        assert failed.value.status_code == 500

    async def _while_a_write_holds_the_rules(
        self, stored: _StoredRouterSettings, holder: FallbackWrite, contender: FallbackWrite
    ) -> None:
        with (
            patch("litellm.proxy.proxy_server.llm_router", _team_router()),
            patch("litellm.proxy.proxy_server.proxy_config", stored.proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            holding: Final = asyncio.create_task(holder())
            await stored.read_held.wait()
            contending: Final = asyncio.create_task(contender())
            blocked: Final = asyncio.create_task(stored.lock_contended.wait())
            await asyncio.wait((contending, blocked), return_when=asyncio.FIRST_COMPLETED)
            blocked.cancel()
            stored.held_read_released.set()
            await asyncio.gather(holding, contending)

    async def test_concurrent_first_creates_keep_both_rules(self) -> None:
        stored: Final = _StoredRouterSettings(None, hold_first_read=True)

        await self._while_a_write_holds_the_rules(
            stored,
            lambda: create_fallback(
                FallbackCreateRequest(model="team-primary", fallback_models=["team-fallback"]),
                MagicMock(),
                fallbacks=stored.fallbacks,
            ),
            lambda: create_fallback(
                FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]),
                MagicMock(),
                fallbacks=stored.fallbacks,
            ),
        )

        assert stored.stored_fallbacks() == [TEAM_RULE, GATEWAY_RULE]

    async def test_a_delete_racing_a_create_keeps_the_created_rule(self) -> None:
        stored: Final = _StoredRouterSettings(
            {"fallbacks": [TEAM_RULE], "routing_strategy": "latency-based-routing"}, hold_first_read=True
        )

        await self._while_a_write_holds_the_rules(
            stored,
            lambda: create_fallback(
                FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]),
                MagicMock(),
                fallbacks=stored.fallbacks,
            ),
            lambda: delete_fallback("team-primary", "general", MagicMock(), fallbacks=stored.fallbacks),
        )

        assert stored.row is not None
        assert stored.row.param_value == {"fallbacks": [GATEWAY_RULE], "routing_strategy": "latency-based-routing"}

    async def test_a_config_section_write_racing_a_fallback_write_keeps_both_changes(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [TEAM_RULE]}, hold_first_read=True)

        await self._while_a_write_holds_the_rules(
            stored,
            lambda: create_fallback(
                FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]),
                MagicMock(),
                fallbacks=stored.fallbacks,
            ),
            lambda: stored.proxy_config._upsert_changed_config_section(
                section_name="router_settings",
                changed_keys={"routing_strategy": "latency-based-routing"},
                removed_keys=frozenset(),
                prisma_client=stored.prisma_client,
            ),
        )

        assert stored.row is not None
        assert stored.row.param_value == {
            "fallbacks": [TEAM_RULE, GATEWAY_RULE],
            "routing_strategy": "latency-based-routing",
        }

    async def test_a_failed_write_changes_neither_the_stored_rules_nor_the_router(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [TEAM_RULE]}, fail_writes=True)
        router: Final = _team_router()
        router.apply(ChangeSet(assign={"fallbacks": [TEAM_RULE]}))

        with (
            patch("litellm.proxy.proxy_server.llm_router", router),
            patch("litellm.proxy.proxy_server.proxy_config", stored.proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
            pytest.raises(HTTPException) as failed,
        ):
            await create_fallback(
                FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]),
                MagicMock(),
                fallbacks=stored.fallbacks,
            )

        assert failed.value.status_code == 500
        assert stored.stored_fallbacks() == [TEAM_RULE]
        assert router.settings().fallbacks == [TEAM_RULE]

    async def test_a_delete_of_a_missing_rule_answers_404_and_leaves_the_row_and_lock_untouched(self) -> None:
        settings: Final = {"fallbacks": [TEAM_RULE], "routing_strategy": "latency-based-routing"}
        stored: Final = _StoredRouterSettings(copy.deepcopy(settings))

        with pytest.raises(HTTPException) as missing:
            await self._delete("gpt-5.4-mini", stored)
        await self._create(FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]), stored)

        assert missing.value.status_code == 404
        assert stored.writes == 1
        assert stored.row is not None
        assert stored.row.param_value == {**settings, "fallbacks": [TEAM_RULE, GATEWAY_RULE]}

    async def test_a_create_reads_the_rules_from_the_writer_while_the_replica_lags(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [TEAM_RULE]}, lagging_replica=True)

        await self._create(FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]), stored)

        assert stored.stored_fallbacks() == [TEAM_RULE, GATEWAY_RULE]

    async def test_a_second_create_keeps_the_first_rule(self) -> None:
        stored: Final = _StoredRouterSettings(None)

        await self._create(FallbackCreateRequest(model="team-primary", fallback_models=["team-fallback"]), stored)
        await self._create(FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]), stored)

        assert stored.stored_fallbacks() == [TEAM_RULE, GATEWAY_RULE]
        assert await stored.cached_fallbacks() == [TEAM_RULE, GATEWAY_RULE]

    async def test_a_second_create_keeps_the_first_rule_when_redis_refuses_to_evict_the_cached_settings(self) -> None:
        stored: Final = _StoredRouterSettings(None)

        with patch.object(litellm_config_cache, "redis_cache", _RedisThatRefusesDel()):
            await self._create(FallbackCreateRequest(model="team-primary", fallback_models=["team-fallback"]), stored)
            await self._create(FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]), stored)
            await self._delete("team-primary", stored)
            await self._create(FallbackCreateRequest(model="team-primary", fallback_models=["team-fallback"]), stored)

        assert stored.stored_fallbacks() == [GATEWAY_RULE, TEAM_RULE]

    async def test_content_policy_writes_keep_the_litellm_settings_rules_for_other_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "content_policy_fallbacks", [TEAM_RULE])
        router: Final = _team_router()
        stored: Final = _StoredRouterSettings(None)
        with (
            patch("litellm.proxy.proxy_server.llm_router", router),
            patch("litellm.proxy.proxy_server.proxy_config", stored.proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            await create_fallback(
                FallbackCreateRequest(
                    model="gpt-5.4-mini", fallback_models=["team-fallback"], fallback_type="content_policy"
                ),
                MagicMock(),
                fallbacks=stored.fallbacks,
            )
            after_create: Final = await get_fallback(
                "team-primary", "content_policy", MagicMock(), fallbacks=stored.fallbacks
            )
            await delete_fallback("gpt-5.4-mini", "content_policy", MagicMock(), fallbacks=stored.fallbacks)
            after_delete: Final = await get_fallback(
                "team-primary", "content_policy", MagicMock(), fallbacks=stored.fallbacks
            )

        assert after_create.fallback_models == ["team-fallback"]
        assert after_delete.fallback_models == ["team-fallback"]

    async def test_a_content_policy_write_applies_beside_an_invalid_stored_rule(self) -> None:
        router: Final = _team_router()
        stored: Final = _StoredRouterSettings({"content_policy_fallbacks": [MALFORMED_GATEWAY_RULE]})
        with (
            patch("litellm.proxy.proxy_server.llm_router", router),
            patch("litellm.proxy.proxy_server.proxy_config", stored.proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            await create_fallback(
                FallbackCreateRequest(
                    model="team-primary", fallback_models=["team-fallback"], fallback_type="content_policy"
                ),
                MagicMock(),
                fallbacks=stored.fallbacks,
            )
            after_create: Final = await get_fallback(
                "team-primary", "content_policy", MagicMock(), fallbacks=stored.fallbacks
            )

        assert after_create.fallback_models == ["team-fallback"]

    @pytest.mark.parametrize(
        ("fallback_type", "router_setting"),
        [("general", "fallbacks"), ("content_policy", "content_policy_fallbacks")],
    )
    async def test_get_names_the_model_of_a_dict_target(
        self, fallback_type: Literal["general", "content_policy"], router_setting: str
    ) -> None:
        router: Final = _team_router()
        router.update_settings(
            **{router_setting: [{"team-primary": [{"model": "gpt-5.4-mini", "api_key": "sk-test"}, "team-fallback"]}]}
        )
        with patch("litellm.proxy.proxy_server.llm_router", router):
            response: Final = await get_fallback("team-primary", fallback_type, MagicMock(), fallbacks=None)

        assert response.fallback_models == ["gpt-5.4-mini", "team-fallback"]

    @pytest.mark.parametrize(
        ("fallback_type", "router_setting"),
        [("general", "fallbacks"), ("content_policy", "content_policy_fallbacks")],
    )
    async def test_get_names_the_rules_own_model_for_a_dict_target_that_names_none(
        self, fallback_type: Literal["general", "content_policy"], router_setting: str
    ) -> None:
        router: Final = _team_router()
        router.update_settings(**{router_setting: [{"team-primary": [{"temperature": 0}, "team-fallback"]}]})
        with patch("litellm.proxy.proxy_server.llm_router", router):
            response: Final = await get_fallback("team-primary", fallback_type, MagicMock(), fallbacks=None)

        assert response.fallback_models == ["team-primary", "team-fallback"]

    async def test_get_answers_an_error_for_a_string_valued_rule_instead_of_its_characters(self) -> None:
        router: Final = _team_router()
        router.update_settings(fallbacks=[{"team-primary": "team-fallback"}])
        with patch("litellm.proxy.proxy_server.llm_router", router), pytest.raises(HTTPException) as raised:
            await get_fallback("team-primary", "general", MagicMock(), fallbacks=None)

        assert raised.value.status_code == 500

    async def test_a_create_keeps_a_rule_another_instance_stored_since_this_one_last_read(self) -> None:
        stored: Final = _StoredRouterSettings({})
        await get_config_param(stored.prisma_client, "router_settings")
        stored.written_by_another_instance({"fallbacks": [TEAM_RULE]})

        await self._create(FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]), stored)

        assert stored.stored_fallbacks() == [TEAM_RULE, GATEWAY_RULE]

    async def test_a_delete_keeps_a_rule_another_instance_stored_since_this_one_last_read(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [TEAM_RULE]})
        await get_config_param(stored.prisma_client, "router_settings")
        stored.written_by_another_instance({"fallbacks": [TEAM_RULE, GATEWAY_RULE]})

        await self._delete("team-primary", stored)

        assert stored.stored_fallbacks() == [GATEWAY_RULE]
        assert await stored.cached_fallbacks() == [GATEWAY_RULE]

    async def test_writes_keep_the_non_standard_rules_the_router_accepts(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [*NON_STANDARD_RULES, TEAM_RULE]})

        await self._create(FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]), stored)
        assert stored.stored_fallbacks() == [*NON_STANDARD_RULES, TEAM_RULE, GATEWAY_RULE]

        await self._delete("team-primary", stored)
        assert stored.stored_fallbacks() == [*NON_STANDARD_RULES, GATEWAY_RULE]

    async def test_a_create_replaces_a_same_key_rule_whatever_its_targets_shape(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [MALFORMED_GATEWAY_RULE, TEAM_RULE]})

        await self._create(FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]), stored)

        assert stored.stored_fallbacks() == [GATEWAY_RULE, TEAM_RULE]

    async def test_a_delete_removes_a_same_key_rule_whatever_its_targets_shape(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [MALFORMED_GATEWAY_RULE, TEAM_RULE]})

        await self._delete("gpt-5.4-mini", stored)

        assert stored.stored_fallbacks() == [TEAM_RULE]

    async def test_a_create_treats_a_null_rule_list_as_empty(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": None})

        await self._create(FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]), stored)

        assert stored.stored_fallbacks() == [GATEWAY_RULE]

    @pytest.mark.parametrize("param_value", [None, ["team-primary"], 7, "not json"])
    async def test_a_create_replaces_a_stored_row_that_is_not_an_object(self, param_value: JsonValue) -> None:
        stored: Final = _StoredRouterSettings(None)
        stored.row = SimpleNamespace(param_name="router_settings", param_value=param_value)

        await self._create(FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]), stored)

        assert stored.row.param_value == {"fallbacks": [GATEWAY_RULE]}


class TestFallbackRoutesResolveTheirRepository:
    @pytest.fixture
    def stored(self) -> Iterator[_StoredRouterSettings]:
        stored: Final = _StoredRouterSettings(None, lagging_replica=True)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
        with (
            patch("litellm.proxy.proxy_server.llm_router", _team_router()),
            patch("litellm.proxy.proxy_server.proxy_config", stored.proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            yield stored
        app.dependency_overrides.pop(user_api_key_auth, None)
        app.dependency_overrides.pop(get_database, None)

    def test_a_create_stores_the_rule_through_the_request_database(self, stored: _StoredRouterSettings) -> None:
        app.dependency_overrides[get_database] = lambda: Database(prisma=stored.prisma_client, native=NATIVE)

        response: Final = TestClient(app).post(
            "/fallback", json={"model": "team-primary", "fallback_models": ["team-fallback"]}
        )

        assert response.status_code == 200, response.text
        assert stored.stored_fallbacks() == [TEAM_RULE]

    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("POST", "/fallback", {"model": "team-primary", "fallback_models": ["team-fallback"]}),
            ("DELETE", "/fallback/team-primary", None),
        ],
    )
    def test_writes_without_a_database_are_refused(
        self, stored: _StoredRouterSettings, method: str, path: str, body: Mapping[str, JsonValue] | None
    ) -> None:
        app.dependency_overrides[get_database] = lambda: None

        response: Final = TestClient(app).request(method, path, json=body)

        assert response.status_code == 400, response.text
        assert "Database storage not enabled" in response.text
        assert stored.row is None

    def test_a_read_serves_the_saved_rule_the_routers_memory_lacks(self, stored: _StoredRouterSettings) -> None:
        stored.written_by_another_instance({"fallbacks": [TEAM_RULE]})
        app.dependency_overrides[get_database] = lambda: Database(prisma=stored.prisma_client, native=NATIVE)

        response: Final = TestClient(app).get("/fallback/team-primary", params={"fallback_type": "general"})

        assert response.status_code == 200, response.text
        assert response.json() == {
            "model": "team-primary",
            "fallback_models": ["team-fallback"],
            "fallback_type": "general",
        }

    def test_a_read_without_a_database_answers_from_the_router(self, stored: _StoredRouterSettings) -> None:
        stored.written_by_another_instance({"fallbacks": [TEAM_RULE]})
        app.dependency_overrides[get_database] = lambda: None

        response: Final = TestClient(app).get("/fallback/team-primary", params={"fallback_type": "general"})

        assert response.status_code == 404, response.text

    def test_an_unknown_model_is_refused_before_the_missing_database(self, stored: _StoredRouterSettings) -> None:
        app.dependency_overrides[get_database] = lambda: None

        response: Final = TestClient(app).post(
            "/fallback", json={"model": "team-missing", "fallback_models": ["team-fallback"]}
        )

        assert response.status_code == 404, response.text


CONFIG_RULE: Final = {"gpt-5.4-mini": ["team-fallback"]}
CONFIG_WINDOW_RULE: Final = {"team-primary": ["gpt-5.4-mini"]}


class TestFallbackWritesHonourTheConfigFile:
    """A fallbacks list set in config.yaml owns the key: /fallback writes that would change it are refused and
    the router keeps serving the file's rules"""

    @pytest.fixture(autouse=True)
    async def clean_config_cache(self) -> AsyncIterator[None]:
        await evict_config_param("router_settings")
        yield
        await evict_config_param("router_settings")

    @pytest.fixture
    def router(self) -> Router:
        return _team_router()

    @pytest.fixture
    def stored(self) -> _StoredRouterSettings:
        return _StoredRouterSettings({"fallbacks": [CONFIG_RULE]}, config_file={"fallbacks": [CONFIG_RULE]})

    async def _create(
        self, request: FallbackCreateRequest, router: Router, stored: _StoredRouterSettings
    ) -> FallbackResponse:
        with (
            patch("litellm.proxy.proxy_server.llm_router", router),
            patch("litellm.proxy.proxy_server.proxy_config", stored.proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            return await create_fallback(request, MagicMock(), fallbacks=stored.fallbacks)

    async def _delete(self, model: str, router: Router, stored: _StoredRouterSettings) -> FallbackDeleteResponse:
        with (
            patch("litellm.proxy.proxy_server.llm_router", router),
            patch("litellm.proxy.proxy_server.proxy_config", stored.proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            return await delete_fallback(model, "general", MagicMock(), fallbacks=stored.fallbacks)

    async def test_a_create_that_adds_a_rule_to_a_config_owned_list_is_refused(
        self, router: Router, stored: _StoredRouterSettings
    ) -> None:
        router.apply(ChangeSet(assign={"fallbacks": [CONFIG_RULE]}))
        request: Final = FallbackCreateRequest(model="team-primary", fallback_models=["team-fallback"])

        with pytest.raises(HTTPException) as refused:
            await self._create(request, router, stored)

        assert refused.value.status_code == 400
        assert "key 'fallbacks' is set in the config file" in str(refused.value.detail)
        assert stored.stored_fallbacks() == [CONFIG_RULE]
        assert stored.writes == 0
        assert router.settings().fallbacks == [CONFIG_RULE]

    async def test_a_create_after_the_config_file_takes_over_the_key_is_refused(self, router: Router) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [TEAM_RULE]})
        stored.proxy_config.config_file = {"fallbacks": [CONFIG_RULE]}
        request: Final = FallbackCreateRequest(model="team-primary", fallback_models=["gpt-5.4-mini"])

        with pytest.raises(HTTPException) as refused:
            await self._create(request, router, stored)

        assert refused.value.status_code == 400
        assert "key 'fallbacks' is set in the config file" in str(refused.value.detail)
        assert stored.stored_fallbacks() == [TEAM_RULE]

    async def test_a_delete_of_a_config_owned_rule_is_refused(
        self, router: Router, stored: _StoredRouterSettings
    ) -> None:
        router.apply(ChangeSet(assign={"fallbacks": [CONFIG_RULE]}))

        with pytest.raises(HTTPException) as refused:
            await self._delete("gpt-5.4-mini", router, stored)

        assert refused.value.status_code == 400
        assert "key 'fallbacks' is set in the config file" in str(refused.value.detail)
        assert stored.stored_fallbacks() == [CONFIG_RULE]
        assert stored.writes == 0
        assert router.settings().fallbacks == [CONFIG_RULE]

    async def test_a_delete_of_a_rule_only_the_config_file_holds_is_refused(self, router: Router) -> None:
        stored: Final = _StoredRouterSettings(None, config_file={"fallbacks": [CONFIG_RULE]})
        router.fallbacks = [CONFIG_RULE]

        with pytest.raises(HTTPException) as refused:
            await self._delete("gpt-5.4-mini", router, stored)

        assert refused.value.status_code == 400
        assert "key 'fallbacks' is set in the config file" in str(refused.value.detail)
        assert stored.row is None
        assert router.fallbacks == [CONFIG_RULE]

    async def test_writes_to_a_config_file_env_reference_that_resolves_to_nothing_are_refused(
        self, router: Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LITELLM_TEST_UNSET_FALLBACKS", raising=False)
        stored: Final = _StoredRouterSettings(
            None, config_file={"fallbacks": "os.environ/LITELLM_TEST_UNSET_FALLBACKS"}
        )

        with pytest.raises(HTTPException) as refused:
            await self._create(
                FallbackCreateRequest(model="team-primary", fallback_models=["team-fallback"]), router, stored
            )
        with pytest.raises(HTTPException) as not_found:
            await self._delete("team-primary", router, stored)

        assert refused.value.status_code == 400
        assert "key 'fallbacks' is set in the config file" in str(refused.value.detail)
        assert not_found.value.status_code == 404
        assert stored.row is None

    async def test_a_create_that_restates_one_of_several_config_rules_writes_nothing(self, router: Router) -> None:
        stored: Final = _StoredRouterSettings(None, config_file={"fallbacks": [CONFIG_RULE, CONFIG_WINDOW_RULE]})
        request: Final = FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"])

        response: Final = await self._create(request, router, stored)

        assert response.fallback_models == ["team-fallback"]
        assert stored.row is None
        assert router.fallbacks == [CONFIG_RULE, CONFIG_WINDOW_RULE]

    async def test_a_content_policy_restate_keeps_the_config_list_without_the_litellm_settings_rules(
        self, router: Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "content_policy_fallbacks", [TEAM_RULE])
        stored: Final = _StoredRouterSettings(None, config_file={"content_policy_fallbacks": [CONFIG_RULE]})
        request: Final = FallbackCreateRequest(
            model="gpt-5.4-mini", fallback_models=["team-fallback"], fallback_type="content_policy"
        )

        response: Final = await self._create(request, router, stored)

        assert response.fallback_models == ["team-fallback"]
        assert stored.row is None
        assert router.content_policy_fallbacks == [CONFIG_RULE]

    async def test_a_create_that_restates_a_config_rule_next_to_an_env_reference_writes_nothing(
        self, router: Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LITELLM_TEST_FALLBACK_MODEL", "team-fallback")
        env_rule: Final = {"team-primary": "os.environ/LITELLM_TEST_FALLBACK_MODEL"}
        stored: Final = _StoredRouterSettings(None, config_file={"fallbacks": [CONFIG_RULE, env_rule]})
        request: Final = FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"])

        response: Final = await self._create(request, router, stored)

        assert response.fallback_models == ["team-fallback"]
        assert stored.row is None

    async def test_a_create_that_restates_the_config_rule_is_accepted(
        self, router: Router, stored: _StoredRouterSettings
    ) -> None:
        request: Final = FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"])

        response: Final = await self._create(request, router, stored)

        assert response.fallback_models == ["team-fallback"]
        assert stored.stored_fallbacks() == [CONFIG_RULE]

    async def test_a_write_stores_only_the_database_rules_never_the_config_files_router_settings(
        self, router: Router
    ) -> None:
        stored: Final = _StoredRouterSettings(
            {"fallbacks": [TEAM_RULE]}, config_file={"routing_strategy": "latency-based-routing"}
        )

        await self._create(
            FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]), router, stored
        )
        assert stored.row is not None
        assert stored.row.param_value == {"fallbacks": [TEAM_RULE, GATEWAY_RULE]}

        await self._delete("team-primary", router, stored)
        assert stored.row.param_value == {"fallbacks": [GATEWAY_RULE]}

    async def test_a_read_of_a_config_owned_list_answers_from_the_file_not_the_shadowed_row(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [TEAM_RULE]}, config_file={"fallbacks": [CONFIG_RULE]})

        with (
            patch("litellm.proxy.proxy_server.llm_router", _team_router()),
            patch("litellm.proxy.proxy_server.proxy_config", stored.proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            from_file: Final = await get_fallback("gpt-5.4-mini", "general", MagicMock(), fallbacks=stored.fallbacks)
            with pytest.raises(HTTPException) as shadowed:
                await get_fallback("team-primary", "general", MagicMock(), fallbacks=stored.fallbacks)

        assert from_file.fallback_models == ["team-fallback"]
        assert shadowed.value.status_code == 404

    async def test_a_sibling_list_the_config_file_does_not_set_stays_writable(
        self, router: Router, stored: _StoredRouterSettings
    ) -> None:
        request: Final = FallbackCreateRequest(
            model="team-primary", fallback_models=["gpt-5.4-mini"], fallback_type="context_window"
        )

        response: Final = await self._create(request, router, stored)

        assert response.fallback_type == "context_window"
        assert stored.row is not None
        assert stored.row.param_value["context_window_fallbacks"] == [CONFIG_WINDOW_RULE]
        assert router.settings().context_window_fallbacks == [CONFIG_WINDOW_RULE]


@pytest.mark.asyncio
class TestCreateFallback:
    """Test the create_fallback endpoint"""

    @pytest.fixture
    def mock_router(self):
        """Create a mock router"""
        router = MagicMock()
        names: Final = {
            "group_set": {"gpt-3.5-turbo", "gpt-4", "claude-3-haiku"},
            "team_public": frozenset(),
        }
        router.registry.return_value.names.side_effect = lambda model_name=None, *, kind="groups", **_: names[kind]
        router.settings.return_value = RouterSettingsSnapshot(
            fallbacks=[], context_window_fallbacks=[], content_policy_fallbacks=[]
        )
        return router

    @pytest.fixture
    def stored(self) -> _StoredRouterSettings:
        return _StoredRouterSettings(None)

    @pytest.fixture
    def mock_prisma_client(self, stored: _StoredRouterSettings) -> MagicMock:
        return stored.prisma_client

    @pytest.fixture
    def mock_proxy_config(self):
        """Create a mock proxy config"""
        return _ConfigFileProxyConfig()

    @pytest.fixture
    def mock_user_api_key_dict(self):
        """Create a mock user API key dict"""
        return MagicMock()

    async def test_create_fallback_success(
        self, mock_router, stored, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test successful fallback creation"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4", "claude-3-haiku"],
            fallback_type="general",
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
        ):
            response = await create_fallback(
                request, mock_user_api_key_dict, fallbacks=ConfigRepository(mock_prisma_client)
            )

            assert response.model == "gpt-3.5-turbo"
            assert response.fallback_models == ["gpt-4", "claude-3-haiku"]
            assert response.fallback_type == "general"
            assert "created" in response.message.lower() or "updated" in response.message.lower()

            assert stored.stored_fallbacks() == [{"gpt-3.5-turbo": ["gpt-4", "claude-3-haiku"]}]

    async def test_create_fallback_router_not_initialized(
        self, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test error when router is not initialized"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4"],
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                None,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_fallback(request, mock_user_api_key_dict, fallbacks=ConfigRepository(mock_prisma_client))

        assert exc_info.value.status_code == 500
        assert "Router not initialized" in str(exc_info.value.detail)

    async def test_create_fallback_model_not_found(
        self, mock_router, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test error when model is not found in router"""
        request = FallbackCreateRequest(
            model="invalid-model",
            fallback_models=["gpt-4"],
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_fallback(request, mock_user_api_key_dict, fallbacks=ConfigRepository(mock_prisma_client))

        assert exc_info.value.status_code == 404
        assert "not found in router" in str(exc_info.value.detail)

    async def test_create_fallback_invalid_fallback_model(
        self, mock_router, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test error when fallback model is not found in router"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["invalid-fallback-model"],
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_fallback(request, mock_user_api_key_dict, fallbacks=ConfigRepository(mock_prisma_client))

        assert exc_info.value.status_code == 400
        assert "Invalid fallback models" in str(exc_info.value.detail)

    async def test_create_fallback_model_is_own_fallback(
        self, mock_router, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test error when model is its own fallback"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-3.5-turbo", "gpt-4"],
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_fallback(request, mock_user_api_key_dict, fallbacks=ConfigRepository(mock_prisma_client))

        assert exc_info.value.status_code == 400
        assert "cannot be its own fallback" in str(exc_info.value.detail)

    async def test_create_fallback_db_not_enabled(self, mock_router, mock_user_api_key_dict):
        """Test error when database storage is not enabled"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4"],
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                False,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_fallback(request, mock_user_api_key_dict, fallbacks=None)

        assert exc_info.value.status_code == 400
        assert "Database storage not enabled" in str(exc_info.value.detail)

    async def test_create_fallback_context_window_type(
        self, mock_router, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test creating context_window fallback"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4"],
            fallback_type="context_window",
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
        ):
            response = await create_fallback(
                request, mock_user_api_key_dict, fallbacks=ConfigRepository(mock_prisma_client)
            )

            assert response.fallback_type == "context_window"
            # Verify the correct setting was updated
            mock_router.apply.assert_called_once_with(
                ChangeSet(assign={"context_window_fallbacks": [{"gpt-3.5-turbo": ["gpt-4"]}]})
            )


@pytest.mark.asyncio
class TestGetFallback:
    """Test the get_fallback endpoint"""

    @pytest.fixture
    def mock_router_with_fallbacks(self):
        """Create a mock router with fallbacks configured"""
        router = MagicMock()
        router.settings.return_value = RouterSettingsSnapshot(
            fallbacks=[{"gpt-3.5-turbo": ["gpt-4", "claude-3-haiku"]}],
            context_window_fallbacks=[],
            content_policy_fallbacks=[],
        )
        return router

    @pytest.fixture
    def mock_user_api_key_dict(self):
        """Create a mock user API key dict"""
        return MagicMock()

    async def test_get_fallback_success(self, mock_router_with_fallbacks, mock_user_api_key_dict):
        """Test successful fallback retrieval"""
        with patch(
            "litellm.proxy.proxy_server.llm_router",
            mock_router_with_fallbacks,
        ):
            response = await get_fallback("gpt-3.5-turbo", "general", mock_user_api_key_dict, fallbacks=None)

            assert response.model == "gpt-3.5-turbo"
            assert response.fallback_models == ["gpt-4", "claude-3-haiku"]
            assert response.fallback_type == "general"

    async def test_get_fallback_not_found(self, mock_router_with_fallbacks, mock_user_api_key_dict):
        """Test error when fallback is not found"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router_with_fallbacks,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_fallback("gpt-4", "general", mock_user_api_key_dict, fallbacks=None)

        assert exc_info.value.status_code == 404
        assert "No general fallbacks configured" in str(exc_info.value.detail)

    async def test_get_fallback_router_not_initialized(self, mock_user_api_key_dict):
        """Test error when router is not initialized"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                None,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_fallback("gpt-3.5-turbo", "general", mock_user_api_key_dict, fallbacks=None)

        assert exc_info.value.status_code == 500
        assert "Router not initialized" in str(exc_info.value.detail)


@pytest.mark.asyncio
class TestDeleteFallback:
    """Test the delete_fallback endpoint"""

    @pytest.fixture
    def mock_router_with_fallbacks(self):
        """Create a mock router with fallbacks configured"""
        router = MagicMock()
        router.settings.return_value = RouterSettingsSnapshot(
            fallbacks=[{"gpt-3.5-turbo": ["gpt-4", "claude-3-haiku"]}],
            context_window_fallbacks=[],
            content_policy_fallbacks=[],
        )
        return router

    @pytest.fixture
    def stored(self) -> _StoredRouterSettings:
        return _StoredRouterSettings({"fallbacks": [{"gpt-3.5-turbo": ["gpt-4", "claude-3-haiku"]}]})

    @pytest.fixture
    def mock_prisma_client(self, stored: _StoredRouterSettings) -> MagicMock:
        return stored.prisma_client

    @pytest.fixture
    def mock_proxy_config(self):
        """Create a mock proxy config"""
        return _ConfigFileProxyConfig()

    @pytest.fixture
    def mock_user_api_key_dict(self):
        """Create a mock user API key dict"""
        return MagicMock()

    async def test_delete_fallback_success(
        self,
        mock_router_with_fallbacks,
        stored,
        mock_prisma_client,
        mock_proxy_config,
        mock_user_api_key_dict,
    ):
        """Test successful fallback deletion"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router_with_fallbacks,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
        ):
            response = await delete_fallback(
                "gpt-3.5-turbo",
                "general",
                mock_user_api_key_dict,
                fallbacks=ConfigRepository(mock_prisma_client),
            )

            assert response.model == "gpt-3.5-turbo"
            assert response.fallback_type == "general"
            assert "deleted" in response.message.lower()

            assert stored.stored_fallbacks() == []

    async def test_delete_fallback_not_found(
        self,
        mock_router_with_fallbacks,
        mock_prisma_client,
        mock_proxy_config,
        mock_user_api_key_dict,
    ):
        """Test error when fallback to delete is not found"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router_with_fallbacks,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await delete_fallback(
                "gpt-4", "general", mock_user_api_key_dict, fallbacks=ConfigRepository(mock_prisma_client)
            )

        assert exc_info.value.status_code == 404
        assert "No general fallbacks configured" in str(exc_info.value.detail)

    async def test_delete_fallback_router_not_initialized(self, mock_user_api_key_dict):
        """Test error when router is not initialized"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                None,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await delete_fallback("gpt-3.5-turbo", "general", mock_user_api_key_dict, fallbacks=None)

        assert exc_info.value.status_code == 500
        assert "Router not initialized" in str(exc_info.value.detail)

    async def test_delete_fallback_db_not_enabled(self, mock_router_with_fallbacks, mock_user_api_key_dict):
        """Test error when database storage is not enabled"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router_with_fallbacks,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                False,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await delete_fallback("gpt-3.5-turbo", "general", mock_user_api_key_dict, fallbacks=None)

        assert exc_info.value.status_code == 400
        assert "Database storage not enabled" in str(exc_info.value.detail)
