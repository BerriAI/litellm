import time
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import (
    LiteLLM_UserTable,
    LitellmUserRoles,
    NewUserRequest,
    parse_stored_user_role,
)
from litellm.proxy.auth.custom_rbac import (
    build_custom_rbac_engine,
    get_active_custom_rbac_engine,
    get_config_custom_rbac_roles,
    invalidate_custom_rbac_engine_cache,
    is_reserved_role_name,
    validate_assigned_user_role,
    validate_role_permissions,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.types.custom_rbac import CustomRBACRole


def _user(role: str) -> LiteLLM_UserTable:
    return LiteLLM_UserTable(user_id="u1", user_role=role, max_budget=None, spend=0.0)


class TestCustomRBACEngine:
    def test_exact_route_permission(self):
        engine = build_custom_rbac_engine(
            roles=(CustomRBACRole(role_name="viewer", allowed_routes=("/key/info",)),)
        )
        assert engine.is_governed_role("viewer") is True
        assert engine.is_route_allowed(role_name="viewer", route="/key/info") is True
        assert engine.is_route_allowed(role_name="viewer", route="/key/generate") is False

    def test_route_group_permission(self):
        engine = build_custom_rbac_engine(
            roles=(CustomRBACRole(role_name="llm-only", allowed_routes=("llm_api_routes",)),)
        )
        assert engine.is_route_allowed(role_name="llm-only", route="/chat/completions") is True
        assert engine.is_route_allowed(role_name="llm-only", route="/team/new") is False

    def test_wildcard_and_all_routes(self):
        engine = build_custom_rbac_engine(
            roles=(
                CustomRBACRole(role_name="team-mgr", allowed_routes=("/team/*",)),
                CustomRBACRole(role_name="superuser", allowed_routes=("*",)),
            )
        )
        assert engine.is_route_allowed(role_name="team-mgr", route="/team/new") is True
        assert engine.is_route_allowed(role_name="team-mgr", route="/key/generate") is False
        assert engine.is_route_allowed(role_name="superuser", route="/key/generate") is True

    def test_inheritance_is_transitive(self):
        engine = build_custom_rbac_engine(
            roles=(
                CustomRBACRole(role_name="base", allowed_routes=("/key/info",)),
                CustomRBACRole(role_name="mid", allowed_routes=("/team/info",), inherits=("base",)),
                CustomRBACRole(role_name="top", allowed_routes=("/user/info",), inherits=("mid",)),
            )
        )
        assert engine.is_route_allowed(role_name="top", route="/key/info") is True
        assert engine.is_route_allowed(role_name="top", route="/team/info") is True
        assert engine.is_route_allowed(role_name="top", route="/user/info") is True
        assert engine.is_route_allowed(role_name="base", route="/user/info") is False

    def test_cyclic_inheritance_terminates(self):
        engine = build_custom_rbac_engine(
            roles=(
                CustomRBACRole(role_name="a", allowed_routes=("/key/info",), inherits=("b",)),
                CustomRBACRole(role_name="b", allowed_routes=("/team/info",), inherits=("a",)),
            )
        )
        assert engine.is_route_allowed(role_name="a", route="/team/info") is True
        assert engine.is_route_allowed(role_name="b", route="/key/info") is True
        assert engine.is_route_allowed(role_name="a", route="/user/info") is False

    def test_unknown_role_is_not_governed(self):
        engine = build_custom_rbac_engine(roles=(CustomRBACRole(role_name="viewer"),))
        assert engine.is_governed_role("internal_user") is False
        assert engine.is_governed_role(None) is False


class TestRouteEnforcement:
    def test_governed_role_denies_ungranted_route(self):
        engine = build_custom_rbac_engine(
            roles=(CustomRBACRole(role_name="viewer", allowed_routes=("/key/info",)),)
        )
        assert (
            RouteChecks.custom_rbac_route_allowed(
                user_obj=_user("viewer"), route="/key/info", custom_rbac_engine=engine
            )
            is True
        )
        with pytest.raises(HTTPException) as exc:
            RouteChecks.custom_rbac_route_allowed(
                user_obj=_user("viewer"), route="/key/generate", custom_rbac_engine=engine
            )
        assert exc.value.status_code == 403

    def test_builtin_role_falls_through_to_builtin_checks(self):
        engine = build_custom_rbac_engine(
            roles=(CustomRBACRole(role_name="viewer", allowed_routes=("/key/info",)),)
        )
        assert (
            RouteChecks.custom_rbac_route_allowed(
                user_obj=_user(LitellmUserRoles.INTERNAL_USER.value),
                route="/key/generate",
                custom_rbac_engine=engine,
            )
            is False
        )

    def test_no_engine_falls_through(self):
        assert (
            RouteChecks.custom_rbac_route_allowed(
                user_obj=_user("viewer"), route="/key/generate", custom_rbac_engine=None
            )
            is False
        )


class TestRoleValidation:
    def test_reserved_role_names(self):
        assert is_reserved_role_name("internal_user") is True
        assert is_reserved_role_name("data-scientist") is False

    def test_invalid_permissions_are_reported(self):
        assert validate_role_permissions(allowed_routes=("*", "llm_api_routes", "/key/info")) == ()
        assert validate_role_permissions(allowed_routes=("key/info", "nonsense")) == ("key/info", "nonsense")

    def test_custom_role_string_survives_request_validation(self):
        assert NewUserRequest(user_role="data-scientist").user_role == "data-scientist"
        assert NewUserRequest(user_role="internal_user").user_role is LitellmUserRoles.INTERNAL_USER

    def test_non_assignable_builtin_role_is_rejected(self):
        with pytest.raises(ValueError):
            NewUserRequest(user_role="team")

    def test_stored_role_parsing_keeps_custom_names(self):
        assert parse_stored_user_role("internal_user") is LitellmUserRoles.INTERNAL_USER
        assert parse_stored_user_role("data-scientist") == "data-scientist"


class _FakeRecord:
    def __init__(self, role_name: str, allowed_routes: tuple[str, ...]) -> None:
        self._role_name = role_name
        self._allowed_routes = allowed_routes

    def dict(self) -> dict[str, object]:
        return {
            "role_name": self._role_name,
            "description": None,
            "allowed_routes": list(self._allowed_routes),
            "inherits": [],
        }


class _FakeTable:
    def __init__(self, records: tuple[_FakeRecord, ...], fail: bool = False) -> None:
        self._records = records
        self._fail = fail
        self.reads = 0

    async def find_many(self, order):
        assert type(order) is dict, "prisma only serializes plain dicts"
        self.reads += 1
        if self._fail:
            raise RuntimeError("db down")
        return self._records


_CONFIG_KEY = "custom_rbac_roles"
_TABLE_PATH = "litellm.proxy.auth.custom_rbac._custom_role_table"


class TestEngineLoading:
    def setup_method(self):
        invalidate_custom_rbac_engine_cache()

    def teardown_method(self):
        invalidate_custom_rbac_engine_cache()

    def test_config_roles_are_parsed(self):
        settings = {_CONFIG_KEY: [{"role_name": "cfg", "allowed_routes": ["/key/info"]}]}
        with patch("litellm.proxy.proxy_server.general_settings", settings):
            assert get_config_custom_rbac_roles() == (
                CustomRBACRole(role_name="cfg", allowed_routes=("/key/info",)),
            )

    def test_malformed_config_roles_are_ignored(self):
        with patch("litellm.proxy.proxy_server.general_settings", {_CONFIG_KEY: [{"allowed_routes": 5}]}):
            assert get_config_custom_rbac_roles() == ()

    @pytest.mark.asyncio
    async def test_config_and_db_roles_are_both_active_and_cached(self):
        table = _FakeTable(records=(_FakeRecord("db-role", ("/team/info",)),))
        settings = {_CONFIG_KEY: [{"role_name": "cfg-role", "allowed_routes": ["/key/info"]}]}
        with (
            patch("litellm.proxy.proxy_server.general_settings", settings),
            patch(_TABLE_PATH, return_value=table),
        ):
            engine = await get_active_custom_rbac_engine()
            cached = await get_active_custom_rbac_engine()

        assert engine is not None
        assert engine.is_route_allowed(role_name="cfg-role", route="/key/info") is True
        assert engine.is_route_allowed(role_name="db-role", route="/team/info") is True
        assert engine.is_route_allowed(role_name="db-role", route="/key/generate") is False
        assert cached is engine
        assert table.reads == 1

    @pytest.mark.asyncio
    async def test_db_failure_reuses_last_known_policy(self):
        healthy = _FakeTable(records=(_FakeRecord("db-role", ("/team/info",)),))
        with patch("litellm.proxy.proxy_server.general_settings", {}), patch(_TABLE_PATH, return_value=healthy):
            engine = await get_active_custom_rbac_engine()

        broken = _FakeTable(records=(), fail=True)
        with (
            patch("litellm.proxy.proxy_server.general_settings", {}),
            patch(_TABLE_PATH, return_value=broken),
            patch("time.monotonic", return_value=time.monotonic() + 3600),
        ):
            after_failure = await get_active_custom_rbac_engine()

        assert after_failure is engine

    @pytest.mark.asyncio
    async def test_db_failure_without_cache_still_serves_config_roles(self):
        settings = {_CONFIG_KEY: [{"role_name": "cfg-role", "allowed_routes": ["/key/info"]}]}
        with (
            patch("litellm.proxy.proxy_server.general_settings", settings),
            patch(_TABLE_PATH, return_value=_FakeTable(records=(), fail=True)),
        ):
            engine = await get_active_custom_rbac_engine()

        assert engine is not None
        assert engine.is_route_allowed(role_name="cfg-role", route="/key/info") is True
        assert engine.is_route_allowed(role_name="cfg-role", route="/key/generate") is False

    @pytest.mark.asyncio
    async def test_no_roles_means_no_engine(self):
        with (
            patch("litellm.proxy.proxy_server.general_settings", {}),
            patch(_TABLE_PATH, return_value=_FakeTable(records=())),
        ):
            assert await get_active_custom_rbac_engine() is None

    @pytest.mark.asyncio
    async def test_prisma_client_without_db_does_not_break_auth(self):
        class _ClientWithoutDb:
            pass

        with (
            patch("litellm.proxy.proxy_server.general_settings", {}),
            patch("litellm.proxy.proxy_server.prisma_client", _ClientWithoutDb()),
        ):
            assert await get_active_custom_rbac_engine() is None

    @pytest.mark.asyncio
    async def test_assigning_undefined_custom_role_is_rejected(self):
        table = _FakeTable(records=(_FakeRecord("db-role", ("/team/info",)),))
        with (
            patch("litellm.proxy.proxy_server.general_settings", {}),
            patch(_TABLE_PATH, return_value=table),
        ):
            await validate_assigned_user_role("db-role")
            await validate_assigned_user_role(LitellmUserRoles.INTERNAL_USER)
            with pytest.raises(HTTPException) as exc:
                await validate_assigned_user_role("ghost-role")
        assert exc.value.status_code == 400
