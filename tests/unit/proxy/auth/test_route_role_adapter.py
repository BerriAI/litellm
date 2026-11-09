from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final

import pytest
from fastapi import HTTPException, Request

from litellm._auth import route_roles as core
from litellm._auth.failure import NativeAuthFailure
from litellm.proxy._types import LiteLLM_JWTAuth, LiteLLM_UserTable, LiteLLMRoutes, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth import route_role_adapter as native
from litellm.rust_bridge import _native

from . import route_role_reference as reference

_ROLES: Final = (None, "", "unknown", "PROXY_ADMIN", *(role.value for role in LitellmUserRoles))


@dataclass(frozen=True, slots=True)
class Failure:
    kind: type[Exception]
    message: str
    status: int | None
    detail: object


def observed(call: Callable[[], object]) -> object:
    try:
        return call()
    except Exception as error:
        return Failure(
            type(error), str(error),
            error.status_code if isinstance(error, HTTPException) else None,
            error.detail if isinstance(error, HTTPException) else None,
        )


@pytest.mark.parametrize("role", _ROLES)
@pytest.mark.parametrize("present", (False, True))
def test_role_and_admin_projection_matches_reference(role: str | None, present: bool) -> None:
    user: Final = LiteLLM_UserTable(user_id="u", user_role=role) if present else None
    assert native.effective_user_role(role) is reference.effective_user_role(role)
    assert native.get_user_role(user) is reference.get_user_role(user)
    assert native.is_user_proxy_admin(user) is reference.is_user_proxy_admin(user)


@pytest.mark.parametrize("role", _ROLES)
@pytest.mark.parametrize("user_id", (None, "", "u"))
@pytest.mark.parametrize("requested", (None, "", "u", "other"))
def test_inside_route_admin_viewer_or_non_none_self(role: str | None, user_id: str | None, requested: str | None) -> None:
    auth: Final = UserAPIKeyAuth.model_construct(user_role=role, user_id=user_id)
    assert native.allowed_route_check_inside_route(auth, requested) is reference.allowed_route_check_inside_route(auth, requested)


@pytest.mark.parametrize("role", _ROLES)
@pytest.mark.parametrize("token_present", (False, True))
@pytest.mark.parametrize("route", ("/chat/completions", "/user/info", "/team/new"))
def test_api_route_role_precedes_token_and_nonadmin_check(role: str | None, token_present: bool, route: str) -> None:
    user: Final = LiteLLM_UserTable(user_id="u", user_role=role)
    token: Final = UserAPIKeyAuth(user_id="u") if token_present else None
    request: Final = Request({"type": "http", "method": "POST", "path": route, "headers": [], "query_string": b""})
    assert observed(lambda: native.is_api_route_allowed(route, request, {}, token, user)) == observed(
        lambda: reference.is_api_route_allowed(route, request, {}, token, user)
    )


@pytest.mark.parametrize("role", tuple(LitellmUserRoles))
@pytest.mark.parametrize("configured", (None, (), ("info_routes",), ("/x*",)))
@pytest.mark.parametrize("route", ("/chat/completions", "/user/info", "/x/y", "/not-configured"))
def test_jwt_admin_team_none_default_and_empty_deny(
    role: LitellmUserRoles, configured: tuple[str, ...] | None, route: str
) -> None:
    config: Final = LiteLLM_JWTAuth.model_construct(
        admin_allowed_routes=None if configured is None else list(configured),
        team_allowed_routes=None if configured is None else list(configured),
    )
    assert observed(lambda: native.allowed_routes_check(role, route, config)) == observed(
        lambda: reference.allowed_routes_check(role, route, config)
    )


@pytest.mark.parametrize("entries", (
    (), ("/x",), ("/x*",), ("/x/{id}",), ("info_routes",),
    ("/x", None), (None, "/x"), (1,), (False,), (list[object](),), (dict[str, object](),),
    (set[object](),), ((list[object](),),), (("x",),), ("missing_group", "info_routes"), ("/x", list[object]()),
))
@pytest.mark.parametrize("route", ("/x", "/x/a", "/x/", "/user/info", ""))
def test_listed_route_invalid_entries_and_short_circuit_match_reference(entries: tuple[object, ...], route: str) -> None:
    assert observed(lambda: native.allowed_routes_list_check(route, entries)) == observed(
        lambda: reference.allowed_routes_list_check(route, entries)
    )


@pytest.mark.parametrize("group", tuple(LiteLLMRoutes.__members__))
@pytest.mark.parametrize("route", ("/chat/completions", "/user/info", "/key/a/reset_spend", "/credentials/a/b", "/x\n", ""))
def test_every_route_group_uses_starlette_template_semantics(group: str, route: str) -> None:
    assert observed(lambda: native.allowed_routes_list_check(route, (group,))) == observed(
        lambda: reference.allowed_routes_list_check(route, (group,))
    )


@pytest.mark.parametrize("entries", (("info_routes", "info_routes"), ("/x", None, 1, ("x",)), ([],), ({},)))
def test_expansion_preserves_order_duplicates_nonstring_values_and_errors(entries: tuple[object, ...]) -> None:
    assert observed(lambda: native.get_actual_routes(entries)) == observed(lambda: reference.get_actual_routes(entries))


@pytest.mark.parametrize("compiled", ("^/thing/(?P<id>[^/]+)$", "^/thing/(?P<id>.*)$", "^/v1\\.0/$"))
@pytest.mark.parametrize("route", ("/thing/", "/thing/a", "/thing/a/b", "/thing/a\n", "/thing/a\nb", "/v1.0/", "/v1x0/"))
def test_compiled_group_regex_keeps_empty_path_literal_escaping_and_final_newline(compiled: str, route: str) -> None:
    import re

    decision: Final = core.listed(core.ListedFacts(route, core.Entry("group", "str", templates=("unused",)), compiled=compiled))
    assert (decision.action == "Allow") is (re.match(compiled, route) is not None)


def test_native_fault_is_terminal_before_reading_first_allowed_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_facts: object) -> object:
        raise ValueError("decode failed")

    def entries() -> Iterable[object]:
        raise AssertionError("entry read after native failure")
        yield "unused"

    monkeypatch.setattr(_native, "request_auth_route_role_listed", broken)
    with pytest.raises(NativeAuthFailure, match="decode failed"):
        native.allowed_routes_list_check("/x", entries())
