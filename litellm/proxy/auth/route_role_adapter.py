from collections.abc import Iterable
from dataclasses import replace
from itertools import chain
from typing import Final, NoReturn, assert_never

from fastapi import Request
from pydantic import TypeAdapter
from starlette.routing import compile_path

from litellm._auth import route_roles as core
from litellm._auth.failure import NativeAuthFailure
from litellm.proxy._types import (
    LiteLLM_JWTAuth,
    LiteLLM_UserTable,
    LiteLLMRoutes,
    LitellmUserRoles,
    UserAPIKeyAuth,
)

_STRINGS: Final = TypeAdapter(tuple[str, ...])
_END: Final = object()


def _role(user: LiteLLM_UserTable | None) -> core.RoleFacts:
    return core.RoleFacts(user is not None, user.user_role if user is not None else None)


def effective_user_role(user_role: str | None) -> LitellmUserRoles:
    return LitellmUserRoles(core.role(core.RoleFacts(True, user_role)).effective)


def get_user_role(user_obj: LiteLLM_UserTable | None) -> LitellmUserRoles | None:
    role: Final = core.role(_role(user_obj)).user_role
    return LitellmUserRoles(role) if role is not None else None


def is_user_proxy_admin(user_obj: LiteLLM_UserTable | None) -> bool:
    return core.role(_role(user_obj)).admin


def is_api_route_allowed(
    route: str,
    request: Request,
    request_data: dict[str, object],  # mutable-ok: pass the original request dictionary to the shared route wrapper
    valid_token: UserAPIKeyAuth | None,
    user_obj: LiteLLM_UserTable | None = None,
) -> bool:
    from litellm.proxy.auth.route_checks import RouteChecks

    role: Final = get_user_role(user_obj)
    initial: Final = core.ApiFacts(valid_token is not None, False, core.RoleFacts(False, None))
    decision: Final = core.api(initial)
    if decision == "InvalidToken":
        raise Exception("Invalid proxy server token passed. valid_token=None.")  # noqa: TRY002  # preserve the public legacy exception class
    result: Final = core.api(replace(initial, admin_read=True, user=_role(user_obj)))
    if result == "NonAdmin":
        assert valid_token is not None
        RouteChecks.non_proxy_admin_allowed_routes_check(  # pyright: ignore[reportUnknownMemberType]  # the shared wrapper still declares an unparameterized request dictionary
            user_obj=user_obj,
            _user_role=role,
            route=route,
            request=request,
            request_data=request_data,
            valid_token=valid_token,
        )
    return True


def _entry(raw: object) -> core.Entry:
    try:
        hash(raw)
    except NativeAuthFailure:
        raise
    except TypeError as error:
        return core.Entry(None, type(raw).__name__, hash_failure=str(error))
    text: Final = raw if isinstance(raw, str) else None
    group: Final = LiteLLMRoutes.__members__.get(text) if text is not None else None
    return core.Entry(
        text,
        type(raw).__name__,
        templates=_STRINGS.validate_python(group.value) if group is not None else None,  # pyright: ignore[reportUnknownMemberType]  # enum groups are legacy route collections
    )


def _raise_failure(failure: core.Failure | None) -> NoReturn:
    assert failure is not None
    match failure.kind:
        case "TypeError":
            raise TypeError(failure.message)
        case "AttributeError":
            raise AttributeError(failure.message)
        case _:
            assert_never(failure.kind)


def allowed_routes_list_check(user_route: str, allowed_routes: Iterable[object]) -> bool:
    entries: Final = iter(allowed_routes)
    facts = core.ListedFacts(user_route)  # rebind-ok: native-selected data-read driver state
    while True:
        decision = core.listed(facts)
        match decision.action:
            case "ReadEntry":
                raw = next(entries, _END)
                facts = core.ListedFacts(user_route, None if raw is _END else _entry(raw), raw is _END)
            case "ReadTemplate":
                assert facts.entry is not None and facts.entry.templates is not None
                compiled, _, _ = compile_path(facts.entry.templates[decision.template_index])
                facts = replace(facts, template_index=decision.template_index, compiled=compiled.pattern)
            case "Allow":
                return True
            case "Deny":
                return False
            case "Failure":
                _raise_failure(decision.failure)
            case _:
                raise AssertionError(f"Unexpected listed-route directive: {decision.action}")


def allowed_routes_check(
    user_role: LitellmUserRoles,
    user_route: str,
    litellm_proxy_roles: LiteLLM_JWTAuth,
) -> bool:
    facts: Final = core.JwtFacts(user_role)
    decision: Final = core.jwt(facts)
    match decision:
        case "Deny":
            return False
        case "ReadAdmin":
            return allowed_routes_list_check(user_route, litellm_proxy_roles.admin_allowed_routes)
        case "ReadTeam":
            configured: Final = litellm_proxy_roles.team_allowed_routes
            selected: Final = core.jwt(replace(facts, config_read=True, config_missing=configured is None))  # pyright: ignore[reportUnnecessaryComparison]  # constructed legacy JWT configs retain the None default branch
            return allowed_routes_list_check(
                user_route, ("openai_routes", "info_routes") if selected == "DefaultTeam" else configured
            )
        case _:
            raise AssertionError(f"Unexpected JWT route directive: {decision}")


def allowed_route_check_inside_route(user_api_key_dict: UserAPIKeyAuth, requested_user_id: str | None) -> bool:
    return core.inside(core.InsideFacts(user_api_key_dict.user_role, user_api_key_dict.user_id, requested_user_id))


def _expanded(raw: object) -> tuple[object, ...]:
    facts: Final = _entry(raw)
    decision: Final = core.expand(facts)
    match decision.action:
        case "Expand":
            assert facts.templates is not None
            return facts.templates
        case "Keep":
            return (raw,)
        case "Failure":
            _raise_failure(decision.failure)
        case _:
            raise AssertionError(f"Unexpected route expansion directive: {decision.action}")


def get_actual_routes(
    allowed_routes: Iterable[object],
) -> list[object]:  # mutable-ok: preserve the public list projection
    return list(chain.from_iterable(_expanded(raw) for raw in allowed_routes))
