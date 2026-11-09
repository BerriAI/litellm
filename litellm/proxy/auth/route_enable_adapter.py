import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final, assert_never

from fastapi import HTTPException
from pydantic import TypeAdapter

from litellm._auth.failure import NativeAuthFailure, NativeAuthRuntimeError
from litellm._auth.route_enable import RouteEnableDecision, RouteEnableFacts, RouteFeature, RouteSetting, route_enable
from litellm.proxy._types import CommonProxyErrors
from litellm.proxy.auth.route_policy_adapter import route_context

_KEYS: Final = {"Admin": "DISABLE_ADMIN_ENDPOINTS", "Llm": "DISABLE_LLM_API_ENDPOINTS"}


_EXEMPTIONS: Final = TypeAdapter(tuple[str, ...])


def read_enterprise() -> object:
    from litellm_enterprise.proxy.auth.route_checks import EnterpriseRouteChecks

    return EnterpriseRouteChecks


def read_setting(feature: RouteFeature) -> RouteSetting:
    from litellm.proxy.proxy_server import premium_user
    from litellm.secret_managers.main import (
        get_secret_bool as _get_secret_bool,  # noqa: F401  # pyright: ignore[reportUnusedImport]  # preserve the Phase 1 import before environment and license reads
    )

    return RouteSetting(_KEYS[feature] in os.environ, bool(premium_user))


def read_secret(feature: RouteFeature) -> bool:
    from litellm.secret_managers.main import get_secret_bool

    return get_secret_bool(_KEYS[feature]) is True


def read_exemptions(enterprise: object) -> tuple[str, ...]:
    return _EXEMPTIONS.validate_python(
        getattr(enterprise, "LLM_API_EXEMPT_ROUTES")  # noqa: B009  # optional class attributes are validated at this boundary
    )


@dataclass(frozen=True, slots=True)
class RouteEnableReaders:
    enterprise: Callable[[], object] = read_enterprise
    setting: Callable[[RouteFeature], RouteSetting] = read_setting
    secret: Callable[[RouteFeature], bool] = read_secret
    exemptions: Callable[[object], tuple[str, ...]] = read_exemptions


_READERS: Final = RouteEnableReaders()


def _answer(
    facts: RouteEnableFacts, decision: RouteEnableDecision, readers: RouteEnableReaders, enterprise: object
) -> tuple[RouteEnableFacts, object]:
    match decision.kind:
        case "ReadImport":
            return replace(facts, imported=True), readers.enterprise()
        case "ReadExemptions":
            return replace(facts, exemptions=readers.exemptions(enterprise)), enterprise
        case "ReadSettings" | "ReadSecret":
            if decision.feature is None:
                raise NativeAuthRuntimeError("route setting read without feature")
            previous: Final = facts.admin if decision.feature == "Admin" else facts.llm
            if decision.kind == "ReadSecret" and previous is None:
                raise NativeAuthRuntimeError("route secret read without settings")
            setting: Final = (
                readers.setting(decision.feature)
                if decision.kind == "ReadSettings" or previous is None
                else replace(previous, secret=readers.secret(decision.feature))
            )
            return (
                replace(facts, admin=setting) if decision.feature == "Admin" else replace(facts, llm=setting)
            ), enterprise
        case "Allow" | "Deny" | "PropagateHttp":
            raise NativeAuthRuntimeError("completed route enable decision requested IO")
        case _:
            assert_never(decision.kind)


def _read(
    facts: RouteEnableFacts,
    readers: RouteEnableReaders,
    enterprise: object = None,
    http_error: HTTPException | None = None,
) -> tuple[RouteEnableFacts, HTTPException | None]:
    decision: Final = route_enable(facts)
    if decision.kind in ("Allow", "Deny", "PropagateHttp"):
        return facts, http_error
    try:
        updated, imported = _answer(facts, decision, readers, enterprise)
    except NativeAuthFailure:
        raise
    except HTTPException as error:
        return _read(replace(facts, failed="Http"), readers, enterprise, error)
    except Exception:  # noqa: BLE001  # optional enterprise import and configuration IO failures were ignored in Phase 1
        return _read(replace(facts, failed="Other"), readers, enterprise)
    return _read(updated, readers, imported)


def check_route_enabled(route: str, *, readers: RouteEnableReaders = _READERS) -> None:
    facts, error = _read(RouteEnableFacts(route_context(route)), readers)
    decision: Final = route_enable(facts)
    if decision.kind == "Allow":
        return
    if decision.kind == "PropagateHttp" and error is not None:
        raise error
    if decision.kind != "Deny" or decision.feature is None or decision.reason is None:
        raise NativeAuthRuntimeError("incomplete route enable decision")
    if decision.reason == "License":
        label: Final = "ADMIN" if decision.feature == "Admin" else "LLM API"
        raise HTTPException(
            500,
            f"\U0001f6a8\U0001f6a8\U0001f6a8 DISABLING {label} ENDPOINTS is an Enterprise feature\n\U0001f6a8 {CommonProxyErrors.not_premium_user.value}",
        )
    disabled_label: Final = "Management" if decision.feature == "Admin" else "LLM API"
    raise HTTPException(403, f"{disabled_label} routes are disabled for this instance.")
