from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy import proxy_server
from litellm.proxy.auth import route_scope_adapter as native


@dataclass(frozen=True, slots=True)
class Result:
    allowed: bool
    status: int | None = None
    detail: object = None
    error: str | None = None


def outcome(call: Callable[[], object]) -> Result:
    try:
        return Result(call() is not False)
    except HTTPException as error:
        return Result(False, error.status_code, error.detail)
    except TypeError as error:
        return Result(False, error=str(error))


def reference(route: str, settings: Mapping[str, object], premium: object, admin: bool) -> object:
    """RouteChecks wrapper behavior frozen from main 2f430873aead."""
    if admin:
        if "admin_only_routes" not in settings or premium is not True:
            return None
        if route in settings["admin_only_routes"]:  # pyright: ignore[reportOperatorIssue]  # freeze malformed configuration TypeError as well as valid membership
            raise HTTPException(
                status_code=403, detail=f"user not allowed to access this route. Route={route} is an admin only route"
            )
        return None
    if "allowed_routes" not in settings:
        return True
    return route in settings["allowed_routes"]  # pyright: ignore[reportOperatorIssue]  # frozen Python membership preserves string, list, mapping and invalid-scalar semantics


@pytest.mark.parametrize("admin", (False, True))
@pytest.mark.parametrize("premium", (False, True, 1, None))
@pytest.mark.parametrize("route", ("/v1/responses", "/missing", "responses", ""))
@pytest.mark.parametrize("present", (False, True))
@pytest.mark.parametrize(
    "value",
    (None, True, 3, (), ("/v1/responses",), "/v1/responses", {"/v1/responses": False}, (None, "/v1/responses", 3)),
)
def test_route_scope_matches_frozen_main(
    monkeypatch: pytest.MonkeyPatch, admin: bool, premium: object, route: str, present: bool, value: object
) -> None:
    settings: Final[Mapping[str, object]] = (
        {("admin_only_routes" if admin else "allowed_routes"): value} if present else {}
    )
    monkeypatch.setattr(proxy_server, "general_settings", dict(settings))
    monkeypatch.setattr(proxy_server, "premium_user", premium)
    policy: Final = native.custom_admin_only_route_check if admin else native.is_allowed_on_proxy
    assert outcome(lambda: policy(route)) == outcome(lambda: reference(route, settings, premium, admin))
