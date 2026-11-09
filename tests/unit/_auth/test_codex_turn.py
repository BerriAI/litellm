from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.codex_models import CodexCandidate, CodexModelFacts
from litellm._auth.codex_turn import (
    CodexAccessFacts,
    CodexCredentials,
    CodexKeyFacts,
    CodexTurnFacts,
    JwtRead,
    codex_access,
    codex_jwt,
    codex_jwt_token,
    codex_key,
    codex_turn,
)
from litellm._auth.failure import NativeAuthRuntimeError
from litellm._auth.route_enable import RouteEnableFacts
from litellm._auth.route_scope import RouteScopeFacts, RouteValues
from litellm._auth.routes import VirtualRouteFacts
from litellm.proxy.auth import handle_jwt
from litellm.proxy.auth.codex_turn_adapter import read_jwt_result
from litellm.proxy.auth.route_policy_adapter import route_context
from tests.unit._auth.support.route_reference import RouteChecks as FrozenRoutes

_CONTEXT: Final = route_context("/v1/responses")
_KEY: Final = CodexKeyFacts(
    RouteScopeFacts("Proxy", _CONTEXT.route, False, False),
    RouteEnableFacts(_CONTEXT, failed="Other"),
    VirtualRouteFacts(_CONTEXT, None),
    True,
    "internal_user",
    RouteScopeFacts("Admin", _CONTEXT.route, False, False),
)
_ROUTING: Final = CodexModelFacts(
    "Routes",
    "model",
    True,
    bundled=(),
    priced=False,
    candidates=(CodexCandidate(True, "openai", "openai/backend", "model"),),
    alias_loaded=True,
)
_KEYS: Final = (
    _KEY,
    replace(_KEY, proxy=RouteScopeFacts("Proxy", _CONTEXT.route, True, True, values=RouteValues())),
    replace(_KEY, enabled=RouteEnableFacts(_CONTEXT, failed="Http")),
    replace(_KEY, virtual_route=VirtualRouteFacts(_CONTEXT, ("/other",))),
    replace(
        _KEY, admin=RouteScopeFacts("Admin", _CONTEXT.route, True, True, values=RouteValues(names=(_CONTEXT.route,)))
    ),
    replace(
        _KEY,
        role="proxy_admin",
        admin=RouteScopeFacts("Admin", _CONTEXT.route, True, True, values=RouteValues(names=(_CONTEXT.route,))),
    ),
)


def _scope(facts: RouteScopeFacts | None) -> bool:
    assert facts is not None
    if not facts.present or facts.kind == "Admin" and not facts.premium:
        return True
    assert facts.values is not None
    values: Final = facts.values.text if facts.values.text is not None else facts.values.names
    return (facts.route in values) == (facts.kind == "Proxy")


def reference(facts: CodexTurnFacts) -> bool:
    """Completed raw facts evaluated in main 2f430873aead order, without native policy."""
    credentials: Final = facts.credentials
    if facts.native and (
        credentials is None or not credentials.bearer_present or not credentials.gateway_present or credentials.equal
    ):
        return False
    routing: Final = facts.routing
    if (
        facts.model is None
        or routing is None
        or not routing.router_present
        or routing.priced
        or facts.model in (routing.bundled or ())
    ):
        return False
    candidates: Final = routing.candidates
    names: Final = (facts.model, routing.alias or facts.model)
    if not candidates or any(
        candidate.provider == "chatgpt"
        or (candidate.model or "").partition("/")[0] == "chatgpt"
        or (candidate.name not in names and candidate.public_name not in names)
        for candidate in candidates
    ):
        return False
    access: Final = facts.access
    assert access is not None
    key: Final = access.key
    if not _scope(key.proxy):
        return False
    assert key.enabled is not None and key.virtual_route is not None
    if key.enabled.failed == "Http":
        return False
    if key.virtual_route.allowed_routes and not FrozenRoutes.check_route_access(
        _CONTEXT.route, key.virtual_route.allowed_routes
    ):
        return False
    if key.role != "proxy_admin" and not _scope(key.admin):
        return False
    if access.jwt_token_present and access.jwt == "Exception":
        return False
    return facts.slugs is not None and facts.model not in facts.slugs


def native(facts: CodexTurnFacts) -> bool:
    decision: Final = codex_turn(facts)
    assert decision.kind == "Complete"
    return decision.allowed


@pytest.mark.parametrize("policy", (reference, native))
@pytest.mark.parametrize("key", _KEYS)
@pytest.mark.parametrize("jwt", (None, "Returned", "Exception"))
@pytest.mark.parametrize(
    "credentials",
    (
        CodexCredentials(True, True, False),
        CodexCredentials(False, True, False),
        CodexCredentials(True, False, False),
        CodexCredentials(True, True, True),
    ),
)
@pytest.mark.parametrize("native_turn", (False, True))
@pytest.mark.parametrize("routed", (False, True))
@pytest.mark.parametrize("slugs", (None, (), ("other",), ("model",)))
def test_turn_destination_matches_main(
    policy: Callable[[CodexTurnFacts], bool],
    key: CodexKeyFacts,
    jwt: JwtRead | None,
    credentials: CodexCredentials,
    native_turn: bool,
    routed: bool,
    slugs: tuple[str, ...] | None,
) -> None:
    facts: Final = CodexTurnFacts(
        "model",
        native_turn,
        credentials,
        CodexAccessFacts(key, jwt is not None, jwt),
        _ROUTING if routed else replace(_ROUTING, bundled=("model",)),
        True,
        slugs,
    )
    expected: Final = (
        (not native_turn or credentials == CodexCredentials(True, True, False))
        and routed
        and key in (_KEYS[0], _KEYS[5])
        and jwt != "Exception"
        and slugs is not None
        and "model" not in slugs
    )
    assert policy(facts) == expected


@pytest.mark.parametrize(
    ("facts", "stage"),
    (
        (CodexTurnFacts("model", native=True), "ReadCredentials"),
        (CodexTurnFacts("model", native=True, credentials=CodexCredentials(True, True, False)), "ReadAccess"),
        (CodexTurnFacts("model", access=CodexAccessFacts(_KEYS[1], False)), "ReadRouting"),
        (CodexTurnFacts("model", access=CodexAccessFacts(_KEY, True), routing=_ROUTING), "ReadJwt"),
        (CodexTurnFacts("model", access=CodexAccessFacts(_KEY, False), routing=_ROUTING), "ReadCatalog"),
    ),
)
def test_turn_retains_main_read_order(facts: CodexTurnFacts, stage: str) -> None:
    assert codex_turn(facts).kind == stage


@pytest.mark.parametrize(
    ("facts", "stage"),
    (
        (CodexKeyFacts(), "ReadProxy"),
        (CodexKeyFacts(proxy=_KEY.proxy), "ReadEnabled"),
        (CodexKeyFacts(proxy=_KEY.proxy, enabled=_KEY.enabled), "ReadVirtual"),
        (replace(_KEY, role_loaded=False), "ReadRole"),
        (replace(_KEY, admin=None), "ReadAdmin"),
    ),
)
def test_key_route_reads_are_selected_in_order(facts: CodexKeyFacts, stage: str) -> None:
    assert codex_key(facts).kind == stage


@pytest.mark.parametrize("claims", (False, True))
@pytest.mark.parametrize("token", (False, True))
def test_standard_jwt_selection(claims: bool, token: bool) -> None:
    assert codex_jwt_token(claims_present=claims, token_present=token).allowed == (claims and not token)


@pytest.mark.parametrize("result", ("Returned", "Exception"))
def test_jwt_validator_return_is_not_reinterpreted(result: JwtRead) -> None:
    assert codex_jwt(result).allowed == (result == "Returned")
    assert codex_access(CodexAccessFacts(_KEYS[1], True, result)).allowed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (None, ValueError("validator rejected"), NativeAuthRuntimeError("native failed")))
async def test_jwt_wrapper_never_turns_a_native_fault_into_subscription_fallback(
    monkeypatch: pytest.MonkeyPatch, failure: Exception | None
) -> None:
    class Validator:
        @staticmethod
        async def auth_builder(
            *, api_key: str, request_data: Mapping[str, object], general_settings: Mapping[str, object], route: str
        ) -> bool:
            if failure is not None:
                raise failure
            return False

    monkeypatch.setattr(handle_jwt, "JWTAuthManager", Validator)
    if isinstance(failure, NativeAuthRuntimeError):
        with pytest.raises(NativeAuthRuntimeError) as raised:
            await read_jwt_result("test-credential", "model", {})
        assert raised.value is failure
    else:
        assert await read_jwt_result("test-credential", "model", {}) == ("Returned" if failure is None else "Exception")
