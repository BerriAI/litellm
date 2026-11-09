from collections.abc import Callable
from dataclasses import replace
from typing import Final

import pytest
from fastapi import Request

from litellm._auth.codex_normalization import CodexNormalizationFacts, codex_normalization
from litellm._auth.codex_turn import CodexAccessFacts, CodexTurnFacts
from litellm.constants import CLIENT_REQUESTED_MODEL_SCOPE_KEY
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.codex_normalization_adapter import resolve_codex_native_chatgpt_model
from litellm.proxy.auth.route_policy_adapter import route_context
from tests.unit._auth.support.route_reference import RouteChecks as FrozenRoutes
from tests.unit._auth.test_codex_models import Registry, Router, deployment
from tests.unit._auth.test_codex_turn import (
    _KEY,  # pyright: ignore[reportPrivateUsage]  # immutable raw key facts shared by the composed planner
    _ROUTING,  # pyright: ignore[reportPrivateUsage]  # immutable raw routing facts shared by the composed planner
)
from tests.unit._auth.test_codex_turn import reference as reference_turn

_NATIVE_ROUTE: Final = "/backend-api/codex/responses"
_TURN: Final = CodexTurnFacts(
    "model", access=CodexAccessFacts(_KEY, False), routing=_ROUTING, catalog_loaded=True, slugs=()
)
_ALIAS: Final = CodexNormalizationFacts(
    "Alias",
    "/chat/completions",
    True,
    True,
    context=route_context("/chat/completions"),
    resolved=False,
    marked=True,
    model_loaded=True,
    model="model",
    explicit=False,
    provider=False,
    routing=_ROUTING,
    body_present=True,
    settings_loaded=True,
    target="target",
    provider_after=False,
    turn=_TURN,
)


def reference(facts: CodexNormalizationFacts) -> str | None:
    """Completed alias reads in _resolve_router_settings_model_group_alias on main 2f430873aead."""
    if not facts.request_present or not facts.router_present or not FrozenRoutes.is_llm_api_route(facts.route):
        return None
    if facts.resolved:
        return None
    if facts.explicit or facts.provider and facts.route != _NATIVE_ROUTE:
        return None
    if facts.provider and facts.routing is not None and facts.model in (facts.routing.bundled or ()):
        return None
    if facts.model is None or not facts.body_present:
        return None
    if not facts.target or facts.target == facts.model:
        return None
    if facts.provider_after and (facts.turn is None or not reference_turn(facts.turn)):
        return None
    return facts.target


def native(facts: CodexNormalizationFacts) -> str | None:
    decision: Final = codex_normalization(facts)
    assert decision.kind in ("Complete", "Rewrite")
    return decision.target


@pytest.mark.parametrize("policy", (reference, native))
@pytest.mark.parametrize("route", ("/key/info", "/chat/completions", _NATIVE_ROUTE))
@pytest.mark.parametrize("provider", (False, True))
@pytest.mark.parametrize("target", (None, "", "model", "target"))
@pytest.mark.parametrize("body", (False, True))
def test_alias_mutation_only_after_main_admission(
    policy: Callable[[CodexNormalizationFacts], str | None], route: str, provider: bool, target: str | None, body: bool
) -> None:
    facts: Final = replace(
        _ALIAS,
        route=route,
        context=route_context(route),
        provider=provider,
        provider_after=provider,
        target=target,
        body_present=body,
    )
    expected: Final = (
        "target"
        if route != "/key/info" and (not provider or route == _NATIVE_ROUTE) and target == "target" and body
        else None
    )
    assert policy(facts) == expected


@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (replace(_ALIAS, request_present=False, context=None), "Complete"),
        (replace(_ALIAS, router_present=False, context=None), "Complete"),
        (replace(_ALIAS, context=None), "ReadContext"),
        (replace(_ALIAS, resolved=None), "ReadResolved"),
        (replace(_ALIAS, resolved=True), "Complete"),
        (replace(_ALIAS, marked=False, explicit=None), "MarkResolved"),
        (replace(_ALIAS, model_loaded=False, explicit=None), "ReadModel"),
        (replace(_ALIAS, explicit=True, body_present=None), "Complete"),
        (replace(_ALIAS, explicit=None), "ReadExplicit"),
        (replace(_ALIAS, provider=None), "ReadProvider"),
        (replace(_ALIAS, body_present=None), "ReadBody"),
        (replace(_ALIAS, settings_loaded=False), "ReadSettings"),
        (replace(_ALIAS, provider_after=None), "ReadProviderAfter"),
        (
            replace(_ALIAS, route=_NATIVE_ROUTE, context=route_context(_NATIVE_ROUTE), provider=True, routing=None),
            "ReadRouting",
        ),
        (
            replace(
                _ALIAS,
                route=_NATIVE_ROUTE,
                context=route_context(_NATIVE_ROUTE),
                provider=True,
                provider_after=True,
                turn=None,
            ),
            "ReadTurn",
        ),
        (
            replace(
                _ALIAS,
                route=_NATIVE_ROUTE,
                context=route_context(_NATIVE_ROUTE),
                provider=True,
                provider_after=True,
                turn=replace(_TURN, slugs=("model",)),
            ),
            "Complete",
        ),
        (CodexNormalizationFacts("Prefix", _NATIVE_ROUTE, True, model="model", model_loaded=True), "ReadRouter"),
        (CodexNormalizationFacts("Prefix", _NATIVE_ROUTE, True, True, model="model", model_loaded=True), "ReadPrefix"),
        (CodexNormalizationFacts("Prefix", "/v1/responses", True, model="model", model_loaded=True), "Complete"),
    ),
)
def test_normalization_read_and_mutation_order(facts: CodexNormalizationFacts, expected: str) -> None:
    assert codex_normalization(facts).kind == expected


@pytest.mark.parametrize("requested_scope", (None, "client-alias"))
def test_prefix_adapter_preserves_client_model_and_rewrites_request(
    monkeypatch: pytest.MonkeyPatch, requested_scope: str | None
) -> None:
    router: Final = Router(Registry((deployment("chatgpt/model", "chatgpt/backend"),)))
    monkeypatch.setattr(proxy_server, "llm_router", router)
    token: Final = UserAPIKeyAuth(models=["chatgpt/model"], team_models=["chatgpt/model"])
    request: Final = Request({"type": "http", "path": _NATIVE_ROUTE, "headers": [], "method": "POST"})
    if requested_scope is not None:
        request.scope[CLIENT_REQUESTED_MODEL_SCOPE_KEY] = requested_scope
    data: Final[dict[str, object]] = {
        "model": "model",
        "input": "hello",
    }  # mutable-ok: production mutates request-local body
    resolve_codex_native_chatgpt_model(data, token, request, _NATIVE_ROUTE)
    assert data == {"model": "chatgpt/model", "input": "hello"}
    assert request.scope[CLIENT_REQUESTED_MODEL_SCOPE_KEY] == (requested_scope or "model")
    before: Final = tuple(router.reader.calls)
    resolve_codex_native_chatgpt_model(data, token, request, _NATIVE_ROUTE)
    assert tuple(router.reader.calls) == before
