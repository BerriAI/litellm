"""Behavior pins for ``litellm/router_utils/pattern_match_deployments.py``."""

from __future__ import annotations

from litellm.router_utils import pattern_match_deployments
from litellm.router_utils.pattern_match_deployments import PatternMatchRouter


def _wildcard_deployment(model_name: str) -> dict:
    return {"model_name": model_name, "litellm_params": {"model": model_name}}


def _matched_models(matches: list[dict] | None) -> list[str]:
    return [deployment["litellm_params"]["model"] for deployment in matches or []]


def test_get_pattern_never_resolves_declared_authenticating_providers(monkeypatch):
    """Regression: resolving a github_copilot/chatgpt name through ``get_llm_provider`` runs the
    provider's OAuth device flow; the auth layer walks every wildcard router on every request, so
    a single metadata lookup for an unserved name would block the proxy's event loop."""
    resolution_attempts: list[str] = []

    def _oauth_tripwire(model, *args, **kwargs):
        resolution_attempts.append(model)
        raise AssertionError("get_llm_provider would run the OAuth device flow")

    monkeypatch.setattr(pattern_match_deployments, "get_llm_provider", _oauth_tripwire)

    unmatched_router = PatternMatchRouter()
    unmatched_router.add_pattern("anthropic/*", _wildcard_deployment("anthropic/*"))
    assert unmatched_router.get_pattern("github_copilot/gpt-4o") is None

    matched_router = PatternMatchRouter()
    matched_router.add_pattern("github_copilot/*", _wildcard_deployment("github_copilot/*"))
    assert _matched_models(matched_router.get_pattern("github_copilot/gpt-4o")) == ["github_copilot/gpt-4o"]
    assert _matched_models(matched_router.get_pattern("gpt-4o", custom_llm_provider="github_copilot")) == [
        "github_copilot/gpt-4o"
    ]

    assert resolution_attempts == []


def test_get_pattern_still_resolves_unqualified_names(monkeypatch):
    monkeypatch.setattr(
        pattern_match_deployments,
        "get_llm_provider",
        lambda model, **kwargs: (model, "openai", None, None),
    )
    router = PatternMatchRouter()
    router.add_pattern("openai/*", _wildcard_deployment("openai/*"))
    assert _matched_models(router.get_pattern("gpt-4o")) == ["openai/gpt-4o"]
