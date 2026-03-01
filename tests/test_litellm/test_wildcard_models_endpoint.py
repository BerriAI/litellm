"""
Tests for issue #13752: Wildcard entries (e.g. openai/*) should NOT appear
in /models endpoint unless return_wildcard_routes=True.

Root cause: _get_wildcard_models() only removed wildcard entries from
unique_models when llm_router was None or the router had no deployment.
When the router DID have a wildcard deployment (the common case), the
wildcard pattern stayed in the list alongside the expanded models.
"""

from litellm.proxy.auth.model_checks import (
    _check_wildcard_routing,
    _get_wildcard_models,
    get_complete_model_list,
)
from litellm.router import Router


class TestCheckWildcardRouting:
    def test_wildcard_star_only(self):
        assert _check_wildcard_routing("*") is True

    def test_provider_wildcard(self):
        assert _check_wildcard_routing("openai/*") is True
        assert _check_wildcard_routing("anthropic/*") is True
        assert _check_wildcard_routing("gemini/*") is True

    def test_non_wildcard(self):
        assert _check_wildcard_routing("openai/gpt-4") is False
        assert _check_wildcard_routing("anthropic/claude-3-sonnet") is False


class TestGetWildcardModelsRemovesPatterns:
    """The core bug: wildcard patterns must be removed from unique_models
    when expanded, regardless of whether the router has a deployment."""

    def test_wildcard_removed_when_router_has_deployment(self):
        """Bug repro: openai/* stays in unique_models when router has deployment."""
        model_list = [
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}}
        ]
        router = Router(model_list=model_list)

        unique_models = ["openai/*", "some-custom-model"]
        expanded = _get_wildcard_models(
            unique_models=unique_models,
            return_wildcard_routes=False,
            llm_router=router,
        )
        # openai/* must be removed from unique_models
        assert "openai/*" not in unique_models
        # custom model must remain
        assert "some-custom-model" in unique_models
        # expanded list should contain real model names
        assert len(expanded) > 0
        assert all("*" not in m for m in expanded)

    def test_wildcard_removed_when_router_has_no_deployment(self):
        """Already worked before fix — BYOK case."""
        router = Router(model_list=[])

        unique_models = ["openai/*"]
        expanded = _get_wildcard_models(
            unique_models=unique_models,
            return_wildcard_routes=False,
            llm_router=router,
        )
        assert "openai/*" not in unique_models
        assert len(expanded) > 0

    def test_wildcard_removed_when_no_router(self):
        """Already worked before fix — no router at all."""
        unique_models = ["anthropic/*"]
        expanded = _get_wildcard_models(
            unique_models=unique_models,
            return_wildcard_routes=False,
            llm_router=None,
        )
        assert "anthropic/*" not in unique_models
        assert len(expanded) > 0

    def test_return_wildcard_routes_true_keeps_pattern(self):
        """When return_wildcard_routes=True, the pattern should be in expanded."""
        model_list = [
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}}
        ]
        router = Router(model_list=model_list)

        unique_models = ["openai/*"]
        expanded = _get_wildcard_models(
            unique_models=unique_models,
            return_wildcard_routes=True,
            llm_router=router,
        )
        # Pattern should appear in expanded list (not unique_models)
        assert "openai/*" in expanded
        # But it must still be removed from unique_models to avoid duplicates
        assert "openai/*" not in unique_models

    def test_multiple_wildcards_all_removed(self):
        """Multiple wildcard providers should all be removed."""
        model_list = [
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}},
            {
                "model_name": "anthropic/*",
                "litellm_params": {"model": "anthropic/*"},
            },
        ]
        router = Router(model_list=model_list)

        unique_models = ["openai/*", "anthropic/*", "my-custom-model"]
        _get_wildcard_models(
            unique_models=unique_models,
            return_wildcard_routes=False,
            llm_router=router,
        )
        assert "openai/*" not in unique_models
        assert "anthropic/*" not in unique_models
        assert "my-custom-model" in unique_models

    def test_wildcard_removed_even_when_no_expansion_possible(self):
        """A wildcard for an unknown provider with no deployments must still
        be removed from unique_models. Users should never see glob patterns
        in the /models response, even if we cannot expand them."""
        router = Router(model_list=[])

        unique_models = ["unknownprovider/*", "my-custom-model"]
        expanded = _get_wildcard_models(
            unique_models=unique_models,
            return_wildcard_routes=False,
            llm_router=router,
        )
        # Wildcard must be removed even though nothing was expanded
        assert "unknownprovider/*" not in unique_models
        assert "my-custom-model" in unique_models
        # No models expanded for unknown provider
        assert len(expanded) == 0

    def test_wildcard_removed_when_no_router(self):
        """Even without a router, unknown-provider wildcards are removed."""
        unique_models = ["unknownprovider/*"]
        expanded = _get_wildcard_models(
            unique_models=unique_models,
            return_wildcard_routes=False,
            llm_router=None,
        )
        assert "unknownprovider/*" not in unique_models
        assert len(expanded) == 0

    def test_global_star_wildcard_removed_with_router(self):
        """The bare '*' wildcard must be removed from unique_models even
        when no concrete models can be expanded (e.g. no API keys set)."""
        router = Router(model_list=[])

        unique_models = ["*", "my-custom-model"]
        expanded = _get_wildcard_models(
            unique_models=unique_models,
            return_wildcard_routes=False,
            llm_router=router,
        )
        assert "*" not in unique_models
        assert "my-custom-model" in unique_models

    def test_global_star_wildcard_removed_without_router(self):
        """The bare '*' wildcard must be removed even without a router."""
        unique_models = ["*"]
        expanded = _get_wildcard_models(
            unique_models=unique_models,
            return_wildcard_routes=False,
            llm_router=None,
        )
        assert "*" not in unique_models


class TestGetCompleteModelListEndToEnd:
    """End-to-end test matching issue #13752 scenario."""

    def test_models_endpoint_no_wildcards_in_response(self):
        """Issue #13752: /models returns openai/*, anthropic/* etc."""
        model_list = [
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}},
            {
                "model_name": "anthropic/*",
                "litellm_params": {"model": "anthropic/*"},
            },
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
            },
        ]
        router = Router(model_list=model_list)

        result = get_complete_model_list(
            key_models=[],
            team_models=[],
            proxy_model_list=["openai/*", "anthropic/*", "gemini/*"],
            user_model=None,
            infer_model_from_keys=False,
            return_wildcard_routes=False,
            llm_router=router,
        )
        # No wildcard patterns in output
        for model in result:
            assert "*" not in model, f"Wildcard pattern found in /models response: {model}"
        # Should contain real expanded model names
        assert len(result) > 0

    def test_models_endpoint_with_mixed_wildcard_and_explicit(self):
        """Explicit models should survive alongside wildcard expansion."""
        model_list = [
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}},
            {
                "model_name": "my-fine-tuned-model",
                "litellm_params": {"model": "openai/ft:gpt-4o:my-org:custom:id"},
            },
        ]
        router = Router(model_list=model_list)

        result = get_complete_model_list(
            key_models=[],
            team_models=[],
            proxy_model_list=["openai/*", "my-fine-tuned-model"],
            user_model=None,
            infer_model_from_keys=False,
            return_wildcard_routes=False,
            llm_router=router,
        )
        # Wildcard gone
        assert "openai/*" not in result
        # Explicit model preserved
        assert "my-fine-tuned-model" in result
        # Real OpenAI models present
        openai_models = [m for m in result if m.startswith("openai/")]
        assert len(openai_models) > 0

    def test_return_wildcard_routes_true_includes_patterns(self):
        """When explicitly requested, wildcards should appear."""
        model_list = [
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}},
        ]
        router = Router(model_list=model_list)

        result = get_complete_model_list(
            key_models=[],
            team_models=[],
            proxy_model_list=["openai/*"],
            user_model=None,
            infer_model_from_keys=False,
            return_wildcard_routes=True,
            llm_router=router,
        )
        # Pattern should be present when explicitly requested
        assert "openai/*" in result
        # Real models should also be present
        real_models = [m for m in result if "*" not in m]
        assert len(real_models) > 0


class TestRegressionIssue13752:
    """Regression tests for issue #13752: the exact scenario from the bug
    report where the router has wildcard deployments and /models returns
    raw wildcard patterns instead of expanded concrete model names."""

    def test_bug_report_scenario_router_with_wildcard_deployments(self):
        """Reproduce the exact config from issue #13752.

        Config had:
            model_list:
              - model_name: openai/*
                litellm_params:
                  model: openai/*
              - model_name: anthropic/*
                litellm_params:
                  model: anthropic/*

        Expected: /models returns concrete model names (openai/gpt-4o, etc.)
        Bug: /models returned "openai/*", "anthropic/*" as literal strings.
        """
        model_list = [
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}},
            {
                "model_name": "anthropic/*",
                "litellm_params": {"model": "anthropic/*"},
            },
        ]
        router = Router(model_list=model_list)

        result = get_complete_model_list(
            key_models=[],
            team_models=[],
            proxy_model_list=["openai/*", "anthropic/*"],
            user_model=None,
            infer_model_from_keys=False,
            return_wildcard_routes=False,
            llm_router=router,
        )

        # Core assertion: no wildcard patterns in the response
        wildcards_found = [m for m in result if "*" in m]
        assert wildcards_found == [], (
            f"Wildcard patterns leaked into /models response: {wildcards_found}"
        )

        # Should have expanded to real model names from both providers
        openai_models = [m for m in result if m.startswith("openai/")]
        anthropic_models = [m for m in result if m.startswith("anthropic/")]
        assert len(openai_models) > 0, "Expected expanded OpenAI models"
        assert len(anthropic_models) > 0, "Expected expanded Anthropic models"
