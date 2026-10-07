"""Behavior pins for ``litellm/router_utils/pattern_match_deployments.py``."""

from __future__ import annotations

from typing import Final
from unittest.mock import Mock, patch

import httpx
import pytest
from pydantic import TypeAdapter

import litellm
from litellm import Router
from litellm.litellm_core_utils import get_llm_provider_logic
from litellm.router import Deployment, LiteLLM_Params
from litellm.router_utils.pattern_match_deployments import PatternMatchRouter, PatternUtils
from litellm.types.router import ModelInfo
import json
from unittest.mock import MagicMock


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

    monkeypatch.setattr(get_llm_provider_logic, "get_llm_provider", _oauth_tripwire)

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


def test_get_pattern_bare_provider_name_never_matches_that_providers_wildcard(monkeypatch):
    """Regression: a bare ``github_copilot`` adopted itself as its provider and retried as
    ``github_copilot/github_copilot``, false-matching the wildcard for a name no deployment serves."""

    def _unknown_provider(model, *args, **kwargs):
        raise ValueError(f"unknown provider for {model}")

    monkeypatch.setattr(get_llm_provider_logic, "get_llm_provider", _unknown_provider)
    router = PatternMatchRouter()
    router.add_pattern("github_copilot/*", _wildcard_deployment("github_copilot/*"))
    assert router.get_pattern("github_copilot") is None


def test_get_pattern_missing_model_returns_none(monkeypatch):
    """Regression: a request without a model reaches the auth layer's pattern walk as ``None``; the
    declared-provider guard raised ``TypeError`` where the old inline resolve swallowed every
    resolver error, so the proxy's missing-model 400 became a crash."""

    def _unknown_provider(model, *args, **kwargs):
        raise ValueError(f"unknown provider for {model}")

    monkeypatch.setattr(get_llm_provider_logic, "get_llm_provider", _unknown_provider)
    router = PatternMatchRouter()
    router.add_pattern("openai/*", _wildcard_deployment("openai/*"))
    assert router.get_pattern(None) is None


def test_get_pattern_still_resolves_unqualified_names(monkeypatch):
    monkeypatch.setattr(
        get_llm_provider_logic,
        "get_llm_provider",
        lambda model, **kwargs: (model, "openai", None, None),
    )
    router = PatternMatchRouter()
    router.add_pattern("openai/*", _wildcard_deployment("openai/*"))
    assert _matched_models(router.get_pattern("gpt-4o")) == ["openai/gpt-4o"]


class _CountingPatternUtils(PatternUtils):
    sorted_patterns = staticmethod(Mock(wraps=PatternUtils.sorted_patterns))


def test_route_never_sorts_and_the_most_specific_pattern_still_wins_after_registry_changes():
    """Regression for LIT-6886: the auth layer walks the wildcard registry for every request, so an
    unmatched model name (an invalid-model 403) re-sorted every pattern by specificity per request and
    a burst of rejections saturated the worker CPU. Lookups must not sort; adding a pattern or removing
    a deployment must still leave the most specific pattern winning."""
    router = PatternMatchRouter(pattern_utils=_CountingPatternUtils)
    router.add_pattern("openai/*", _wildcard_deployment("openai/*"))
    router.add_pattern("anthropic/*", _wildcard_deployment("anthropic/*"))
    router.add_pattern("openai/gpt-*", {"model_name": "openai/gpt-*", "litellm_params": {"model": "azure/gpt-*"}})
    sorts_after_setup = _CountingPatternUtils.sorted_patterns.call_count

    for _ in range(3):
        assert router.route("does-not-exist") is None
    assert _matched_models(router.route("openai/gpt-4o")) == ["azure/gpt-4o"]
    assert _matched_models(router.route("openai/o3")) == ["openai/o3"]
    assert _CountingPatternUtils.sorted_patterns.call_count == sorts_after_setup

    router.add_pattern("openai/*", {**_wildcard_deployment("openai/*"), "model_info": {"id": "id-1"}})
    assert len(_matched_models(router.route("openai/o3"))) == 2
    router.remove_deployment("id-1")
    assert _matched_models(router.route("openai/gpt-4o")) == ["azure/gpt-4o"]
    assert _matched_models(router.route("openai/o3")) == ["openai/o3"]


def test_pattern_match_router_initialization():
    router = PatternMatchRouter()
    assert router.patterns == {}


def test_add_pattern():
    """
    Tests that openai/* is added to the patterns

    when we try to get the pattern, it should return the deployment
    """
    router = PatternMatchRouter()
    deployment = Deployment(
        model_name="openai-1",
        litellm_params=LiteLLM_Params(model="gpt-3.5-turbo"),
        model_info=ModelInfo(),
    )
    router.add_pattern("openai/*", deployment.to_json(exclude_none=True))
    assert len(router.patterns) == 1
    assert list(router.patterns.keys())[0] == "openai/(.*)"

    # try getting the pattern
    assert router.route(request="openai/gpt-15") == [
        deployment.to_json(exclude_none=True)
    ]


def test_add_pattern_vertex_ai():
    """
    Tests that vertex_ai/* is added to the patterns

    when we try to get the pattern, it should return the deployment
    """
    router = PatternMatchRouter()
    deployment = Deployment(
        model_name="this-can-be-anything",
        litellm_params=LiteLLM_Params(model="vertex_ai/gemini-1.5-flash-latest"),
        model_info=ModelInfo(),
    )
    router.add_pattern("vertex_ai/*", deployment.to_json(exclude_none=True))
    assert len(router.patterns) == 1
    assert list(router.patterns.keys())[0] == "vertex_ai/(.*)"

    # try getting the pattern
    assert router.route(request="vertex_ai/gemini-1.5-flash-latest") == [
        deployment.to_json(exclude_none=True)
    ]


def test_add_multiple_deployments():
    """
    Tests adding multiple deployments for the same pattern

    when we try to get the pattern, it should return the deployment
    """
    router = PatternMatchRouter()
    deployment1 = Deployment(
        model_name="openai-1",
        litellm_params=LiteLLM_Params(model="gpt-3.5-turbo"),
        model_info=ModelInfo(),
    )
    deployment2 = Deployment(
        model_name="openai-2",
        litellm_params=LiteLLM_Params(model="gpt-4"),
        model_info=ModelInfo(),
    )
    router.add_pattern("openai/*", deployment1.to_json(exclude_none=True))
    router.add_pattern("openai/*", deployment2.to_json(exclude_none=True))
    assert len(router.route("openai/gpt-4o")) == 2


def test_pattern_to_regex():
    """
    Tests that the pattern is converted to a regex
    """
    router = PatternMatchRouter()
    assert router.pattern_to_regex("openai/*") == "openai/(.*)"
    assert (
        router.pattern_to_regex("openai/fo::*::static::*")
        == "openai/fo::(.*)::static::(.*)"
    )


def test_route_with_none():
    """
    Tests that the router returns None when the request is None
    """
    router = PatternMatchRouter()
    assert router.route(None) is None


def test_route_with_multiple_matching_patterns():
    """
    Tests that the router returns the first matching pattern when there are multiple matching patterns
    """
    router = PatternMatchRouter()
    deployment1 = Deployment(
        model_name="openai-1",
        litellm_params=LiteLLM_Params(model="gpt-3.5-turbo"),
        model_info=ModelInfo(),
    )
    deployment2 = Deployment(
        model_name="openai-2",
        litellm_params=LiteLLM_Params(model="gpt-4"),
        model_info=ModelInfo(),
    )
    router.add_pattern("openai/*", deployment1.to_json(exclude_none=True))
    router.add_pattern("openai/gpt-*", deployment2.to_json(exclude_none=True))
    assert router.route("openai/gpt-3.5-turbo") == [
        deployment2.to_json(exclude_none=True)
    ]


def test_route_with_exception():
    """
    Tests that the router returns None when there is an exception calling router.route()
    """
    router = PatternMatchRouter()
    deployment = Deployment(
        model_name="openai-1",
        litellm_params=LiteLLM_Params(model="gpt-3.5-turbo"),
        model_info=ModelInfo(),
    )
    router.add_pattern("openai/*", deployment.to_json(exclude_none=True))

    router.patterns = (
        []
    )  # this will cause router.route to raise an exception, since router.patterns should be a dict

    result = router.route("openai/gpt-3.5-turbo")
    assert result is None


@pytest.mark.asyncio
async def test_route_with_no_matching_pattern():
    """
    Tests that the router returns None when there is no matching pattern
    """
    from litellm.types.router import RouterErrors

    router = Router(
        model_list=[
            {
                "model_name": "*meta.llama3*",
                "litellm_params": {"model": "bedrock/meta.llama3*"},
            }
        ]
    )

    ## WORKS
    result = await router.acompletion(
        model="bedrock/meta.llama3-70b",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="Works",
    )
    assert result.choices[0].message.content == "Works"

    ## WORKS
    result = await router.acompletion(
        model="meta.llama3-70b-instruct-v1:0",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="Works",
    )
    assert result.choices[0].message.content == "Works"

    ## FAILS
    with pytest.raises(litellm.BadRequestError) as e:
        await router.acompletion(
            model="my-fake-model",
            messages=[{"role": "user", "content": "Hello, world!"}],
            mock_response="Works",
        )

    assert RouterErrors.no_deployments_available.value not in str(e.value)

    with pytest.raises(litellm.BadRequestError):
        await router.aembedding(
            model="my-fake-model",
            input="Hello, world!",
        )


def test_router_pattern_match_e2e():
    """
    Tests the end to end flow of the router
    """
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    router = Router(
        model_list=[
            {
                "model_name": "llmengine/*",
                "litellm_params": {"model": "anthropic/*", "api_key": "test"},
            }
        ]
    )

    with patch.object(client, "post", new=MagicMock()) as mock_post:

        router.completion(
            model="llmengine/my-custom-model",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            client=client,
            api_key="test",
        )
        mock_post.assert_called_once()
        request_body = json.loads(mock_post.call_args.kwargs["data"])
        assert request_body["model"] == "my-custom-model"
        assert request_body["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
        ]


def test_pattern_matching_router_with_default_wildcard_and_model_wildcard():
    """
    Match to more specific pattern first.
    """
    router = Router(
        model_list=[
            {
                "model_name": "*",
                "litellm_params": {"model": "*"},
                "model_info": {"access_groups": ["default"]},
            },
            {
                "model_name": "llmengine/*",
                "litellm_params": {"model": "openai/*"},
            },
        ]
    )

    assert len(router.pattern_router.patterns) > 0

    pattern_router = router.pattern_router
    deployments = pattern_router.route("llmengine/gpt-3.5-turbo")
    assert len(deployments) == 1
    assert deployments[0]["model_name"] == "llmengine/*"


def test_sorted_patterns():
    """
    Tests that the pattern specificity is calculated correctly
    """
    from litellm.router_utils.pattern_match_deployments import PatternUtils

    sorted_patterns = PatternUtils.sorted_patterns(
        {
            "llmengine/*": [{"model_name": "anthropic/claude-3-5-sonnet"}],
            "*": [{"model_name": "openai/*"}],
        },
    )
    assert sorted_patterns[0][0] == "llmengine/*"


def test_calculate_pattern_specificity():
    from litellm.router_utils.pattern_match_deployments import PatternUtils

    assert PatternUtils.calculate_pattern_specificity("llmengine/*") == (11, 1)
    assert PatternUtils.calculate_pattern_specificity("*") == (1, 1)


def test_wildcard_priority_over_deployment_names():
    """
    Test that wildcard routes take priority over deployment_names (litellm_params.model) matching.

    Scenario:
    - deployment 1: model_name="zapier-multi-provider-text-embedding-3-small", model="openai/text-embedding-3-small"
    - deployment 2: model_name="*", model="openai/*"
    - deployment 3: model_name="openai/*", model="openai/*"

    When calling "openai/text-embedding-3-small", it should match deployment 3 (wildcard),
    NOT deployment 1 (even though deployment 1's litellm_params.model matches).

    Priority order should be:
    1. Exact model_name match
    2. Wildcard model_name match
    3. deployment_names (litellm_params.model) match
    """
    router = Router(
        model_list=[
            {
                "model_name": "zapier-multi-provider-text-embedding-3-small",
                "litellm_params": {
                    "model": "openai/text-embedding-3-small",
                    "api_base": "http://localhost:8080/openai",
                    "api_key": "test-key-1",
                },
                "model_info": {
                    "id": "zapier-multi-provider-text-embedding-3-small-openai"
                },
            },
            {
                "model_name": "*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_base": "http://localhost:8081/openai",
                    "api_key": "test-key-2",
                },
            },
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_base": "http://localhost:8082/openai",
                    "api_key": "test-key-3",
                },
            },
        ]
    )

    # Test 1: Request "openai/text-embedding-3-small" should match wildcard "openai/*", not deployment_names
    deployments = router.get_model_list(model_name="openai/text-embedding-3-small")

    assert deployments is not None, "No deployments found"
    assert len(deployments) == 1, f"Expected 1 deployment, got {len(deployments)}"

    # Should match the "openai/*" wildcard deployment (api_base ending in 8082)
    assert (
        deployments[0]["litellm_params"]["api_base"] == "http://localhost:8082/openai"
    ), f"Expected wildcard deployment (8082), got {deployments[0]['litellm_params']['api_base']}"

    # Test 2: Request exact model_name should still work
    deployments = router.get_model_list(
        model_name="zapier-multi-provider-text-embedding-3-small"
    )

    assert deployments is not None, "No deployments found"
    assert len(deployments) == 1, f"Expected 1 deployment, got {len(deployments)}"
    assert (
        deployments[0]["litellm_params"]["api_base"] == "http://localhost:8080/openai"
    ), f"Expected exact match deployment (8080), got {deployments[0]['litellm_params']['api_base']}"

    # Test 3: Request with "*" wildcard should match the "*" deployment
    deployments = router.get_model_list(model_name="some-random-model")

    assert deployments is not None, "No deployments found"
    assert len(deployments) == 1, f"Expected 1 deployment, got {len(deployments)}"
    assert (
        deployments[0]["litellm_params"]["api_base"] == "http://localhost:8081/openai"
    ), f"Expected '*' wildcard deployment (8081), got {deployments[0]['litellm_params']['api_base']}"
