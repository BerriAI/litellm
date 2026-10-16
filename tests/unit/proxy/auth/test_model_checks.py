from typing import Final, get_args
from unittest.mock import patch

import httpx
import pytest
import respx

from litellm.llms.bedrock.common_utils import BedrockRoute

_BEDROCK_MODEL_ROUTES: Final = (
    *(route for route in get_args(BedrockRoute) if route not in ("agent", "agentcore")),
    "nova",
    "nova-2",
)


def test_get_team_models_for_all_models_and_team_only_models():
    from litellm.proxy.auth.model_checks import get_team_models

    team_models = ["all-proxy-models", "team-only-model", "team-only-model-2"]
    proxy_model_list = ["model1", "model2", "model3"]
    model_access_groups = {}
    include_model_access_groups = False

    result: Final = get_team_models(team_models, proxy_model_list, model_access_groups, include_model_access_groups)
    combined_models = team_models + proxy_model_list
    assert set(result) == set(combined_models)


def test_get_team_models_all_proxy_models_includes_access_groups():
    """
    When a team has 'all-proxy-models' and include_model_access_groups=True,
    the result should include model access group names (e.g. 'claude-model-group')
    in addition to individual model names.
    """
    from litellm.proxy.auth.model_checks import get_team_models

    team_models = ["all-proxy-models"]
    proxy_model_list = ["model1", "model2"]
    model_access_groups = {
        "group-a": ["model1"],
        "group-b": ["model2"],
    }

    result = get_team_models(
        team_models,
        proxy_model_list,
        model_access_groups,
        include_model_access_groups=True,
    )
    assert "group-a" in result
    assert "group-b" in result
    assert "model1" in result
    assert "model2" in result
    assert len(result) == len(set(result)), "result should have no duplicates"


def test_get_team_models_all_proxy_models_without_include_flag():
    """
    When include_model_access_groups=False, access group names should NOT
    appear in the result even with 'all-proxy-models'.
    """
    from litellm.proxy.auth.model_checks import get_team_models

    team_models = ["all-proxy-models"]
    proxy_model_list = ["model1", "model2"]
    model_access_groups = {
        "group-a": ["model1"],
        "group-b": ["model2"],
    }

    result = get_team_models(
        team_models,
        proxy_model_list,
        model_access_groups,
        include_model_access_groups=False,
    )
    assert "group-a" not in result
    assert "group-b" not in result
    assert "model1" in result
    assert "model2" in result


def test_get_key_models_all_proxy_models_includes_access_groups():
    """
    When a key has 'all-proxy-models' and include_model_access_groups=True,
    the result should include model access group names.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.model_checks import get_key_models

    user_api_key_dict = UserAPIKeyAuth(
        models=["all-proxy-models"],
        api_key="test-key",
    )
    proxy_model_list = ["model1", "model2"]
    model_access_groups = {
        "group-a": ["model1"],
    }

    result = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=proxy_model_list,
        model_access_groups=model_access_groups,
        include_model_access_groups=True,
    )
    assert "group-a" in result
    assert "model1" in result
    assert "model2" in result
    assert len(result) == len(set(result)), "result should have no duplicates"


def test_get_key_models_passes_include_model_access_groups():
    """
    When a key explicitly has an access group name in its models list and
    include_model_access_groups=True, the group name should be retained
    (not stripped by _get_models_from_access_groups).
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.model_checks import get_key_models

    user_api_key_dict = UserAPIKeyAuth(
        models=["group-a"],
        api_key="test-key",
    )
    proxy_model_list = ["model1", "model2"]
    model_access_groups = {
        "group-a": ["model1", "model2"],
    }

    result = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=proxy_model_list,
        model_access_groups=model_access_groups,
        include_model_access_groups=True,
    )
    assert "group-a" in result
    assert "model1" in result
    assert "model2" in result


def test_get_key_models_keeps_literal_model_colliding_with_group_name():
    """A name that is BOTH a deployed model and an access group grants both at
    runtime (_check_model_access_helper unions them), so the listing must keep
    the literal model alongside the group members instead of dropping it."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.model_checks import get_key_models

    user_api_key_dict = UserAPIKeyAuth(models=["beta-models"], api_key="test-key")

    result = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=["beta-models", "member-a", "unrelated"],
        model_access_groups={"beta-models": ["member-a"]},
        include_model_access_groups=False,
    )
    assert sorted(result) == ["beta-models", "member-a"]


def test_get_team_models_keeps_literal_model_colliding_with_group_name():
    """Team flavor of the collision case: literal deployment survives group expansion."""
    from litellm.proxy.auth.model_checks import get_team_models

    result = get_team_models(
        team_models=["beta-models"],
        proxy_model_list=["beta-models", "member-a", "unrelated"],
        model_access_groups={"beta-models": ["member-a"]},
        include_model_access_groups=False,
    )
    assert sorted(result) == ["beta-models", "member-a"]


def test_get_team_models_drops_group_name_that_is_not_a_deployed_model():
    """No collision: a pure access-group name is still replaced by its members."""
    from litellm.proxy.auth.model_checks import get_team_models

    result = get_team_models(
        team_models=["beta-models"],
        proxy_model_list=["member-a", "unrelated"],
        model_access_groups={"beta-models": ["member-a"]},
        include_model_access_groups=False,
    )
    assert result == ["member-a"]


def test_get_key_models_does_not_mutate_input():
    """
    get_key_models must not mutate user_api_key_dict.models in-place.
    _get_models_from_access_groups uses .pop()/.extend() which would corrupt
    cached UserAPIKeyAuth objects if all_models were an alias instead of a copy.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.model_checks import get_key_models

    original_models = ["group-a", "extra-model"]
    user_api_key_dict = UserAPIKeyAuth(
        models=list(original_models),  # give it a list
        api_key="test-key",
    )
    model_access_groups = {
        "group-a": ["model1", "model2"],
    }

    _ = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=["model1", "model2"],
        model_access_groups=model_access_groups,
        include_model_access_groups=False,
    )
    # The original models list on the auth object must be unchanged
    assert user_api_key_dict.models == original_models


@pytest.mark.parametrize(
    "key_models,team_models,proxy_model_list,model_list,expected",
    [
        (
            [
                "anthropic/claude-3-haiku-20240307",
                "anthropic/claude-3-5-haiku-20241022",
            ],
            [],
            [],
            [{"model_name": "anthropic/*", "litellm_params": {"model": "anthropic/*"}}],
            [
                "anthropic/claude-3-haiku-20240307",
                "anthropic/claude-3-5-haiku-20241022",
            ],
        ),
        (
            [],
            [
                "anthropic/claude-3-haiku-20240307",
                "anthropic/claude-3-5-haiku-20241022",
            ],
            [],
            [{"model_name": "anthropic/*", "litellm_params": {"model": "anthropic/*"}}],
            [
                "anthropic/claude-3-haiku-20240307",
                "anthropic/claude-3-5-haiku-20241022",
            ],
        ),
        (
            [],
            [],
            [
                "anthropic/claude-3-haiku-20240307",
                "anthropic/claude-3-5-haiku-20241022",
            ],
            [{"model_name": "anthropic/*", "litellm_params": {"model": "anthropic/*"}}],
            [
                "anthropic/claude-3-haiku-20240307",
                "anthropic/claude-3-5-haiku-20241022",
            ],
        ),
    ],
)
def test_get_complete_model_list_order(key_models, team_models, proxy_model_list, model_list, expected):
    """
    Test that get_complete_model_list preserves order
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list
    from litellm import Router

    assert (
        get_complete_model_list(
            proxy_model_list=proxy_model_list,
            key_models=key_models,
            team_models=team_models,
            user_model=None,
            infer_model_from_keys=False,
            llm_router=Router(model_list=model_list),
        )
        == expected
    )


def test_get_complete_model_list_byok_wildcard_expansion():
    """
    Test that wildcard models (e.g., openai/*) are expanded when the router has
    no deployment for them - BYOK case where team has openai/* but proxy has
    no openai config.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list
    from litellm import Router

    # Router with empty model_list - no openai/* deployment (BYOK scenario)
    result = get_complete_model_list(
        key_models=[],
        team_models=["openai/*"],
        proxy_model_list=[],
        user_model=None,
        infer_model_from_keys=False,
        llm_router=Router(model_list=[]),
    )
    # Should expand openai/* to actual OpenAI models
    assert len(result) > 0
    assert all(m.startswith("openai/") for m in result)
    assert "openai/*" not in result


def test_get_complete_model_list_expands_team_scoped_wildcard_with_stored_credential(
    monkeypatch,
):
    """
    Team-scoped BYOK wildcard deployments are stored under an internal model_name,
    with the public wildcard name in model_info.team_public_model_name.
    """
    import litellm
    from litellm import Router
    from litellm.proxy.auth import model_checks
    from litellm.proxy.auth.model_checks import get_complete_model_list
    from litellm.types.utils import CredentialItem

    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="openai-credential",
                credential_info={"provider": "openai"},
                credential_values={
                    "api_key": "stored-openai-key",
                    "api_base": "https://example.openai.test/v1",
                },
            )
        ],
    )

    captured_params = {}

    def fake_get_provider_models(provider, litellm_params=None):
        captured_params["provider"] = provider
        captured_params["api_key"] = litellm_params.api_key
        captured_params["api_base"] = litellm_params.api_base
        captured_params["credential_name"] = litellm_params.litellm_credential_name
        return ["gpt-4o"]

    monkeypatch.setattr(model_checks, "get_provider_models", fake_get_provider_models)

    router = Router(
        model_list=[
            {
                "model_name": "model_name_team-1_generated",
                "litellm_params": {
                    "model": "openai/*",
                    "custom_llm_provider": "openai",
                    "litellm_credential_name": "openai-credential",
                },
                "model_info": {
                    "team_id": "team-1",
                    "team_public_model_name": "openai/*",
                },
            }
        ]
    )

    result = get_complete_model_list(
        key_models=[],
        team_models=["openai/*"],
        proxy_model_list=[],
        user_model=None,
        infer_model_from_keys=False,
        llm_router=router,
        team_id="team-1",
    )

    assert "openai/gpt-4o" in result
    assert captured_params == {
        "provider": "openai",
        "api_key": "stored-openai-key",
        "api_base": "https://example.openai.test/v1",
        "credential_name": None,
    }


def test_wildcard_credential_hydration_preserves_deployment_params(
    monkeypatch,
):
    import litellm
    from litellm.proxy.auth import model_checks
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params
    from litellm.types.utils import CredentialItem

    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="openai-credential",
                credential_info={"provider": "openai"},
                credential_values={
                    "api_key": "stored-openai-key",
                    "api_version": "credential-version",
                    "model": "openai/wrong-model",
                    "unexpected_field": "unexpected-value",
                },
            )
        ],
    )

    captured_params = {}

    def fake_get_provider_models(provider, litellm_params=None):
        captured_params["provider"] = provider
        captured_params["model"] = litellm_params.model
        captured_params["api_key"] = litellm_params.api_key
        captured_params["api_version"] = litellm_params.api_version
        captured_params["credential_name"] = litellm_params.litellm_credential_name
        captured_params["has_unexpected_field"] = hasattr(litellm_params, "unexpected_field")
        return ["gpt-4o"]

    monkeypatch.setattr(model_checks, "get_provider_models", fake_get_provider_models)

    result = get_known_models_from_wildcard(
        wildcard_model="openai/*",
        litellm_params=LiteLLM_Params(
            model="openai/*",
            custom_llm_provider="openai",
            api_version="deployment-version",
            litellm_credential_name="openai-credential",
        ),
    )

    assert result == ["openai/gpt-4o"]
    assert captured_params == {
        "provider": "openai",
        "model": "openai/*",
        "api_key": "stored-openai-key",
        "api_version": "deployment-version",
        "credential_name": None,
        "has_unexpected_field": False,
    }


def test_wildcard_custom_prefix_does_not_stack_provider_prefix(monkeypatch):
    """Regression test for #30358.

    A wildcard with a custom prefix (e.g. ``ollama_server1/*`` to distinguish multiple Ollama
    instances) must not stack the provider's own prefix onto the expanded model ids. The expanded
    ids should be ``ollama_server1/gemma3:1b`` rather than ``ollama_server1/ollama/gemma3:1b``.
    """
    from litellm.proxy.auth import model_checks
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    monkeypatch.setattr(
        model_checks,
        "get_provider_models",
        lambda provider, litellm_params=None: ["ollama/gemma3:1b", "ollama/llama3:8b"],
    )

    result = get_known_models_from_wildcard(
        wildcard_model="ollama_server1/*",
        litellm_params=LiteLLM_Params(model="ollama_chat/*", custom_llm_provider="ollama_chat"),
    )

    assert result == ["ollama_server1/gemma3:1b", "ollama_server1/llama3:8b"]


def test_wildcard_custom_prefix_keeps_org_segment_for_non_provider_first_segment(
    monkeypatch,
):
    """Only a known provider prefix should be stripped before re-prefixing.

    If ``get_provider_models`` returns ids whose first segment is an org rather than a litellm
    provider (e.g. ``meta-llama/Llama-3-8B``), stripping the first slash segment would drop the
    org and produce an uncallable id. The org segment must be preserved.
    """
    from litellm.proxy.auth import model_checks
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    monkeypatch.setattr(
        model_checks,
        "get_provider_models",
        lambda provider, litellm_params=None: ["meta-llama/Llama-3-8B"],
    )

    result = get_known_models_from_wildcard(
        wildcard_model="my_hf/*",
        litellm_params=LiteLLM_Params(model="huggingface/*", custom_llm_provider="huggingface"),
    )

    assert result == ["my_hf/meta-llama/Llama-3-8B"]


def test_cloudflare_partial_wildcard_matches_provider_stripped_cost_map_ids(
    monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None
) -> None:
    import litellm
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    wildcard: Final = "cloudflare/@cf/*"
    monkeypatch.setattr(litellm, "check_provider_endpoint", False)

    result: Final = tuple(
        get_known_models_from_wildcard(
            wildcard_model=wildcard,
            litellm_params=LiteLLM_Params(model=wildcard),
        )
    )

    assert result
    assert all(model_id.startswith("cloudflare/@cf/") for model_id in result)
    assert all(model_id in litellm.model_cost for model_id in result)
    assert all(model_id.count("@cf/") == 1 for model_id in result)


def test_gemini_partial_wildcard_deduplicates_cost_map_ids(
    monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None
) -> None:
    import litellm
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    monkeypatch.setattr(litellm, "check_provider_endpoint", False)
    wildcard: Final = "gemini/gemini-2.5*"

    result: Final = get_known_models_from_wildcard(
        wildcard_model=wildcard,
        litellm_params=LiteLLM_Params(model=wildcard),
    )

    assert len(result) == len(set(result))
    assert result.count("gemini/gemini-2.5-flash") == 1


def test_gemini_model_info_partial_wildcard_has_unique_model_names(
    monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None
) -> None:
    import litellm
    from litellm.proxy.auth.model_checks import expand_wildcard_deployments_for_model_info

    monkeypatch.setattr(litellm, "check_provider_endpoint", False)
    wildcard: Final = "gemini/gemini-2.5*"

    expanded: Final = tuple(
        expand_wildcard_deployments_for_model_info(
            [{"model_name": wildcard, "litellm_params": {"model": wildcard}}]
        )
    )
    model_names: Final = tuple(str(deployment["model_name"]) for deployment in expanded)

    assert model_names
    assert len(model_names) == len(set(model_names))


@pytest.mark.parametrize(
    ("model_name", "deployment_model", "route_type_prefix"),
    [
        ("cloudflare/@cf/*", "cloudflare/@cf/*", ""),
        ("gemini/gemini-2.5*", "gemini/gemini-2.5*", ""),
        ("vertex_ai/gemini-*", "vertex_ai/gemini-*", ""),
        ("fireworks_ai/accounts/fireworks/models/llama*", "fireworks_ai/accounts/fireworks/models/llama*", ""),
        ("genai/test/*", "openai/*", ""),
        ("team/claude-*", "anthropic/claude-*", ""),
        ("team/oai-*", "openai/*", ""),
        ("team/*", "anthropic/claude-*", ""),
        ("custom/gemini-*", "gemini/*", ""),
        ("*", "openai/*", ""),
        ("*", "cloudflare/@cf/*", ""),
        ("*", "gemini/gemini-2.5*", ""),
        ("openai/*", "openai/responses/*", "responses/"),
        *(("bedrock/*", f"bedrock/{route}/*", f"{route}/") for route in _BEDROCK_MODEL_ROUTES),
    ],
)
def test_every_known_model_a_wildcard_reaches_is_listed_once_under_the_id_that_routes_to_it(
    monkeypatch: pytest.MonkeyPatch,
    local_model_cost_map: None,
    model_name: str,
    deployment_model: str,
    route_type_prefix: str,
) -> None:
    from fnmatch import fnmatchcase

    import litellm
    from litellm.proxy.auth.model_checks import (
        expand_wildcard_deployments_for_model_info,
        get_known_models_from_wildcard,
    )
    from litellm.router_utils.pattern_match_deployments import PatternMatchRouter
    from litellm.types.router import LiteLLM_Params

    monkeypatch.setattr(litellm, "check_provider_endpoint", False)
    provider, _, deployment_path = deployment_model.partition("/")
    model_pattern: Final = deployment_path.removeprefix(route_type_prefix)
    known_ids: Final = frozenset(
        model_id.removeprefix(f"{provider}/") for model_id in litellm.models_by_provider[provider]
    )
    router: Final = PatternMatchRouter()
    router.add_pattern(model_name, {"model_name": model_name, "litellm_params": {"model": deployment_model}})

    listed: Final = get_known_models_from_wildcard(
        wildcard_model=model_name,
        litellm_params=LiteLLM_Params(model=deployment_model),
    )
    routed: Final = tuple(
        str((router.route(model_id) or [{}])[0].get("litellm_params", {}).get("model", "")) for model_id in listed
    )
    model_info_rows: Final = expand_wildcard_deployments_for_model_info(
        [{"model_name": model_name, "litellm_params": {"model": deployment_model}}]
    )
    routed_ids: Final = frozenset(model.removeprefix(f"{provider}/{route_type_prefix}") for model in routed)

    assert listed
    assert len(listed) == len(set(listed))
    assert [model for model in routed if not model.startswith(f"{provider}/{route_type_prefix}")] == []
    assert routed_ids == {model_id for model_id in known_ids if fnmatchcase(model_id, model_pattern)}
    assert [(row["model_name"], row["litellm_params"]["model"]) for row in model_info_rows] == list(zip(listed, routed))


@pytest.mark.parametrize("route_type_prefix", ["o_series/", "gpt5_series/", "responses/"])
def test_azure_route_type_infix_lists_the_same_ids_as_the_route_without_it(
    monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None, route_type_prefix: str
) -> None:
    import litellm
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    monkeypatch.setattr(litellm, "check_provider_endpoint", False)

    without_infix: Final = get_known_models_from_wildcard("azure/*", LiteLLM_Params(model="azure/*"))

    assert without_infix
    assert get_known_models_from_wildcard("azure/*", LiteLLM_Params(model=f"azure/{route_type_prefix}*")) == (
        without_infix
    )


@pytest.mark.parametrize(
    ("model_name", "deployment_model"),
    [
        ("*", "*"),
        ("team/*", "openai/fo::*::static::*"),
    ],
)
def test_wildcard_route_that_cannot_name_a_known_model_lists_no_ids(
    monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None, model_name: str, deployment_model: str
) -> None:
    import litellm
    from litellm.proxy.auth.model_checks import (
        expand_wildcard_deployments_for_model_info,
        get_known_models_from_wildcard,
    )
    from litellm.types.router import LiteLLM_Params

    monkeypatch.setattr(litellm, "check_provider_endpoint", False)
    deployment: Final = {"model_name": model_name, "litellm_params": {"model": deployment_model}}

    assert (
        get_known_models_from_wildcard(wildcard_model=model_name, litellm_params=LiteLLM_Params(model=deployment_model))
        == []
    )
    assert expand_wildcard_deployments_for_model_info([deployment]) == [deployment]


@pytest.mark.parametrize(
    ("model_name", "deployment_model", "listed_id"),
    [
        ("team/*", "openai/gpt-4o-mini", "team/gpt-4o-mini"),
        ("*", "openai/gpt-4o-mini", "gpt-4o-mini"),
        ("team/*", "bedrock/converse/amazon.nova-micro-v1:0", "team/amazon.nova-micro-v1:0"),
    ],
)
def test_wildcard_route_over_one_fixed_model_lists_that_model_once(
    model_name: str, deployment_model: str, listed_id: str
) -> None:
    from litellm.proxy.auth.model_checks import (
        expand_wildcard_deployments_for_model_info,
        get_known_models_from_wildcard,
    )
    from litellm.router_utils.pattern_match_deployments import PatternMatchRouter
    from litellm.types.router import LiteLLM_Params

    deployment: Final = {"model_name": model_name, "litellm_params": {"model": deployment_model}}
    router: Final = PatternMatchRouter()
    router.add_pattern(model_name, deployment)

    assert (
        get_known_models_from_wildcard(wildcard_model=model_name, litellm_params=LiteLLM_Params(model=deployment_model))
        == [listed_id]
    )
    assert (router.route(listed_id) or [{}])[0]["litellm_params"]["model"] == deployment_model
    assert [
        (row["model_name"], row["litellm_params"]["model"])
        for row in expand_wildcard_deployments_for_model_info([deployment])
    ] == [(listed_id, deployment_model)]


def test_wildcard_lists_no_id_that_a_more_specific_route_serves(
    monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None
) -> None:
    import copy

    import litellm
    from litellm import Router
    from litellm.proxy.auth.model_checks import expand_wildcard_deployments_for_model_info, get_complete_model_list

    monkeypatch.setattr(litellm, "check_provider_endpoint", False)
    monkeypatch.setitem(
        litellm.models_by_provider, "openrouter", ["openrouter/openai/gpt-4o-via-openrouter", "meta-llama/llama-3-8b"]
    )
    router: Final = Router(
        model_list=[
            {"model_name": "*", "litellm_params": {"model": "openrouter/*", "api_key": "key"}},
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*", "api_key": "key"}},
        ]
    )

    def served_by(model_id: str) -> str:
        return str((router.pattern_router.route(model_id) or [{}])[0]["litellm_params"]["model"])

    def listed_for(key_models: list[str]) -> list[str]:
        return get_complete_model_list(
            key_models=key_models,
            team_models=[],
            proxy_model_list=[],
            user_model=None,
            infer_model_from_keys=False,
            llm_router=router,
        )

    listed_by_star: Final = listed_for(["*"])
    model_info_rows: Final = expand_wildcard_deployments_for_model_info(
        copy.deepcopy(router.model_list), pattern_router=router.pattern_router
    )

    assert sorted(listed_by_star) == sorted(["meta-llama/llama-3-8b", *listed_for(["openai/*"])])
    assert "openai/gpt-4o" in [row["model_name"] for row in model_info_rows]
    assert [
        row["model_name"] for row in model_info_rows if served_by(row["model_name"]) != row["litellm_params"]["model"]
    ] == []


def test_partial_wildcard_without_a_deployment_lists_the_same_ids_as_with_one(
    monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None
) -> None:
    import litellm
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    monkeypatch.setattr(litellm, "check_provider_endpoint", False)
    wildcard: Final = "cloudflare/@cf/*"

    without_deployment: Final = get_known_models_from_wildcard(wildcard_model=wildcard, litellm_params=None)

    assert without_deployment
    assert without_deployment == get_known_models_from_wildcard(
        wildcard_model=wildcard, litellm_params=LiteLLM_Params(model=wildcard)
    )


def test_explicit_model_that_a_wildcard_also_covers_is_listed_once(local_model_cost_map: None) -> None:
    from litellm.proxy.auth.model_checks import get_complete_model_list

    listed: Final = get_complete_model_list(
        key_models=[],
        team_models=[],
        proxy_model_list=["openai/gpt-4o-mini", "openai/*"],
        user_model=None,
        infer_model_from_keys=False,
    )

    assert "openai/gpt-4o-mini" in listed
    assert len(listed) == len(set(listed))


@pytest.mark.timeout(5)
def test_key_wildcard_with_many_stars_lists_promptly(local_model_cost_map: None) -> None:
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    many_stars: Final = "openai/" + "*" * 30

    assert get_known_models_from_wildcard(wildcard_model=f"{many_stars}!", litellm_params=None) == []
    assert "openai/gpt-4o" in get_known_models_from_wildcard(wildcard_model=f"{many_stars}o", litellm_params=None)


@pytest.mark.parametrize(
    ("wildcard", "expected"),
    [
        ("wandb/*", ["wandb/openai/gpt-oss-20b", "wandb/openai/gpt-oss-120b"]),
        ("wandb/openai/*", ["wandb/openai/gpt-oss-20b", "wandb/openai/gpt-oss-120b"]),
        ("wandb/gpt*", []),
    ],
)
def test_wildcard_keeps_a_vendor_namespace_that_matches_another_provider_name(
    monkeypatch: pytest.MonkeyPatch, wildcard: str, expected: list[str]
) -> None:
    import litellm
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    monkeypatch.setattr(litellm, "check_provider_endpoint", False)
    monkeypatch.setitem(litellm.models_by_provider, "wandb", ["openai/gpt-oss-20b", "wandb/openai/gpt-oss-120b"])

    assert get_known_models_from_wildcard(wildcard_model=wildcard, litellm_params=LiteLLM_Params(model=wildcard)) == (
        expected
    )


def test_complete_model_list_lists_each_id_once_across_deployments_of_one_wildcard(
    monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None
) -> None:
    import litellm
    from litellm import Router
    from litellm.proxy.auth.model_checks import get_complete_model_list

    monkeypatch.setattr(litellm, "check_provider_endpoint", False)
    wildcard: Final = "gemini/gemini-2.5*"
    router: Final = Router(
        model_list=[
            {"model_name": wildcard, "litellm_params": {"model": wildcard, "api_key": "key-a"}},
            {"model_name": wildcard, "litellm_params": {"model": wildcard, "api_key": "key-b"}},
        ]
    )

    listed: Final = get_complete_model_list(
        key_models=[],
        team_models=[],
        proxy_model_list=[wildcard],
        user_model=None,
        infer_model_from_keys=False,
        llm_router=router,
    )

    assert listed
    assert len(listed) == len(set(listed))


@pytest.mark.parametrize(
    "wildcard",
    ["databricks/system.ai.*", "openai/ft:gpt-4o*", "anthropic/claude-nonexistent-*"],
)
def test_partial_wildcard_matching_no_known_model_lists_nothing(
    monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None, wildcard: str
) -> None:
    import litellm
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    monkeypatch.setattr(litellm, "check_provider_endpoint", False)

    assert (
        get_known_models_from_wildcard(
            wildcard_model=wildcard,
            litellm_params=LiteLLM_Params(model=wildcard, api_key="x", api_base="https://example.invalid"),
        )
        == []
    )


def test_wildcard_credential_hydration_preserves_missing_credential_name(
    monkeypatch,
):
    import litellm
    from litellm.proxy.auth import model_checks
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    monkeypatch.setattr(litellm, "credential_list", [])

    captured_params = {}

    def fake_get_provider_models(provider, litellm_params=None):
        captured_params["provider"] = provider
        captured_params["api_key"] = litellm_params.api_key
        captured_params["credential_name"] = litellm_params.litellm_credential_name
        return ["gpt-4o"]

    monkeypatch.setattr(model_checks, "get_provider_models", fake_get_provider_models)

    result = get_known_models_from_wildcard(
        wildcard_model="openai/*",
        litellm_params=LiteLLM_Params(
            model="openai/*",
            custom_llm_provider="openai",
            api_key=None,
            litellm_credential_name="missing-credential",
        ),
    )

    assert result == ["openai/gpt-4o"]
    assert captured_params == {
        "provider": "openai",
        "api_key": None,
        "credential_name": "missing-credential",
    }


def test_hydrate_credential_name_none_leaves_params_untouched(monkeypatch):
    import litellm
    from litellm.proxy.auth.model_checks import _hydrate_litellm_credential_name
    from litellm.types.router import LiteLLM_Params
    from litellm.types.utils import CredentialItem

    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="shared-credential",
                credential_info={},
                credential_values={"api_key": "sk-shared"},
            )
        ],
    )
    params = LiteLLM_Params(model="openai/gpt-4o", litellm_credential_name=None)

    result = _hydrate_litellm_credential_name(params)

    assert result is not None
    assert result.api_key is None
    assert result.litellm_credential_name is None


def test_hydrate_replaced_credential_uses_new_credential_values(monkeypatch):
    import litellm
    from litellm.proxy.auth.model_checks import _hydrate_litellm_credential_name
    from litellm.types.router import LiteLLM_Params
    from litellm.types.utils import CredentialItem

    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="shared-credential",
                credential_info={},
                credential_values={"api_key": "sk-shared"},
            ),
            CredentialItem(
                credential_name="other-credential",
                credential_info={},
                credential_values={"api_key": "sk-other"},
            ),
        ],
    )
    params = LiteLLM_Params(model="openai/gpt-4o", litellm_credential_name="other-credential")

    result = _hydrate_litellm_credential_name(params)

    assert result is not None
    assert result.api_key == "sk-other"
    assert result.litellm_credential_name is None


def test_hydrate_inline_api_key_wins_over_stored_credential(monkeypatch):
    import litellm
    from litellm.proxy.auth.model_checks import _hydrate_litellm_credential_name
    from litellm.types.router import LiteLLM_Params
    from litellm.types.utils import CredentialItem

    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="shared-credential",
                credential_info={},
                credential_values={"api_key": "sk-shared"},
            )
        ],
    )
    params = LiteLLM_Params(
        model="openai/gpt-4o",
        api_key="sk-inline",
        litellm_credential_name="shared-credential",
    )

    result = _hydrate_litellm_credential_name(params)

    assert result is not None
    assert result.api_key == "sk-inline"


@pytest.mark.asyncio
async def test_get_available_models_for_user_expands_query_team_wildcard(
    monkeypatch,
):
    import litellm
    from litellm import Router
    from litellm.proxy.auth import model_checks
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import get_available_models_for_user
    from litellm.types.utils import CredentialItem

    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="openai-credential",
                credential_info={"provider": "openai"},
                credential_values={"api_key": "stored-openai-key"},
            )
        ],
    )

    def fake_get_provider_models(provider, litellm_params=None):
        assert litellm_params.api_key == "stored-openai-key"
        assert litellm_params.litellm_credential_name is None
        return ["gpt-4o-mini"]

    monkeypatch.setattr(model_checks, "get_provider_models", fake_get_provider_models)

    router = Router(
        model_list=[
            {
                "model_name": "model_name_team-1_generated",
                "litellm_params": {
                    "model": "openai/*",
                    "custom_llm_provider": "openai",
                    "litellm_credential_name": "openai-credential",
                },
                "model_info": {
                    "team_id": "team-1",
                    "team_public_model_name": "openai/*",
                },
            }
        ]
    )

    result = await get_available_models_for_user(
        user_api_key_dict=UserAPIKeyAuth(
            api_key="sk-test",
            models=[],
            team_id="team-1",
            team_models=["openai/*"],
        ),
        llm_router=router,
        general_settings={},
        user_model=None,
        team_id="team-1",
    )

    assert "openai/gpt-4o-mini" in result


def test_get_key_models_all_team_models_recursive_team():
    """GH#30619: when key and team both have all-team-models,
    the sentinel should expand to proxy_model_list."""
    from litellm.proxy.auth.model_checks import get_key_models
    from litellm.proxy._types import SpecialModelNames

    user_api_key_dict = type(
        "obj",
        (object,),
        {
            "models": [SpecialModelNames.all_team_models.value],
            "team_id": "team-1",
            "team_models": [SpecialModelNames.all_team_models.value],
        },
    )()
    proxy_model_list = ["model-a", "model-b"]
    result = get_key_models(user_api_key_dict, proxy_model_list, {})
    assert SpecialModelNames.all_team_models.value not in result
    assert set(result) == {"model-a", "model-b"}


def test_get_key_models_all_team_models_keeps_mixed_team_entries():
    from litellm.proxy.auth.model_checks import get_key_models
    from litellm.proxy._types import SpecialModelNames

    user_api_key_dict = type(
        "obj",
        (object,),
        {
            "models": [SpecialModelNames.all_team_models.value],
            "team_id": "team-1",
            "team_models": [
                SpecialModelNames.all_team_models.value,
                "restricted-model",
            ],
        },
    )()
    result = get_key_models(user_api_key_dict, ["model-a", "model-b"], {})
    assert SpecialModelNames.all_team_models.value not in result
    assert set(result) == {"model-a", "model-b", "restricted-model"}


def test_get_team_models_all_team_models_expands():
    """GH#30619: all-team-models in team_models should expand."""
    from litellm.proxy.auth.model_checks import get_team_models
    from litellm.proxy._types import SpecialModelNames

    result = get_team_models(
        [SpecialModelNames.all_team_models.value],
        ["model-a", "model-b"],
        {},
    )
    assert SpecialModelNames.all_team_models.value not in result
    assert set(result) == {"model-a", "model-b"}


def test_get_team_models_all_team_models_expands_with_access_groups():
    """GH#30619: all-team-models with include_model_access_groups
    should include access group keys."""
    from litellm.proxy.auth.model_checks import get_team_models
    from litellm.proxy._types import SpecialModelNames

    result = get_team_models(
        [SpecialModelNames.all_team_models.value],
        ["model-a", "model-b"],
        {"group-1": ["g1-model"], "group-2": ["g2-model"]},
        include_model_access_groups=True,
    )
    assert SpecialModelNames.all_team_models.value not in result
    assert "model-a" in result
    assert "model-b" in result
    assert "group-1" in result
    assert "group-2" in result


def test_get_key_models_teamless_all_team_models_returns_unrestricted():
    """Teamless key with all-team-models must resolve the same as leaving the
    models field empty ([] = unrestricted). The sentinel must not leak into
    the returned list. Fails if someone adds a team_id guard to the sentinel
    expansion in get_key_models."""
    from litellm.proxy._types import SpecialModelNames
    from litellm.proxy.auth.model_checks import get_key_models

    user_api_key_dict = type(
        "obj",
        (object,),
        {
            "models": [SpecialModelNames.all_team_models.value],
            "team_id": None,
            "team_models": [],
        },
    )()
    proxy_model_list = ["gpt-4o", "claude-sonnet-4-20250514"]
    result = get_key_models(user_api_key_dict, proxy_model_list, {})
    assert SpecialModelNames.all_team_models.value not in result
    assert result == [], "should return [] (unrestricted), same as an unscoped key"


def test_expand_wildcard_deployments_non_wildcard_passthrough():
    """Non-wildcard deployments must be returned unchanged."""
    from litellm.proxy.auth.model_checks import (
        expand_wildcard_deployments_for_model_info,
    )

    deployment = {"model_name": "gpt-4o", "litellm_params": {"model": "gpt-4o"}}
    result = expand_wildcard_deployments_for_model_info([deployment])
    assert result == [deployment]


def test_expand_wildcard_deployments_openai_wildcard(monkeypatch: pytest.MonkeyPatch):
    """openai/* should expand into ≥1 known openai model entries."""
    import litellm
    from litellm.proxy.auth.model_checks import (
        expand_wildcard_deployments_for_model_info,
    )

    fake_models = ["openai/gpt-4o", "openai/gpt-4o-mini"]
    monkeypatch.setattr(litellm, "check_provider_endpoint", False)
    monkeypatch.setitem(litellm.models_by_provider, "openai", ["gpt-4o", "gpt-4o-mini"])
    deployment = {
        "model_name": "openai/*",
        "litellm_params": {"model": "openai/*"},
    }
    result = expand_wildcard_deployments_for_model_info([deployment])

    assert len(result) == 2
    assert all(r["model_name"] in fake_models for r in result)
    assert all(r["litellm_params"]["model"] in fake_models for r in result)


def test_expand_wildcard_lists_each_id_once_under_the_deployment_that_serves_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm
    from litellm.proxy.auth.model_checks import expand_wildcard_deployments_for_model_info
    from litellm.router_utils.pattern_match_deployments import PatternMatchRouter

    monkeypatch.setattr(litellm, "check_provider_endpoint", False)
    monkeypatch.setitem(litellm.models_by_provider, "openai", ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"])
    wide: Final = {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}, "model_info": {"id": "wide"}}
    narrow: Final = {
        "model_name": "openai/gpt-4.1*",
        "litellm_params": {"model": "openai/gpt-4.1*"},
        "model_info": {"id": "narrow"},
    }
    exact: Final = {
        "model_name": "openai/gpt-4o-mini",
        "litellm_params": {"model": "openai/gpt-4o-mini"},
        "model_info": {"id": "exact"},
    }
    pattern_router: Final = PatternMatchRouter()
    pattern_router.add_pattern("openai/*", wide)
    pattern_router.add_pattern("openai/gpt-4.1*", narrow)

    rows: Final = expand_wildcard_deployments_for_model_info([wide, narrow, exact], pattern_router=pattern_router)

    assert sorted((row["model_name"], row["model_info"]["id"]) for row in rows) == [
        ("openai/gpt-4.1-mini", "narrow"),
        ("openai/gpt-4o", "wide"),
        ("openai/gpt-4o-mini", "exact"),
    ]


def test_key_scoped_to_a_wide_route_lists_ids_a_narrower_route_sends_to_the_same_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm
    from litellm import Router
    from litellm.proxy.auth.model_checks import get_complete_model_list

    monkeypatch.setattr(litellm, "check_provider_endpoint", False)
    monkeypatch.setitem(litellm.models_by_provider, "openai", ["gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"])
    router: Final = Router(
        model_list=[
            {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}},
            {"model_name": "openai/gpt-4.1*", "litellm_params": {"model": "openai/gpt-4.1*"}},
        ]
    )

    listed: Final = get_complete_model_list(
        key_models=["openai/*"],
        team_models=[],
        proxy_model_list=["openai/*", "openai/gpt-4.1*"],
        user_model=None,
        infer_model_from_keys=False,
        llm_router=router,
    )

    assert sorted(listed) == ["openai/gpt-4.1-mini", "openai/gpt-4.1-nano", "openai/gpt-4o"]


@pytest.mark.parametrize(
    ("key_models", "expected"),
    [
        ([], ["wide/gpt-4.1", "wide/gpt-4.1-mini", "wide/gpt-4o", "wide/gpt-4o-mini", "wide/gpt-5"]),
        (["wide/*"], ["wide/gpt-4.1", "wide/gpt-4.1-mini", "wide/gpt-4o", "wide/gpt-4o-mini", "wide/gpt-5"]),
        (["wide/gpt-4.1*"], ["wide/gpt-4.1", "wide/gpt-4.1-mini"]),
        (["wide/gpt-4o*"], ["wide/gpt-4o", "wide/gpt-4o-mini"]),
        (["other/*"], []),
    ],
    ids=("admin", "key_on_wide_route", "key_on_narrow_route", "key_pattern_without_a_route", "key_elsewhere"),
)
def test_key_scoped_to_a_wildcard_lists_the_ids_every_route_serves_under_its_pattern(
    monkeypatch: pytest.MonkeyPatch, key_models: list[str], expected: list[str]
) -> None:
    import litellm
    from litellm import Router
    from litellm.proxy.auth.model_checks import get_complete_model_list

    monkeypatch.setattr(litellm, "check_provider_endpoint", False)
    monkeypatch.setitem(litellm.models_by_provider, "openai", ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-5"])
    router: Final = Router(
        model_list=[
            {"model_name": "wide/*", "litellm_params": {"model": "openai/*"}},
            {"model_name": "wide/gpt-4.1*", "litellm_params": {"model": "openai/gpt-4o*"}},
        ]
    )

    listed: Final = get_complete_model_list(
        key_models=key_models,
        team_models=[],
        proxy_model_list=["wide/*", "wide/gpt-4.1*"],
        user_model=None,
        infer_model_from_keys=False,
        llm_router=router,
    )

    assert sorted(listed) == expected


def test_key_scoped_to_a_wildcard_lists_only_its_own_teams_narrower_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm
    from litellm import Router
    from litellm.proxy.auth.model_checks import get_complete_model_list

    monkeypatch.setattr(litellm, "check_provider_endpoint", False)
    monkeypatch.setitem(litellm.models_by_provider, "openai", ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-5"])
    router: Final = Router(
        model_list=[
            {"model_name": "wide/*", "litellm_params": {"model": "openai/*"}},
            {
                "model_name": "team-b-narrow",
                "litellm_params": {"model": "openai/gpt-4o*"},
                "model_info": {"team_id": "team-b", "team_public_model_name": "wide/gpt-4.1*"},
            },
        ]
    )

    def listed_for(team_id: str) -> list[str]:
        return sorted(
            get_complete_model_list(
                key_models=["wide/*"],
                team_models=[],
                proxy_model_list=["wide/*"],
                user_model=None,
                infer_model_from_keys=False,
                llm_router=router,
                team_id=team_id,
            )
        )

    assert "wide/gpt-4.1-mini" in listed_for("team-b")
    assert "wide/gpt-4.1-mini" not in listed_for("team-a")


def _wide_route_beside_a_narrower_route_to_another_provider(monkeypatch: pytest.MonkeyPatch):
    import litellm
    from litellm import Router

    monkeypatch.setattr(litellm, "check_provider_endpoint", False)
    monkeypatch.setitem(litellm.models_by_provider, "openai", ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"])
    monkeypatch.setitem(litellm.models_by_provider, "azure", [])
    return Router(
        model_list=[
            {"model_name": "az/*", "litellm_params": {"model": "openai/*"}, "model_info": {"id": "wide"}},
            {"model_name": "az/gpt-4o*", "litellm_params": {"model": "azure/gpt-4o*"}, "model_info": {"id": "narrow"}},
        ]
    )


def test_listing_keeps_ids_a_narrower_route_sends_to_the_same_model_at_another_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.auth.model_checks import get_complete_model_list

    router: Final = _wide_route_beside_a_narrower_route_to_another_provider(monkeypatch)

    listed: Final = get_complete_model_list(
        key_models=[],
        team_models=[],
        proxy_model_list=["az/*", "az/gpt-4o*"],
        user_model=None,
        infer_model_from_keys=False,
        llm_router=router,
    )

    assert {"az/gpt-4o", "az/gpt-4o-mini", "az/gpt-4.1-mini"} <= set(listed)


def test_model_info_keeps_rows_a_narrower_route_cannot_list_but_sends_to_the_same_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.auth.model_checks import expand_wildcard_deployments_for_model_info

    router: Final = _wide_route_beside_a_narrower_route_to_another_provider(monkeypatch)

    rows: Final = expand_wildcard_deployments_for_model_info(router.get_model_list(), router.pattern_router)

    assert sorted(row["model_name"] for row in rows if row["model_info"]["id"] == "wide") == [
        "az/gpt-4.1-mini",
        "az/gpt-4o",
        "az/gpt-4o-mini",
    ]


def test_expand_wildcard_concrete_model_name_with_wildcard_litellm_params():
    """Concrete model_name must not be overwritten when only litellm_params.model is wildcard."""
    from litellm.proxy.auth.model_checks import (
        expand_wildcard_deployments_for_model_info,
    )

    deployment = {
        "model_name": "my-custom-alias",
        "litellm_params": {"model": "openai/*"},
    }
    result = expand_wildcard_deployments_for_model_info([deployment])
    # model_name is not a wildcard, so the deployment passes through unchanged
    assert result == [deployment]


def test_expand_wildcard_invalid_litellm_params_passthrough():
    """Deployments with invalid litellm_params must pass through unchanged (no 500)."""
    from litellm.proxy.auth.model_checks import (
        expand_wildcard_deployments_for_model_info,
    )

    deployment = {
        "model_name": "openai/*",
        "litellm_params": {
            "model": "openai/*",
            "max_retries": "not-an-int-field-that-breaks",
        },
    }
    # Even if LiteLLM_Params construction fails the deployment should survive
    result = expand_wildcard_deployments_for_model_info([deployment])
    assert result == [deployment]


def test_get_complete_model_list_excludes_wildcard_routes_by_default():
    """Regression (LIT-4108): a wildcard with a matching router deployment leaked into /v1/models."""
    from litellm import Router
    from litellm.proxy.auth.model_checks import get_complete_model_list

    router = Router(
        model_list=[
            {
                "model_name": "bedrock/*",
                "litellm_params": {"model": "bedrock/*"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "openai/gpt-4"},
            },
        ]
    )

    result = get_complete_model_list(
        key_models=[],
        team_models=[],
        proxy_model_list=["bedrock/*", "gpt-4"],
        user_model=None,
        infer_model_from_keys=False,
        return_wildcard_routes=False,
        llm_router=router,
    )

    assert "bedrock/*" not in result
    assert "gpt-4" in result
    assert any(m.startswith("bedrock/") for m in result)


def test_get_complete_model_list_excludes_wildcard_routes_without_router():
    from litellm.proxy.auth.model_checks import get_complete_model_list

    result = get_complete_model_list(
        key_models=[],
        team_models=[],
        proxy_model_list=["bedrock/*", "gpt-4"],
        user_model=None,
        infer_model_from_keys=False,
        return_wildcard_routes=False,
        llm_router=None,
    )

    assert "bedrock/*" not in result
    assert "gpt-4" in result
    assert any(m.startswith("bedrock/") for m in result)


def test_get_complete_model_list_includes_wildcard_routes_when_requested():
    from litellm import Router
    from litellm.proxy.auth.model_checks import get_complete_model_list

    router = Router(
        model_list=[
            {
                "model_name": "bedrock/*",
                "litellm_params": {"model": "bedrock/*"},
            },
        ]
    )

    result = get_complete_model_list(
        key_models=[],
        team_models=[],
        proxy_model_list=["bedrock/*"],
        user_model=None,
        infer_model_from_keys=False,
        return_wildcard_routes=True,
        llm_router=router,
    )

    assert result.count("bedrock/*") == 1
    assert any(m.startswith("bedrock/") and m != "bedrock/*" for m in result)


def test_add_known_models_refreshes_models_by_provider_for_wildcard_expansion():
    """models_by_provider was a frozen import-time snapshot of set unions, so cost map
    reloads (which call add_known_models) never reached wildcard expansion until a
    process restart (LIT-4947)."""
    import litellm
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    fake_model = "vertex_ai/gemini-lit4947-regression"
    captured_reference = litellm.models_by_provider
    assert fake_model not in litellm.models_by_provider["vertex_ai"]
    try:
        litellm.add_known_models(
            model_cost_map={fake_model: {"litellm_provider": "vertex_ai-language-models", "mode": "chat"}}
        )
        assert fake_model in litellm.models_by_provider["vertex_ai"]
        assert litellm.models_by_provider is captured_reference
        assert fake_model in captured_reference["vertex_ai"]
        assert fake_model in get_known_models_from_wildcard("vertex_ai/*")
    finally:
        litellm.vertex_language_models.discard(fake_model)
        litellm.add_known_models(model_cost_map={})
    assert fake_model not in litellm.models_by_provider["vertex_ai"]


def test_get_complete_model_list_drops_no_default_models_sentinel():
    from litellm.proxy.auth.model_checks import get_complete_model_list

    result = get_complete_model_list(
        key_models=["no-default-models", "model-a"],
        team_models=[],
        proxy_model_list=["model-a", "model-b"],
        user_model=None,
        infer_model_from_keys=False,
    )
    assert result == ["model-a"]


def test_get_complete_model_list_sentinel_only_grants_nothing():
    from litellm.proxy.auth.model_checks import get_complete_model_list

    result = get_complete_model_list(
        key_models=["no-default-models"],
        team_models=["no-default-models"],
        proxy_model_list=["model-a", "model-b"],
        user_model=None,
        infer_model_from_keys=False,
    )
    assert result == []


def test_get_provider_models_admits_providers_without_a_static_catalog():
    """Providers without a static model list are no longer rejected up front.

    With endpoint discovery off, get_valid_models falls back to the (empty)
    static list, so the result is [] rather than None. Before the fix this
    returned None and the wildcard was never expanded.
    """
    import litellm
    from litellm.proxy.auth.model_checks import get_provider_models
    from litellm.types.router import LiteLLM_Params

    assert "litellm_proxy" not in litellm.models_by_provider
    assert "hosted_vllm" not in litellm.models_by_provider

    result = get_provider_models(
        "litellm_proxy",
        litellm_params=LiteLLM_Params(
            model="litellm_proxy/*",
            api_base="http://upstream:4000",
            api_key="sk-upstream",
        ),
    )

    assert result == []


def test_get_complete_model_list_discovers_litellm_proxy_wildcard_models(monkeypatch):
    """A litellm_proxy/* deployment lists the upstream proxy's models when endpoint discovery is on."""
    import litellm
    from litellm import Router
    from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
    from litellm.proxy.auth.model_checks import get_complete_model_list

    monkeypatch.setattr(litellm, "check_provider_endpoint", True)
    captured = {}

    def fake_get_models(self, api_key=None, api_base=None):
        captured["api_key"] = api_key
        captured["api_base"] = api_base
        return ["gpt-4o", "claude-sonnet"]

    monkeypatch.setattr(OpenAIGPTConfig, "get_models", fake_get_models)

    router = Router(
        model_list=[
            {
                "model_name": "litellm_proxy/*",
                "litellm_params": {
                    "model": "litellm_proxy/*",
                    "api_base": "http://upstream:4000",
                    "api_key": "sk-upstream",
                },
            }
        ]
    )
    result = get_complete_model_list(
        key_models=[],
        team_models=[],
        proxy_model_list=["litellm_proxy/*"],
        user_model=None,
        infer_model_from_keys=False,
        llm_router=router,
    )

    assert captured == {"api_key": "sk-upstream", "api_base": "http://upstream:4000"}
    assert "litellm_proxy/gpt-4o" in result
    assert "litellm_proxy/claude-sonnet" in result


def test_get_provider_models_returns_none_for_an_unknown_provider():
    from litellm.proxy.auth.model_checks import get_provider_models

    assert get_provider_models("not-a-real-provider") is None


def test_transcribe_is_a_known_provider_for_wildcard_expansion():
    import litellm
    from litellm.proxy.auth.model_checks import (
        get_known_models_from_wildcard,
        get_provider_models,
    )

    assert "transcribe" in litellm.models_by_provider
    assert "transcribe/StartTranscriptionJob" in litellm.models_by_provider["transcribe"]
    assert get_provider_models("transcribe") == ["transcribe/StartTranscriptionJob"]
    assert get_known_models_from_wildcard("transcribe/*") == ["transcribe/StartTranscriptionJob"]


@respx.mock
def test_partial_bedrock_wildcard_filters_the_discovered_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    monkeypatch.setattr(litellm, "check_provider_endpoint", True)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    region: Final = "ap-south-1"
    respx.get(
        f"https://bedrock.{region}.amazonaws.com/foundation-models", params={"byInferenceType": "ON_DEMAND"}
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "modelSummaries": [
                    {"modelId": "anthropic.claude-haiku-4-5-20251001-v1:0"},
                    {"modelId": "amazon.nova-micro-v1:0"},
                ]
            },
        )
    )
    respx.get(
        f"https://bedrock.{region}.amazonaws.com/inference-profiles", params={"typeEquals": "SYSTEM_DEFINED"}
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "inferenceProfileSummaries": [
                    {"inferenceProfileId": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "status": "ACTIVE"}
                ]
            },
        )
    )
    deployment: Final = LiteLLM_Params(
        model="bedrock/anthropic.*",
        aws_access_key_id="AKIAPARTIALWILDCARD",
        aws_secret_access_key="partial-wildcard-secret",
        aws_region_name=region,
    )

    assert get_known_models_from_wildcard("bedrock/anthropic.*", deployment) == [
        "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0"
    ]


@respx.mock
def test_partial_bedrock_wildcard_lists_nothing_when_no_discovered_id_carries_its_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    monkeypatch.setattr(litellm, "check_provider_endpoint", True)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    region: Final = "ap-southeast-2"
    respx.get(
        f"https://bedrock.{region}.amazonaws.com/foundation-models", params={"byInferenceType": "ON_DEMAND"}
    ).mock(return_value=httpx.Response(200, json={"modelSummaries": [{"modelId": "amazon.nova-micro-v1:0"}]}))
    respx.get(
        f"https://bedrock.{region}.amazonaws.com/inference-profiles", params={"typeEquals": "SYSTEM_DEFINED"}
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "inferenceProfileSummaries": [
                    {"inferenceProfileId": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "status": "ACTIVE"}
                ]
            },
        )
    )
    deployment: Final = LiteLLM_Params(
        model="bedrock/anthropic.*",
        aws_access_key_id="AKIAPROFILEONLYACCOUNT",
        aws_secret_access_key="profile-only-secret",
        aws_region_name=region,
    )

    assert get_known_models_from_wildcard("bedrock/anthropic.*", deployment) == []


@respx.mock
def test_custom_prefix_that_starts_a_discovered_id_still_prefixes_every_listed_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    monkeypatch.setattr(litellm, "check_provider_endpoint", True)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    region: Final = "ca-central-1"
    respx.get(
        f"https://bedrock.{region}.amazonaws.com/foundation-models", params={"byInferenceType": "ON_DEMAND"}
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "modelSummaries": [
                    {"modelId": "anthropic.claude-haiku-4-5-20251001-v1:0"},
                    {"modelId": "amazon.nova-micro-v1:0"},
                ]
            },
        )
    )
    respx.get(
        f"https://bedrock.{region}.amazonaws.com/inference-profiles", params={"typeEquals": "SYSTEM_DEFINED"}
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "inferenceProfileSummaries": [
                    {"inferenceProfileId": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "status": "ACTIVE"}
                ]
            },
        )
    )
    deployment: Final = LiteLLM_Params(
        model="bedrock/*",
        aws_access_key_id="AKIACUSTOMPREFIXACCOUNT",
        aws_secret_access_key="custom-prefix-secret",
        aws_region_name=region,
    )

    assert get_known_models_from_wildcard("anthropic/*", deployment) == [
        "anthropic/amazon.nova-micro-v1:0",
        "anthropic/anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    ]
