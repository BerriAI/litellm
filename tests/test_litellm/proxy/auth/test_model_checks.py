from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, Member
from litellm.proxy.auth.handle_jwt import JWTAuthManager


def test_get_team_models_for_all_models_and_team_only_models():
    from litellm.proxy.auth.model_checks import get_team_models

    team_models = ["all-proxy-models", "team-only-model", "team-only-model-2"]
    proxy_model_list = ["model1", "model2", "model3"]
    model_access_groups = {}
    include_model_access_groups = False

    result = get_team_models(
        team_models, proxy_model_list, model_access_groups, include_model_access_groups
    )
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
def test_get_complete_model_list_order(
    key_models, team_models, proxy_model_list, model_list, expected
):
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


def test_get_complete_model_list_bare_wildcard_star():
    """
    When model_name is "*" (bare wildcard without provider prefix)
    and litellm_params.model contains the provider wildcard (e.g. "openai/*"),
    get_complete_model_list should expand using the provider from litellm_params
    without adding a wildcard alias prefix.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list
    from litellm import Router

    router = Router(
        model_list=[
            {"model_name": "*", "litellm_params": {"model": "openai/*"}},
        ]
    )

    result = get_complete_model_list(
        key_models=[],
        team_models=[],
        proxy_model_list=router.get_model_names(),
        user_model=None,
        infer_model_from_keys=False,
        llm_router=router,
    )

    assert "*" not in result
    assert "openai/*" not in result
    assert len(result) > 0
    assert "gpt-4o" in result or "gpt-3.5-turbo" in result


def test_get_complete_model_list_bare_wildcard_star_slash():
    """
    When model_name is "*/" it should behave the same as bare "*" and avoid
    adding a wildcard alias prefix to expanded models.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list
    from litellm import Router

    router = Router(
        model_list=[
            {"model_name": "*/", "litellm_params": {"model": "openai/*"}},
        ]
    )

    result = get_complete_model_list(
        key_models=[],
        team_models=[],
        proxy_model_list=router.get_model_names(),
        user_model=None,
        infer_model_from_keys=False,
        llm_router=router,
    )

    assert "*/" not in result
    assert len(result) > 0
    assert "gpt-4o" in result or "gpt-3.5-turbo" in result


def test_get_complete_model_list_bare_wildcard_multiple_providers():
    """
    When multiple deployments share model_name "*" with different providers,
    each deployment should expand independently and results should be deduplicated.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list
    from litellm import Router

    router = Router(
        model_list=[
            {"model_name": "*", "litellm_params": {"model": "openai/*"}},
            {"model_name": "*", "litellm_params": {"model": "anthropic/*"}},
        ]
    )

    result = get_complete_model_list(
        key_models=[],
        team_models=[],
        proxy_model_list=router.get_model_names(),
        user_model=None,
        infer_model_from_keys=False,
        llm_router=router,
    )

    assert "*" not in result
    assert "openai/*" not in result
    assert "anthropic/*" not in result
    assert len(result) > 0
    assert len(result) == len(set(result)), "results should be deduplicated"


def test_get_complete_model_list_deduplication():
    """
    When two deployments with wildcard model_name resolve to overlapping models
    (e.g. two ollama/* entries), the result should not contain duplicates.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list
    from litellm import Router

    router = Router(
        model_list=[
            {"model_name": "ollama/*", "litellm_params": {"model": "ollama/*"}},
            {"model_name": "ollama/*", "litellm_params": {"model": "ollama/*"}},
        ]
    )

    result = get_complete_model_list(
        key_models=[],
        team_models=[],
        proxy_model_list=router.get_model_names(),
        user_model=None,
        infer_model_from_keys=False,
        llm_router=router,
    )

    assert "ollama/*" not in result
    assert any(m.startswith("ollama/") for m in result)
    assert len(result) == len(set(result)), "results should be deduplicated"


def test_get_known_models_from_wildcard_avoids_prefix_string_false_positive():
    """
    Providers with model ids like `deepseek-chat` should still get the
    `deepseek/` alias added for `deepseek/*` routes.
    """
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    wildcard_models = get_known_models_from_wildcard(
        wildcard_model="deepseek/*",
        litellm_params=LiteLLM_Params(model="deepseek/*"),
    )

    assert "deepseek/deepseek-chat" in wildcard_models
    assert "deepseek-chat" not in wildcard_models


@patch("litellm.proxy.auth.model_checks.get_provider_models")
def test_get_known_models_from_bare_wildcard_star_does_not_add_alias_prefix(
    mock_get_provider_models,
):
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    mock_get_provider_models.return_value = ["gpt-4o", "gpt-4o-mini"]

    wildcard_models = get_known_models_from_wildcard(
        wildcard_model="*",
        litellm_params=LiteLLM_Params(model="openai/*"),
    )

    assert wildcard_models == ["gpt-4o", "gpt-4o-mini"]


@patch("litellm.proxy.auth.model_checks.get_provider_models")
def test_get_known_models_from_bare_wildcard_star_slash_does_not_add_alias_prefix(
    mock_get_provider_models,
):
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    mock_get_provider_models.return_value = ["gpt-4o", "gpt-4o-mini"]

    wildcard_models = get_known_models_from_wildcard(
        wildcard_model="*/",
        litellm_params=LiteLLM_Params(model="openai/*"),
    )

    assert wildcard_models == ["gpt-4o", "gpt-4o-mini"]


@patch("litellm.proxy.auth.model_checks.get_provider_models")
def test_get_known_models_from_bare_wildcard_preserves_upstream_prefixed_ids(
    mock_get_provider_models,
):
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    mock_get_provider_models.return_value = ["ollama/deepseek-v3.2", "glm-5"]

    wildcard_models = get_known_models_from_wildcard(
        wildcard_model="*",
        litellm_params=LiteLLM_Params(model="ollama/*"),
    )

    assert wildcard_models == ["ollama/deepseek-v3.2", "glm-5"]
