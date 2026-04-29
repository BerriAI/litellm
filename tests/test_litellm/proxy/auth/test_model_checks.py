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


def test_get_complete_model_list_drops_stale_access_group_string():
    """
    Regression for issue #25550.

    A virtual key with `models=["team-sales-api"]` where "team-sales-api"
    is neither a configured proxy model nor an active access group should
    NOT leak the bare access-group string into /v1/models.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list

    result = get_complete_model_list(
        key_models=["team-sales-api"],
        team_models=[],
        proxy_model_list=["gpt-4o-mini"],
        user_model=None,
        infer_model_from_keys=False,
        model_access_groups={},
    )

    assert result == []


def test_get_complete_model_list_drops_stale_access_group_string_team():
    """Same regression as above but exercised through the team_models path."""
    from litellm.proxy.auth.model_checks import get_complete_model_list

    result = get_complete_model_list(
        key_models=[],
        team_models=["team-sales-api"],
        proxy_model_list=["gpt-4o-mini"],
        user_model=None,
        infer_model_from_keys=False,
        model_access_groups={},
    )

    assert result == []


def test_get_complete_model_list_keeps_active_access_group_expansion():
    """
    Active access groups still expand to their member models. The filter
    must not interfere with the existing expansion path (i.e., key_models
    must arrive already-expanded from get_key_models).
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.model_checks import get_complete_model_list, get_key_models

    user_api_key_dict = UserAPIKeyAuth(
        models=["group-engineering"],
        api_key="test-key",
    )
    proxy_model_list = ["gpt-4o-mini"]
    model_access_groups = {"group-engineering": ["gpt-4o-mini"]}

    key_models = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=proxy_model_list,
        model_access_groups=model_access_groups,
    )

    result = get_complete_model_list(
        key_models=key_models,
        team_models=[],
        proxy_model_list=proxy_model_list,
        user_model=None,
        infer_model_from_keys=False,
        model_access_groups=model_access_groups,
    )

    assert result == ["gpt-4o-mini"]


def test_get_complete_model_list_keeps_provider_qualified_string():
    """
    Provider-qualified identifiers carry a syntactic marker (`/`) and must
    survive the access-group filter — even if the exact model id is not
    present in the static litellm.model_list_set.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list

    result = get_complete_model_list(
        key_models=["bedrock/very-new-model"],
        team_models=[],
        proxy_model_list=[],
        user_model=None,
        infer_model_from_keys=False,
        model_access_groups={},
    )

    assert result == ["bedrock/very-new-model"]


def test_get_complete_model_list_keeps_known_base_model():
    """
    A known LiteLLM base model id (in litellm.model_list_set) must survive
    the filter even when not configured on the proxy.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list

    result = get_complete_model_list(
        key_models=["gpt-4o-mini"],
        team_models=[],
        proxy_model_list=[],
        user_model=None,
        infer_model_from_keys=False,
        model_access_groups={},
    )

    assert result == ["gpt-4o-mini"]


def test_get_complete_model_list_keeps_custom_proxy_alias():
    """
    A custom enterprise proxy model name (e.g. 'internal-assistant') that is
    listed in proxy_model_list but is NOT a known LiteLLM base model id must
    survive the filter. This locks in the proxy_model_list-membership branch
    of _is_unresolvable_model_identifier — the most common preservation path
    in real deployments.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list

    result = get_complete_model_list(
        key_models=["internal-assistant"],
        team_models=[],
        proxy_model_list=["internal-assistant"],
        user_model=None,
        infer_model_from_keys=False,
        model_access_groups={},
    )

    assert result == ["internal-assistant"]


def test_get_complete_model_list_keeps_finetune_id():
    """OpenAI fine-tune ids (`ft:...`) must survive the filter."""
    from litellm.proxy.auth.model_checks import get_complete_model_list

    ft_id = "ft:gpt-4o:my-org:custom-suffix:abc123"
    result = get_complete_model_list(
        key_models=[ft_id],
        team_models=[],
        proxy_model_list=[],
        user_model=None,
        infer_model_from_keys=False,
        model_access_groups={},
    )

    assert result == [ft_id]


def test_get_complete_model_list_does_not_filter_proxy_admin_path():
    """
    The filter must only apply when key_models or team_models is set. When
    both are empty (proxy-admin / scope=expand path), unique_models is
    sourced from the authoritative proxy_model_list and should pass through
    untouched.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list

    result = get_complete_model_list(
        key_models=[],
        team_models=[],
        proxy_model_list=["arbitrary-named-model"],
        user_model=None,
        infer_model_from_keys=False,
        model_access_groups={},
    )

    assert result == ["arbitrary-named-model"]


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
