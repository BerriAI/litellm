from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, Member
from litellm.proxy.auth.handle_jwt import JWTAuthManager


@pytest.mark.asyncio
async def test_access_group_ids_restricts_model_list():
    """
    Issue #23850: when a key has access_group_ids but no native models,
    the access group models must be used as key_models so that
    get_complete_model_list() does NOT fall back to team_models.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.model_checks import get_complete_model_list, get_key_models

    user_api_key_dict = UserAPIKeyAuth(
        models=[],
        team_models=["gpt-4", "claude-3", "gemini-pro"],
        access_group_ids=["ag-limited-123"],
        api_key="test-key",
    )

    proxy_model_list = ["gpt-4", "claude-3", "gpt-3.5-turbo", "gemini-pro"]
    key_models = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=proxy_model_list,
        model_access_groups={},
    )

    # Simulate the inline resolution that proxy_server.py / utils.py now perform
    if not key_models and user_api_key_dict.access_group_ids:
        with patch(
            "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
            new_callable=AsyncMock,
            return_value=["gpt-3.5-turbo"],
        ):
            from litellm.proxy.auth.auth_checks import _get_models_from_access_groups

            key_models = await _get_models_from_access_groups(
                access_group_ids=user_api_key_dict.access_group_ids,
            )

    result = get_complete_model_list(
        key_models=key_models,
        team_models=user_api_key_dict.team_models,
        proxy_model_list=proxy_model_list,
        user_model=None,
        infer_model_from_keys=False,
    )

    assert result == ["gpt-3.5-turbo"], f"Expected only access group models, got: {result}"
    assert "gpt-4" not in result
    assert "claude-3" not in result
    assert "gemini-pro" not in result


@pytest.mark.asyncio
async def test_no_access_group_ids_falls_back_to_team_models():
    """
    When access_group_ids is empty/None, existing fallback behaviour is preserved:
    key_models stays empty so get_complete_model_list() uses team_models.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.model_checks import get_complete_model_list, get_key_models

    user_api_key_dict = UserAPIKeyAuth(
        models=[],
        team_models=["gpt-4", "claude-3"],
        access_group_ids=None,
        api_key="test-key",
    )

    proxy_model_list = ["gpt-4", "claude-3"]
    key_models = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=proxy_model_list,
        model_access_groups={},
    )

    # No access_group_ids → key_models stays empty, no DB call needed
    assert key_models == []

    result = get_complete_model_list(
        key_models=key_models,
        team_models=user_api_key_dict.team_models,
        proxy_model_list=proxy_model_list,
        user_model=None,
        infer_model_from_keys=False,
    )

    # Falls back to team_models
    assert set(result) == {"gpt-4", "claude-3"}


@pytest.mark.asyncio
async def test_native_models_take_precedence_over_access_group_ids():
    """
    When the key already has native model restrictions, access_group_ids
    resolution is skipped — the native model list is authoritative.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.model_checks import get_key_models

    user_api_key_dict = UserAPIKeyAuth(
        models=["gpt-4"],
        team_models=["gpt-4", "claude-3"],
        access_group_ids=["ag-limited-123"],
        api_key="test-key",
    )

    key_models = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=["gpt-4", "claude-3", "gpt-3.5-turbo"],
        model_access_groups={},
    )

    # Native models are non-empty → access_group_ids resolution would be skipped
    assert key_models == ["gpt-4"]
    # The condition `not key_models and access_group_ids` is False, so no DB call


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
