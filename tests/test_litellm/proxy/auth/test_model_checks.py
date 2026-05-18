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


def test_get_complete_model_list_strips_no_default_models_sentinel():
    """
    `no-default-models` is an internal sentinel meaning "no standalone model
    grants - only inherited (team / access group) grants apply". It must never
    appear in the advertised /v1/models list, regardless of where it was set.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list

    # Sentinel sitting in team_models alongside real models - the bug scenario
    # where /v1/models would otherwise leak the literal string to OpenWebUI.
    result = get_complete_model_list(
        key_models=[],
        team_models=["no-default-models", "gpt-4o", "claude-opus-4-5"],
        proxy_model_list=["gpt-3.5-turbo"],
        user_model=None,
        infer_model_from_keys=False,
    )
    assert "no-default-models" not in result
    assert set(result) == {"gpt-4o", "claude-opus-4-5"}

    # Sentinel alone in key_models -> result must be empty (nothing leaks
    # through, and team_models is NOT consulted because key_models is truthy).
    result_key_only = get_complete_model_list(
        key_models=["no-default-models"],
        team_models=["gpt-4o"],
        proxy_model_list=[],
        user_model=None,
        infer_model_from_keys=False,
    )
    assert result_key_only == []


@pytest.mark.asyncio
async def test_get_available_models_for_user_expands_db_access_group_ids():
    """
    Regression: when a team has models=['no-default-models'] and grants model
    access via team.access_group_ids (LiteLLM_AccessGroupTable), /v1/models
    must return the expanded access_model_names. Previously it returned only
    the sentinel because the listing path ignored access_group_ids entirely,
    even though the enforcement path (can_team_access_model) honored them.

    team_access_group_ids is denormalised onto the verification-token row via
    combined_view, so we put it directly on UserAPIKeyAuth here (mirroring the
    runtime data flow) instead of patching a separate DB lookup.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import get_available_models_for_user

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed",
        user_id="u1",
        team_id="team-123",
        team_models=["no-default-models"],
        team_access_group_ids=["ag-premium"],
        models=[],
        access_group_ids=[],
    )

    async def fake_expand_db_access_group_ids(access_group_ids, **kwargs):
        if "ag-premium" in (access_group_ids or []):
            return ["gpt-4o", "claude-opus-4-5"]
        return []

    # Patch target is the source module (litellm.proxy.auth.auth_checks). This
    # works because get_available_models_for_user imports the helper inside its
    # function body, so the name is resolved against the source module at call
    # time. If those imports are ever hoisted to module level in utils.py, this
    # patch must change to target litellm.proxy.utils.expand_db_access_group_ids.
    with patch(
        "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
        side_effect=fake_expand_db_access_group_ids,
    ):
        result = await get_available_models_for_user(
            user_api_key_dict=user_api_key_dict,
            llm_router=None,
            general_settings={},
            user_model=None,
            prisma_client=AsyncMock(),
            proxy_logging_obj=AsyncMock(),
            user_api_key_cache=AsyncMock(),
        )

    assert "no-default-models" not in result
    assert set(result) == {"gpt-4o", "claude-opus-4-5"}


@pytest.mark.asyncio
async def test_get_available_models_for_user_expands_key_access_group_ids():
    """
    Key-level (not team-level) access_group_ids must also expand into the
    advertised model list. Covers the key_models branch alongside the
    existing team_models test.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import get_available_models_for_user

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed",
        user_id="u1",
        team_id=None,
        team_models=[],
        team_access_group_ids=[],
        models=["no-default-models"],
        access_group_ids=["ag-key-direct"],
    )

    async def fake_expand_db_access_group_ids(access_group_ids, **kwargs):
        if "ag-key-direct" in (access_group_ids or []):
            return ["gpt-4o"]
        return []

    with patch(
        "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
        side_effect=fake_expand_db_access_group_ids,
    ):
        result = await get_available_models_for_user(
            user_api_key_dict=user_api_key_dict,
            llm_router=None,
            general_settings={},
            user_model=None,
            prisma_client=AsyncMock(),
            proxy_logging_obj=AsyncMock(),
            user_api_key_cache=AsyncMock(),
        )

    assert "no-default-models" not in result
    assert result == ["gpt-4o"]


@pytest.mark.asyncio
async def test_get_available_models_for_user_explicit_team_id_query_param():
    """
    Covers the explicit `?team_id=` query-param branch in /v1/models, which
    calls get_team_object and captures team.access_group_ids from the
    returned team object (instead of relying on the verification-token
    denormalisation that the default path uses).
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import get_available_models_for_user

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed",
        user_id="u1",
        team_id="team-456",
        team_models=[],
        team_access_group_ids=[],
        models=[],
        access_group_ids=[],
    )

    fake_team_obj = AsyncMock()
    fake_team_obj.team_id = "team-456"
    fake_team_obj.models = ["no-default-models"]
    fake_team_obj.access_group_ids = ["ag-team-explicit"]

    async def fake_get_team_object(team_id, **kwargs):
        return fake_team_obj

    async def fake_validate_membership(*args, **kwargs):
        return None

    async def fake_expand_db_access_group_ids(access_group_ids, **kwargs):
        if "ag-team-explicit" in (access_group_ids or []):
            return ["claude-opus-4-5"]
        return []

    with (
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            side_effect=fake_get_team_object,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.validate_membership",
            side_effect=fake_validate_membership,
        ),
        patch(
            "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
            side_effect=fake_expand_db_access_group_ids,
        ),
    ):
        result = await get_available_models_for_user(
            user_api_key_dict=user_api_key_dict,
            llm_router=None,
            general_settings={},
            user_model=None,
            prisma_client=AsyncMock(),
            proxy_logging_obj=AsyncMock(),
            team_id="team-456",
            user_api_key_cache=AsyncMock(),
        )

    assert "no-default-models" not in result
    assert result == ["claude-opus-4-5"]
