import pytest


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


def test_get_complete_model_list_filters_unknown_non_model_strings():
    """
    If a key contains an arbitrary string that is neither:
    - a configured proxy model
    - a configured access group
    - a known LiteLLM model id
    - nor a recognized provider-qualified route

    it should not leak into the final /v1/models response.
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

    assert "team-sales-api" not in result
    assert result == []


def test_get_complete_model_list_keeps_known_base_model_ids():
    """
    Exact model IDs can be valid even when they are not configured as proxy
    model groups, so known LiteLLM model ids should remain in the final list.
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


def test_get_complete_model_list_keeps_provider_qualified_models():
    """
    Provider-qualified model identifiers should survive filtering even if the
    exact model name is newer than LiteLLM's baked-in model list.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list

    result = get_complete_model_list(
        key_models=["bedrock/very_new_model"],
        team_models=[],
        proxy_model_list=[],
        user_model=None,
        infer_model_from_keys=False,
        model_access_groups={},
    )

    assert result == ["bedrock/very_new_model"]


def test_get_complete_model_list_keeps_json_registry_provider_models():
    """
    JSON-registry-only providers should also survive filtering even when the
    exact model name is not present in LiteLLM's static model list.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list

    result = get_complete_model_list(
        key_models=["publicai/some-new-model"],
        team_models=[],
        proxy_model_list=[],
        user_model=None,
        infer_model_from_keys=False,
        model_access_groups={},
    )

    assert result == ["publicai/some-new-model"]


def test_get_complete_model_list_keeps_openai_finetune_model_ids():
    """
    OpenAI fine-tuned model IDs are dynamic valid models and should remain in
    the final list even though they are not in LiteLLM's static model list.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list

    finetuned_model = "ft:gpt-4o:my-org:custom-suffix:model-id"

    result = get_complete_model_list(
        key_models=[finetuned_model],
        team_models=[],
        proxy_model_list=[],
        user_model=None,
        infer_model_from_keys=False,
        model_access_groups={},
    )

    assert result == [finetuned_model]


@pytest.mark.parametrize(
    "finetuned_model",
    [
        "ft:davinci-002:my-org:custom-suffix:model-id",
        "ft:babbage-002:my-org:custom-suffix:model-id",
    ],
)
def test_get_complete_model_list_keeps_legacy_openai_finetune_model_ids(
    finetuned_model: str,
):
    """
    Legacy OpenAI fine-tuned model IDs should also remain visible even though
    get_llm_provider() does not recognize all historical fine-tune base names.
    """
    from litellm.proxy.auth.model_checks import get_complete_model_list

    result = get_complete_model_list(
        key_models=[finetuned_model],
        team_models=[],
        proxy_model_list=[],
        user_model=None,
        infer_model_from_keys=False,
        model_access_groups={},
    )

    assert result == [finetuned_model]
