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
        captured_params["has_unexpected_field"] = hasattr(
            litellm_params, "unexpected_field"
        )
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
        litellm_params=LiteLLM_Params(
            model="ollama_chat/*", custom_llm_provider="ollama_chat"
        ),
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
        litellm_params=LiteLLM_Params(
            model="huggingface/*", custom_llm_provider="huggingface"
        ),
    )

    assert result == ["my_hf/meta-llama/Llama-3-8B"]


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


def test_expand_wildcard_deployments_openai_wildcard():
    """openai/* should expand into ≥1 known openai model entries."""
    from unittest.mock import patch

    from litellm.proxy.auth.model_checks import (
        expand_wildcard_deployments_for_model_info,
    )

    fake_models = ["openai/gpt-4o", "openai/gpt-4o-mini"]
    deployment = {
        "model_name": "openai/*",
        "litellm_params": {"model": "openai/*"},
    }
    with patch(
        "litellm.proxy.auth.model_checks.get_known_models_from_wildcard",
        return_value=fake_models,
    ):
        result = expand_wildcard_deployments_for_model_info([deployment])

    assert len(result) == 2
    assert all(r["model_name"] in fake_models for r in result)
    assert all(r["litellm_params"]["model"] in fake_models for r in result)


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
