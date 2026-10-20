from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx


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


def test_expand_wildcard_deployments_openai_wildcard():
    """openai/* should expand into ≥1 known openai model entries."""

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


def test_wildcard_lists_a_model_once_when_registered_both_bare_and_provider_prefixed():
    import litellm
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    bare: Final = "gpt-dedupe-regression"
    entry: Final = {"litellm_provider": "openai", "mode": "chat"}
    try:
        litellm.register_model({bare: entry, f"openai/{bare}": entry}, persist_across_reloads=False)

        listed: Final = get_known_models_from_wildcard("openai/*")

        assert listed.count(f"openai/{bare}") == 1
    finally:
        litellm.model_cost.pop(bare, None)
        litellm.model_cost.pop(f"openai/{bare}", None)
        litellm.open_ai_chat_completion_models.discard(bare)
        litellm.open_ai_chat_completion_models.discard(f"openai/{bare}")
        litellm.add_known_models(model_cost_map={})


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


@contextmanager
def _registered_chat_models(provider_by_model: Mapping[str, str]) -> Generator[None, None, None]:
    import litellm

    litellm.register_model(
        {model: {"litellm_provider": provider, "mode": "chat"} for model, provider in provider_by_model.items()},
        persist_across_reloads=False,
    )
    try:
        yield
    finally:
        provider_model_sets: Final = tuple(known for known in vars(litellm).values() if isinstance(known, set))
        for model in provider_by_model:
            litellm.model_cost.pop(model, None)
            for provider_models in provider_model_sets:
                provider_models.discard(model)
        litellm.add_known_models(model_cost_map={})


@pytest.mark.parametrize(
    ("wildcard", "deployment_model", "catalog", "expected"),
    [
        pytest.param(
            "cloudflare/@cf/partial-wildcard-org/*",
            "cloudflare/@cf/partial-wildcard-org/*",
            ("cloudflare/@cf/partial-wildcard-org/model-a", "cloudflare/@hf/partial-wildcard-org/model-b"),
            ["cloudflare/@cf/partial-wildcard-org/model-a"],
            id="prefixed-catalog-with-deployment",
        ),
        pytest.param(
            "cloudflare/@cf/partial-wildcard-org/*",
            None,
            ("cloudflare/@cf/partial-wildcard-org/model-a", "cloudflare/@hf/partial-wildcard-org/model-b"),
            ["cloudflare/@cf/partial-wildcard-org/model-a"],
            id="prefixed-catalog-without-deployment",
        ),
        pytest.param(
            "gemini/gemini-partial-wildcard-9*",
            "gemini/gemini-partial-wildcard-9*",
            (
                "gemini-partial-wildcard-9-flash",
                "gemini/gemini-partial-wildcard-9-pro",
                "gemini/gemini-other-partial-wildcard-9",
            ),
            ["gemini/gemini-partial-wildcard-9-flash", "gemini/gemini-partial-wildcard-9-pro"],
            id="mixed-bare-and-prefixed-catalog",
        ),
    ],
)
def test_partial_wildcard_lists_catalog_ids_stored_with_the_provider_prefix(
    wildcard: str, deployment_model: str | None, catalog: tuple[str, ...], expected: list[str]
) -> None:
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
    from litellm.types.router import LiteLLM_Params

    provider: Final = wildcard.split("/", 1)[0]
    deployment: Final = None if deployment_model is None else LiteLLM_Params(model=deployment_model)
    with _registered_chat_models(dict.fromkeys(catalog, provider)):
        listed: Final = get_known_models_from_wildcard(wildcard, deployment)

    assert sorted(listed) == expected


_ROUTE_CF_ID: Final = "cloudflare/@cf/partial-route-org/model-a-instruct"
_ROUTE_CF_SUFFIXED_ID: Final = "cloudflare/@cf/partial-route-org/model-b-instruct-v2"
_ROUTE_HF_ID: Final = "cloudflare/@hf/partial-route-org/model-c"
_ROUTE_GEMINI_ID: Final = "gemini-partial-route-9-flash"
_ROUTE_OPENAI_ID: Final = "gpt-partial-route-9-mini"
_ROUTE_BEDROCK_ID: Final = "anthropic.partial-route-9-v1:0"
_ROUTE_CF_IDS: Final = {_ROUTE_CF_ID, _ROUTE_CF_SUFFIXED_ID}


@pytest.mark.parametrize(
    ("model_name", "deployment_model", "key_model", "reachable"),
    [
        pytest.param("cloudflare/@cf/*", "cloudflare/@cf/*", None, _ROUTE_CF_IDS, id="filter"),
        pytest.param(
            "cloudflare/@cf/*-instruct", "cloudflare/@cf/*-instruct", None, {_ROUTE_CF_ID}, id="filter-with-suffix"
        ),
        pytest.param(
            "cloudflare/@cf/*",
            "cloudflare/*",
            None,
            {*_ROUTE_CF_IDS, _ROUTE_HF_ID},
            id="alias-whose-prefix-matches-catalog-ids",
        ),
        pytest.param("cloudflare/team-*", "cloudflare/*", None, {*_ROUTE_CF_IDS, _ROUTE_HF_ID}, id="alias"),
        pytest.param("cloudflare/cf-*", "cloudflare/@cf/*", None, _ROUTE_CF_IDS, id="alias-over-partial-deployment"),
        pytest.param("my-cf/*", "cloudflare/@cf/*", None, _ROUTE_CF_IDS, id="bare-alias-over-partial-deployment"),
        pytest.param(
            "cloudflare/team-*", "cloudflare/@cf/*/*", None, set(), id="alias-with-fewer-wildcards-than-its-deployment"
        ),
        pytest.param("partial-route-v/*/*", "cloudflare/@cf/*/*", None, _ROUTE_CF_IDS, id="multi-wildcard-alias"),
        pytest.param(
            "partial-route-v/*-*", "cloudflare/@cf/*/*", None, set(), id="multi-wildcard-alias-routing-splits-elsewhere"
        ),
        pytest.param(
            "partial-route-team/*", "openai/gpt-partial-route-9*", None, set(), id="alias-over-mid-word-deployment"
        ),
        pytest.param(
            "team/gpt-partial-route-9*",
            "openai/gpt-partial-route-9*",
            None,
            {f"openai/{_ROUTE_OPENAI_ID}"},
            id="alias-repeating-the-deployment-partial-word",
        ),
        pytest.param(
            "team/gpt-partial-route-9-prod*",
            "openai/gpt-partial-route-9*",
            None,
            {f"openai/{_ROUTE_OPENAI_ID}"},
            id="alias-extending-the-deployment-partial-word",
        ),
        pytest.param(
            "partial-route-g/*",
            "gemini/gemini-partial-route-*",
            None,
            {_ROUTE_GEMINI_ID},
            id="alias-over-deployment-cut-at-a-word-boundary",
        ),
        pytest.param(
            "cloudflare/*", "cloudflare/*", "cloudflare/@cf/*", _ROUTE_CF_IDS, id="key-filter-over-provider-wildcard"
        ),
        pytest.param("gemini/gemini-*", "gemini/*", None, {_ROUTE_GEMINI_ID}, id="alias-over-bare-id"),
        pytest.param(
            "gemini/*", "gemini/*", "gemini/gemini-partial-route-9*", {_ROUTE_GEMINI_ID}, id="key-filter-over-bare-id"
        ),
        pytest.param(
            "openai/*", "openai/responses/*", None, {f"openai/responses/{_ROUTE_OPENAI_ID}"}, id="responses-route"
        ),
        pytest.param(
            "openai/*",
            "openai/responses/*",
            "openai/gpt-partial-route-9*",
            {f"openai/responses/{_ROUTE_OPENAI_ID}"},
            id="key-filter-over-responses-route",
        ),
        pytest.param(
            "bedrock/*", "bedrock/converse/*", None, {f"bedrock/converse/{_ROUTE_BEDROCK_ID}"}, id="converse-route"
        ),
        pytest.param(
            "openai/partial-route-org/*",
            "openai/partial-route-org/*",
            None,
            set(),
            id="same-name-namespace-the-catalog-lacks",
        ),
        pytest.param(
            "*", "openai/*", "openai/gpt-partial-route-9*", set(), id="provider-prefixed-key-filter-over-catch-all"
        ),
        pytest.param("*", "cloudflare/@cf/*", None, _ROUTE_CF_IDS, id="catch-all-over-partial-deployment"),
        pytest.param("*", "gemini/gemini-partial-route-9*", None, set(), id="catch-all-over-mid-word-deployment"),
        pytest.param("*", "openai/responses/gpt-partial-route-9*", None, set(), id="catch-all-over-mid-word-route"),
        pytest.param(
            "*", "gemini/gemini-partial-route-9-flash*", None, set(), id="catch-all-whose-only-match-is-its-fixed-text"
        ),
    ],
)
def test_every_listed_partial_wildcard_id_routes_to_the_catalog_model_it_names(
    model_name: str, deployment_model: str, key_model: str | None, reachable: set[str]
) -> None:
    from pydantic import TypeAdapter

    from litellm import Router
    from litellm.proxy.auth.model_checks import expand_wildcard_deployments_for_model_info, get_complete_model_list
    from litellm.types.router import LiteLLM_Params

    routed_params: Final = TypeAdapter(tuple[LiteLLM_Params, ...])
    router: Final = Router(model_list=[{"model_name": model_name, "litellm_params": {"model": deployment_model}}])

    def routed_model(listed_id: str) -> str:
        routed: Final = routed_params.validate_python(
            tuple(deployment["litellm_params"] for deployment in router.wildcard_route_deployments(listed_id, None))
        )
        return routed[0].model if routed else listed_id

    def routed_catalog_ids(listed: list[str]) -> list[str]:
        routed_models: Final = [routed_model(listed_id) for listed_id in listed]
        return sorted(
            routed.removeprefix("gemini/") if routed.startswith("gemini/gemini-") else routed
            for routed in routed_models
            if "partial-route" in routed
        )

    with _registered_chat_models(
        {
            _ROUTE_CF_ID: "cloudflare",
            _ROUTE_CF_SUFFIXED_ID: "cloudflare",
            _ROUTE_HF_ID: "cloudflare",
            _ROUTE_GEMINI_ID: "gemini",
            _ROUTE_OPENAI_ID: "openai",
            _ROUTE_BEDROCK_ID: "bedrock",
        }
    ):
        v1_models: Final = get_complete_model_list(
            key_models=[key_model or model_name],
            team_models=[],
            proxy_model_list=[model_name],
            user_model=None,
            infer_model_from_keys=False,
            llm_router=router,
        )
        model_info_names: Final = [
            str(row["model_name"])
            for row in expand_wildcard_deployments_for_model_info(
                [dict(deployment) for deployment in router.get_model_list() or []], router
            )
            if row["model_name"] != model_name
        ]
        listed_v1_models: Final = routed_catalog_ids(v1_models)
        listed_model_info_names: Final = routed_catalog_ids(model_info_names)

    unrouted_ids: Final = [
        listed_id
        for listed_id in [*v1_models, *model_info_names]
        if not router.wildcard_route_deployments(listed_id, None)
    ]
    assert unrouted_ids == []
    assert all(v1_models) and all(model_info_names)
    if model_name == "*":
        assert [listed_id for listed_id in v1_models if not routed_model(listed_id).endswith(f"/{listed_id}")] == []
    assert listed_v1_models == sorted(reachable)
    if key_model is None:
        assert listed_model_info_names == sorted(reachable)


def test_catch_all_lists_no_id_a_more_specific_sibling_wildcard_routes_elsewhere() -> None:
    from litellm import Router
    from litellm.proxy.auth.model_checks import expand_wildcard_deployments_for_model_info, get_complete_model_list

    unshadowed_id: Final = "cloudflare/@cf/unshadowed-route-org/model-d"
    router: Final = Router(
        model_list=[
            {"model_name": "*", "litellm_params": {"model": "cloudflare/@cf/*"}},
            {"model_name": "partial-route-org/*", "litellm_params": {"model": "openai/*"}},
        ]
    )

    def routed_models(listed: list[str]) -> dict[str, object]:
        return {
            listed_id: router.wildcard_route_deployments(listed_id, None)[0]["litellm_params"]["model"]
            for listed_id in listed
            if "route-org/model-" in listed_id
        }

    with _registered_chat_models(
        {_ROUTE_CF_ID: "cloudflare", _ROUTE_CF_SUFFIXED_ID: "cloudflare", unshadowed_id: "cloudflare"}
    ):
        v1_models: Final = get_complete_model_list(
            key_models=[],
            team_models=[],
            proxy_model_list=["*", "partial-route-org/*"],
            user_model=None,
            infer_model_from_keys=False,
            llm_router=router,
        )
        model_info_names: Final = [
            str(row["model_name"])
            for row in expand_wildcard_deployments_for_model_info(
                [dict(deployment) for deployment in router.get_model_list() or []], router
            )
        ]

    assert routed_models(v1_models) == {"unshadowed-route-org/model-d": unshadowed_id}
    assert routed_models(model_info_names) == {"unshadowed-route-org/model-d": unshadowed_id}


def test_wildcard_lists_an_id_a_more_specific_sibling_serves_with_the_same_model() -> None:
    from litellm import Router
    from litellm.proxy.auth.model_checks import expand_wildcard_deployments_for_model_info, get_complete_model_list

    router: Final = Router(
        model_list=[
            {"model_name": "openai/gpt-partial-route-9*", "litellm_params": {"model": "openai/gpt-partial-route-9*"}},
            {
                "model_name": "openai/gpt-partial-route-9-mini*",
                "litellm_params": {"model": "openai/gpt-partial-route-9-mini*"},
            },
        ]
    )
    with _registered_chat_models({_ROUTE_OPENAI_ID: "openai"}):
        v1_models: Final = get_complete_model_list(
            key_models=["openai/gpt-partial-route-9*"],
            team_models=[],
            proxy_model_list=["openai/gpt-partial-route-9*", "openai/gpt-partial-route-9-mini*"],
            user_model=None,
            infer_model_from_keys=False,
            llm_router=router,
        )
        model_info_names: Final = [
            str(row["model_name"])
            for row in expand_wildcard_deployments_for_model_info(
                [dict(deployment) for deployment in router.get_model_list() or []], router
            )
        ]

    assert v1_models == [f"openai/{_ROUTE_OPENAI_ID}"]
    assert "openai/gpt-partial-route-9*" not in model_info_names


def test_prefixed_alias_over_a_bare_catch_all_lists_ids_that_route_through_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm import Router
    from litellm.proxy.auth.model_checks import get_complete_model_list

    monkeypatch.setenv("OPENAI_API_KEY", "sk-partial-route-test")
    router: Final = Router(model_list=[{"model_name": "team-partial-route/*", "litellm_params": {"model": "*"}}])
    with _registered_chat_models({_ROUTE_OPENAI_ID: "openai"}):
        v1_models: Final = get_complete_model_list(
            key_models=[],
            team_models=[],
            proxy_model_list=["team-partial-route/*"],
            user_model=None,
            infer_model_from_keys=False,
            llm_router=router,
        )

    unrouted_ids: Final = [
        listed_id for listed_id in v1_models if not router.wildcard_route_deployments(listed_id, None)
    ]
    assert f"team-partial-route/{_ROUTE_OPENAI_ID}" in v1_models
    assert unrouted_ids == []
