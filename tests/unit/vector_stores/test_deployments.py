from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from litellm.vector_stores.deployments import (
    config_model_list,
    parse_vector_store_deployments,
    vector_store_deployment_problems,
    with_deployment_group,
    with_primary_deployment,
)

EASTUS = {
    "deployment_name": "kb-search",
    "litellm_params": {
        "custom_llm_provider": "azure_ai",
        "api_base": "https://eastus.search.windows.net",
        "api_key": "key-eastus",
        "order": 1,
    },
}
SCUS = {
    "deployment_name": "kb-search",
    "litellm_params": {
        "custom_llm_provider": "azure_ai",
        "api_base": "https://scus.search.windows.net",
        "api_key": "key-scus",
        "order": 2,
    },
}
CHAT_MODEL = {"model_name": "gpt", "litellm_params": {"model": "openai/gpt-5"}}


def _store(**params: object) -> Mapping[str, object]:
    return {
        "vector_store_name": "claims-docs",
        "litellm_params": {
            "vector_store_id": "claims-docs",
            "custom_llm_provider": "azure_ai",
            "vector_store_deployment": "kb-search",
            **params,
        },
    }


def _problems(*stores: Mapping[str, object], model_names: frozenset[str] = frozenset({"gpt"})) -> tuple[str, ...]:
    return vector_store_deployment_problems(
        {
            "model_list": [{"model_name": name, "litellm_params": {"model": "openai/gpt-5"}} for name in model_names],
            "vector_store_deployments": [EASTUS, SCUS],
            "vector_store_registry": list(stores),
        }
    )


def test_config_model_list_keeps_models_and_adds_vector_store_deployments_as_one_group():
    model_list = config_model_list({"model_list": [CHAT_MODEL], "vector_store_deployments": [EASTUS, SCUS]})

    assert model_list[0] is CHAT_MODEL
    assert [
        (d["model_name"], d["litellm_params"]["api_base"], d["litellm_params"]["order"]) for d in model_list[1:]
    ] == [
        ("kb-search", "https://eastus.search.windows.net", 1),
        ("kb-search", "https://scus.search.windows.net", 2),
    ]
    assert {d["litellm_params"]["model"] for d in model_list[1:]} == {"azure_ai/kb-search"}
    assert {d["model_info"]["mode"] for d in model_list[1:]} == {"vector_store"}


def test_config_model_list_without_vector_store_deployments_is_the_model_list():
    assert config_model_list({"model_list": [CHAT_MODEL]}) == [CHAT_MODEL]


@pytest.mark.parametrize(
    "litellm_params",
    [
        {"api_base": "https://eastus.search.windows.net"},
        {"custom_llm_provider": "azure_ai", "model": "azure_ai/x"},
    ],
)
def test_a_deployment_without_provider_or_with_its_own_model_is_rejected(litellm_params: Mapping[str, object]):
    with pytest.raises(ValidationError):
        parse_vector_store_deployments(
            {"vector_store_deployments": [{"deployment_name": "kb-search", "litellm_params": litellm_params}]}
        )


def test_a_store_on_a_defined_group_with_no_connection_settings_is_valid():
    assert (
        _problems(_store(litellm_embedding_model="query-embeddings"), {"litellm_params": {"vector_store_id": "plain"}})
        == ()
    )


@pytest.mark.parametrize(
    ("store", "model_names", "problem"),
    [
        (_store(vector_store_deployment="missing"), frozenset(), "'missing', which is not defined"),
        (_store(), frozenset({"kb-search"}), "already model_list model names"),
        (_store(custom_llm_provider="milvus"), frozenset(), "has custom_llm_provider 'milvus'"),
        (_store(api_base="https://eastus.search.windows.net"), frozenset(), "['api_base']"),
        (_store(litellm_credential_name="cred"), frozenset(), "['litellm_credential_name']"),
        (_store(api_key="k"), frozenset(), "['api_key']"),
    ],
)
def test_a_grouped_store_that_would_bypass_or_miss_its_group_fails_at_load(
    store: Mapping[str, object], model_names: frozenset[str], problem: str
):
    assert any(problem in found for found in _problems(store, model_names=model_names))


def test_a_grouped_store_may_not_repeat_a_setting_its_deployments_define():
    problems = vector_store_deployment_problems(
        {
            "vector_store_deployments": [
                {
                    "deployment_name": "kb-search",
                    "litellm_params": {"custom_llm_provider": "azure_ai", "api_version": "2024-07-01"},
                }
            ],
            "vector_store_registry": [_store(api_version="2025-01-01")],
        }
    )

    assert len(problems) == 1
    assert "['api_version']" in problems[0]


def test_with_deployment_group_routes_a_grouped_store_to_its_group():
    params = with_deployment_group({"vector_store_id": "claims-docs", "vector_store_deployment": "kb-search"})

    assert dict(params) == {"vector_store_id": "claims-docs", "model": "kb-search"}


def test_with_deployment_group_leaves_an_ungrouped_store_alone():
    params = {"vector_store_id": "claims-docs", "api_base": "https://eastus.search.windows.net"}

    assert with_deployment_group(params) is params


def test_with_primary_deployment_uses_the_lowest_order_connection():
    router_deployments = config_model_list({"vector_store_deployments": [SCUS, EASTUS]})

    params = with_primary_deployment(
        {"vector_store_id": "claims-docs", "vector_store_deployment": "kb-search"}, router_deployments
    )

    assert dict(params) == {
        "vector_store_id": "claims-docs",
        "custom_llm_provider": "azure_ai",
        "api_base": "https://eastus.search.windows.net",
        "api_key": "key-eastus",
    }


def test_with_primary_deployment_fails_when_the_group_has_no_deployments():
    with pytest.raises(ValueError, match="'kb-search' has no deployments"):
        with_primary_deployment({"vector_store_deployment": "kb-search"}, ())
