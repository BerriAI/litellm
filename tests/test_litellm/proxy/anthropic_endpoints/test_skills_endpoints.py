from types import SimpleNamespace

from litellm.litellm_core_utils.skill_id_utils import encode_skill_id
from litellm.proxy.anthropic_endpoints.skills_endpoints import _set_skill_route_params
from litellm.skills.main import _get_skill_model


def _request(*, query_params=None, headers=None):
    return SimpleNamespace(query_params=query_params or {}, headers=headers or {})


def test_skill_route_params_reuse_model_source_precedence():
    request = _request(
        query_params={"model": "query-model"},
        headers={"x-litellm-model": "header-model"},
    )

    assert (
        _set_skill_route_params({"model": "body-model"}, request, custom_llm_provider="anthropic")["model"]
        == "body-model"
    )
    assert _set_skill_route_params({}, request, custom_llm_provider="anthropic")["model"] == "query-model"
    assert (
        _set_skill_route_params(
            {}, _request(headers={"x-litellm-model": "header-model"}), custom_llm_provider="anthropic"
        )["model"]
        == "header-model"
    )


def test_skill_route_params_preserves_resource_model_and_reuses_provider_sources():
    skill_id = encode_skill_id("skill_native", "resource-model")
    request = _request(
        query_params={"custom_llm_provider": "query-provider"},
        headers={"custom-llm-provider": "header-provider"},
    )

    data = _set_skill_route_params(
        {"custom_llm_provider": "body-provider"},
        request,
        skill_id=skill_id,
        custom_llm_provider="anthropic",
    )

    assert data["model"] == "resource-model"
    assert data["skill_id"] == "skill_native"
    assert data["custom_llm_provider"] == "header-provider"


def test_skill_response_routing_uses_existing_router_model_metadata():
    assert (
        _get_skill_model({"model": "deployment-model", "litellm_metadata": {"model_group": "model-alias"}})
        == "model-alias"
    )
