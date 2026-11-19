import pytest

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.proxy.litellm_pre_call_utils import _strip_router_reserved_metadata
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


def test_access_programs_in_supported_openai_params():
    cfg = OpenAIResponsesAPIConfig()
    assert "access_programs" in cfg.get_supported_openai_params("gpt-6.1-terra")


def test_daybreak_disabled_without_deployment_opt_in():
    cfg = OpenAIResponsesAPIConfig()
    out = cfg.transform_responses_api_request(
        model="openai/gpt-6.1-terra",
        input="hi",
        response_api_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert "access_programs" not in out


@pytest.mark.parametrize(
    "model",
    [
        "gpt-6-sol",
        "openai/gpt-6-sol",
        "gpt-6-luna",
        "gpt-6-astra",
        "gpt-6.1-sol",
        "gpt-6.1-terra",
        "openai/gpt-6.1-terra-2026-10-01",
        "gpt-6.2-astra",
        "gpt-6.2-sol",
    ],
)
def test_daybreak_blue_defaults_on_opted_in_deployment(model: str):
    cfg = OpenAIResponsesAPIConfig()
    lp = GenericLiteLLMParams(model_info={"enable_daybreak_access_programs": True})
    out = cfg.transform_responses_api_request(
        model=model,
        input="hi",
        response_api_optional_request_params={},
        litellm_params=lp,
        headers={},
    )
    assert out["access_programs"] == {"cyber": "daybreak_blue"}


def test_daybreak_red_not_enabled_by_default_requires_explicit_mapping():
    cfg = OpenAIResponsesAPIConfig()
    lp_default = GenericLiteLLMParams(
        model_info={"enable_daybreak_access_programs": True}
    )
    assert "access_programs" not in cfg.transform_responses_api_request(
        model="openai/gpt-6-cyber",
        input="hi",
        response_api_optional_request_params={},
        litellm_params=lp_default,
        headers={},
    )

    lp_explicit = GenericLiteLLMParams(
        model_info={"daybreak_models": {"gpt-6-cyber": "daybreak_red"}}
    )
    out_red = cfg.transform_responses_api_request(
        model="openai/gpt-6-cyber",
        input="hi",
        response_api_optional_request_params={},
        litellm_params=lp_explicit,
        headers={},
    )
    assert out_red["access_programs"] == {"cyber": "daybreak_red"}
    assert "access_programs" not in cfg.transform_responses_api_request(
        model="openai/gpt-6-sol",
        input="hi",
        response_api_optional_request_params={},
        litellm_params=lp_explicit,
        headers={},
    )


def test_proxy_ingress_strips_caller_supplied_daybreak_controls():
    client_payload: dict[str, object] = {
        "model": "gpt-6.1-terra",
        "enable_daybreak_access_programs": True,
        "daybreak_models": {"gpt-6.1": "daybreak_red"},
        "model_info": {
            "id": "keep-id",
            "enable_daybreak_access_programs": True,
            "daybreak_models": {"gpt-6.1": "daybreak_red"},
        },
        "metadata": {
            "model_info": {
                "enable_daybreak_access_programs": True,
                "daybreak_models": {"gpt-6.1": "daybreak_red"},
            }
        },
        "litellm_metadata": {
            "model_info": {
                "enable_daybreak_access_programs": True,
                "daybreak_models": {"gpt-6.1": "daybreak_red"},
            }
        },
    }
    _strip_router_reserved_metadata(client_payload)
    assert "enable_daybreak_access_programs" not in client_payload
    assert "daybreak_models" not in client_payload
    assert client_payload["model_info"] == {"id": "keep-id"}
    assert client_payload["metadata"] == {"model_info": {}}
    assert client_payload["litellm_metadata"] == {"model_info": {}}


def test_daybreak_skips_non_daybreak_collisions_and_non_openai():
    cfg = OpenAIResponsesAPIConfig()
    lp = GenericLiteLLMParams(model_info={"enable_daybreak_access_programs": True})
    for non_match in ("gpt-4o", "gpt-6-solar", "o3-mini"):
        assert "access_programs" not in cfg.transform_responses_api_request(
            model=non_match,
            input="hi",
            response_api_optional_request_params={},
            litellm_params=lp,
            headers={},
        )

    cfg_azure = OpenAIResponsesAPIConfig()
    cfg_azure.custom_llm_provider = LlmProviders.AZURE
    assert "access_programs" not in cfg_azure.transform_responses_api_request(
        model="gpt-6.1-terra",
        input="hi",
        response_api_optional_request_params={},
        litellm_params=lp,
        headers={},
    )


def test_daybreak_preserves_caller_override_immutably():
    cfg = OpenAIResponsesAPIConfig()
    lp = GenericLiteLLMParams(model_info={"enable_daybreak_access_programs": True})
    user_ap = {"cyber": "standard"}
    out = cfg.transform_responses_api_request(
        model="gpt-6.1-terra",
        input="hi",
        response_api_optional_request_params={"access_programs": user_ap},
        litellm_params=lp,
        headers={},
    )
    assert out["access_programs"] == {"cyber": "standard"}

    partial_ap = {"other": "val"}
    out2 = cfg.transform_responses_api_request(
        model="gpt-6.1-terra",
        input="hi",
        response_api_optional_request_params={"access_programs": partial_ap},
        litellm_params=lp,
        headers={},
    )
    assert out2["access_programs"] == {"other": "val", "cyber": "daybreak_blue"}
    assert partial_ap == {"other": "val"}
