from typing import Final, Literal

import pytest

from litellm.cost_calculator import response_cost_calculator
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.route_llm_request import ProxyMissingRequiredParamError, raise_if_required_body_param_missing
from litellm.types.tokenize import TokenizeResponse


@pytest.mark.parametrize("route", ["/tokenize", "/v1/tokenize"])
def test_tokenization_uses_inference_authorization_without_generation_billing(route: str) -> None:
    assert RouteChecks.is_llm_api_route(route)
    assert RouteChecks.is_unbilled_route(route)


@pytest.mark.parametrize("call_type", ["tokenize", "atokenize"])
def test_tokenization_count_is_not_billed_as_generated_tokens(call_type: Literal["tokenize", "atokenize"]) -> None:
    response: Final = TokenizeResponse(model="acme/model", count=1000, token_ids=[1])
    assert (
        response_cost_calculator(
            response_object=response,
            model=response.model,
            custom_llm_provider="hosted_vllm",
            call_type=call_type,
            optional_params={},
        )
        == 0
    )


def test_missing_prompt_fails_before_routing() -> None:
    with pytest.raises(ProxyMissingRequiredParamError):
        raise_if_required_body_param_missing(route_type="atokenize", data={"model": "tokenizer"}, llm_router=None)


def test_empty_prompt_is_valid() -> None:
    raise_if_required_body_param_missing(
        route_type="atokenize", data={"model": "tokenizer", "prompt": ""}, llm_router=None
    )


def test_provider_validation_receives_unknown_options_without_callback_objects() -> None:
    from litellm.rust_bridge.host.routes.tokenize import body_params

    assert body_params(
        {
            "model": "hosted_vllm/acme/model",
            "prompt": "hello",
            "metadata": {"opaque": object()},
            "timeout": 5,
            "extra_headers": {"x-user": "test"},
            "unknown_option": True,
            "api_version": "2024-03-13",
        }
    ) == {"unknown_option": True, "api_version": "2024-03-13"}
