from typing import Final, Literal

import pytest
from fastapi import HTTPException, Request, Response
from starlette.types import Message

from litellm.cost_calculator import response_cost_calculator
from litellm.proxy._types import ProxyException
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.route_llm_request import ProxyMissingRequiredParamError, raise_if_required_body_param_missing
from litellm.proxy.tokenize_endpoints.endpoints import tokenize
from litellm.types.tokenize import TokenizeRequest, TokenizeResponse


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
    request: Final = TokenizeRequest(model="tokenizer", prompt="")
    assert request.prompt == ""


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


@pytest.mark.parametrize(
    "body",
    [
        b"null",
        b"[]",
        b'"hello"',
        b'{"model":"tokenizer"}',
        b'{"prompt":"hello"}',
        b'{"model":"tokenizer","prompt":[]}',
        b'{"model":3,"prompt":"hello"}',
    ],
    ids=["null", "array", "string", "missing-prompt", "missing-model", "prompt-array", "model-number"],
)
async def test_malformed_tokenize_body_returns_client_error(body: bytes) -> None:
    async def receive() -> Message:
        return {"type": "http.request", "body": body}

    request: Final = Request(
        {"type": "http", "method": "POST", "path": "/tokenize", "headers": [(b"content-type", b"application/json")]},
        receive=receive,
    )
    with pytest.raises(HTTPException) as caught:
        await tokenize(request, Response(), UserAPIKeyAuth())
    assert caught.value.status_code == 400
    assert caught.value.detail == "Tokenize request requires a JSON object with string model and prompt fields."


@pytest.mark.parametrize("route", ["/tokenize", "/v1/tokenize"])
async def test_valid_body_reaches_proxy_model_routing(route: str) -> None:
    async def receive() -> Message:
        return {"type": "http.request", "body": b'{"model":"unconfigured-tokenizer","prompt":"Hello"}'}

    request: Final = Request(
        {
            "type": "http",
            "method": "POST",
            "path": route,
            "headers": [(b"content-type", b"application/json")],
            "scheme": "http",
            "server": ("testserver", 80),
            "query_string": b"",
        },
        receive=receive,
    )
    with pytest.raises(ProxyException) as caught:
        await tokenize(request, Response(), UserAPIKeyAuth())
    assert caught.value.code == "400"
    assert caught.value.message == (
        "/tokenize: Invalid model name passed in model=unconfigured-tokenizer. "
        "Call `/v1/models` to view available models for your key."
    )
