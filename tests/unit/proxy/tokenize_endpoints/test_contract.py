from typing import Final, Literal

import pytest
from fastapi import HTTPException, Request, Response
from pydantic import ValidationError
from starlette.types import Message

from litellm.cost_calculator import response_cost_calculator
from litellm.proxy._types import ProxyException
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.route_llm_request import ProxyMissingRequiredParamError, raise_if_required_body_param_missing
from litellm.proxy.tokenize_endpoints.endpoints import tokenize
from litellm.tokenize.routing import deployment_kwargs, deployment_params, source_for
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
    assert request.include == ()
    assert request.add_special_tokens is None


def test_tokenize_request_schema_matches_runtime_contract() -> None:
    schema: Final = TokenizeRequest.model_json_schema()
    assert schema["properties"]["include"]["uniqueItems"] is True
    assert schema["properties"]["add_special_tokens"]["type"] == "boolean"
    assert "default" not in schema["properties"]["add_special_tokens"]
    assert "add_special_tokens" not in schema["required"]


@pytest.mark.parametrize(
    ("include", "expected"),
    [
        ([], ()),
        (["token_ids"], ("token_ids",)),
        (["tokens"], ("tokens",)),
        (["token_ids", "tokens"], ("token_ids", "tokens")),
    ],
    ids=["count-only", "token-ids", "tokens", "both"],
)
def test_tokenize_request_normalizes_include_arrays_and_serializes_them(
    include: list[str], expected: tuple[str, ...]
) -> None:
    request: Final = TokenizeRequest.model_validate({"model": "tokenizer", "prompt": "", "include": include})
    assert request.include == expected
    assert request.model_dump(mode="json")["include"] == include


def test_tokenize_request_rejects_unknown_fields_as_extra_forbidden() -> None:
    with pytest.raises(ValidationError) as caught:
        TokenizeRequest.model_validate({"model": "tokenizer", "prompt": "", "unsupported_option": True})
    assert caught.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize(
    "include",
    [None, "token_ids", ["unknown"], ["tokens", "tokens"], ["token_ids", None]],
    ids=["null", "scalar", "unknown-entry", "duplicate", "null-entry"],
)
def test_include_accepts_only_unique_supported_representations(include: object) -> None:
    with pytest.raises(ValidationError):
        TokenizeRequest.model_validate({"model": "tokenizer", "prompt": "", "include": include})


@pytest.mark.parametrize("add_special_tokens", [None, 1, "true"], ids=["null", "integer", "string"])
def test_explicit_add_special_tokens_requires_boolean(add_special_tokens: object) -> None:
    with pytest.raises(ValidationError):
        TokenizeRequest.model_validate({"model": "tokenizer", "prompt": "", "add_special_tokens": add_special_tokens})


def test_provider_validation_receives_optional_options_without_callback_objects() -> None:
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
            "include": ["token_ids"],
            "add_special_tokens": False,
        }
    ) == {"unknown_option": True, "api_version": "2024-03-13"}


def test_provider_validation_keeps_optional_params_but_drops_callback_metadata() -> None:
    from litellm.rust_bridge.host.routes.tokenize import body_params

    assert body_params(
        {
            "temperature": 0.7,
            "max_tokens": 128,
            "user": "proxy-user",
            "metadata": {"opaque": object()},
            "include": ["token_ids"],
            "add_special_tokens": False,
            "_tokenize_context": object(),
        }
    ) == {"temperature": 0.7, "max_tokens": 128}


def test_router_deployment_defaults_are_separate_from_request_provenance() -> None:
    from litellm.rust_bridge.host.routes.tokenize import body_params

    deployment: Final = {
        "model": "hosted_vllm/acme/model",
        "api_base": "http://deployment",
        "temperature": 0.7,
        "max_tokens": 128,
        "extra_body": {"deployment_option": True},
        "add_special_tokens": False,
    }
    effective_request: Final = {
        **deployment,
        "base_url": "http://deployment",
    }
    routed: Final = deployment_kwargs(
        deployment,
        effective_request,
        caller_fields=frozenset({"model", "base_url"}),
    )
    assert "temperature" not in routed
    assert "max_tokens" not in routed
    assert "extra_body" not in routed
    assert routed["add_special_tokens"] is False
    assert "add_special_tokens" not in body_params(routed)
    assert deployment_params(routed) == {
        "temperature": 0.7,
        "max_tokens": 128,
        "extra_body": {"deployment_option": True},
    }
    assert source_for(routed, "add_special_tokens") == "deployment"
    assert source_for(routed, "api_base") == "request"


def test_router_keeps_caller_option_request_sourced_when_equal_to_default() -> None:
    deployment: Final = {"model": "hosted_vllm/acme/model", "temperature": 0.7}
    request: Final = {"model": "hosted_vllm/acme/model", "temperature": 0.7}
    routed: Final = deployment_kwargs(
        deployment,
        request,
        caller_fields=frozenset({"model", "temperature"}),
    )
    assert routed["temperature"] == deployment["temperature"]
    assert source_for(routed, "temperature") == "request"


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
        b'{"model":"","prompt":""}',
        b'{"model":"tokenizer","prompt":"hello","include":null}',
        b'{"model":"tokenizer","prompt":"hello","include":["unsupported"]}',
        b'{"model":"tokenizer","prompt":"hello","include":["tokens","tokens"]}',
        b'{"model":"tokenizer","prompt":"hello","add_special_tokens":null}',
        b'{"model":"tokenizer","prompt":"hello","output":"count"}',
        b'{"model":"tokenizer","prompt":"hello","backend":{"type":"local","config":{}}}',
        b'{"model":"tokenizer","prompt":"hello","artifact_path":"/tmp/tokenizer.json"}',
        b'{"model":"tokenizer","prompt":"hello","_tokenize_context":{}}',
        b'{"model":"tokenizer","prompt":"hello","messages":[]}',
        b'{"model":"tokenizer","prompt":"hello","tools":[]}',
        b'{"model":"tokenizer","prompt":"hello","images":[]}',
        b'{"model":"tokenizer","prompt":"hello","chat_template":"template"}',
        b'{"model":"tokenizer","prompt":"hello","add_generation_prompt":true}',
        b'{"model":"tokenizer","prompt":"hello","metadata":{"nested":[{"backend":{"type":"local"}}]}}',
        b'{"model":"tokenizer","prompt":"hello","metadata":{"nested":[{"_tokenize_context":{}}]}}',
    ],
    ids=[
        "null",
        "array",
        "string",
        "missing-prompt",
        "missing-model",
        "prompt-array",
        "model-number",
        "empty-model",
        "include-null",
        "include-unknown",
        "include-duplicate",
        "special-tokens-null",
        "legacy-output",
        "client-local-backend",
        "client-artifact-path",
        "forged-tokenize-context",
        "messages",
        "tools",
        "images",
        "chat-template",
        "generation-prompt",
        "nested-local-backend",
        "nested-internal-context",
    ],
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
