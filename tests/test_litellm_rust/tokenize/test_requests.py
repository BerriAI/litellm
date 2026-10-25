import asyncio
from typing import TYPE_CHECKING, Final

import pytest
from pydantic import ValidationError

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.tokenize import TokenizeRequest, TokenizeResponse
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import (
    RecordingServer,
    ResponseSpec,
)
from tests.test_litellm_rust.support.routing import assert_served_natively, call_public

pytestmark = pytest.mark.requires_rust_extension


if TYPE_CHECKING:
    from collections.abc import Coroutine

    from typing_extensions import assert_type

    async def public_tokenize_signature() -> None:
        assert_type(litellm.tokenize("provider/model", "Hello"), litellm.TokenizeResponse)
        assert_type(
            litellm.tokenize(
                "provider/model",
                "Hello",
                api_key="key",
                api_base="https://example.com",
                timeout=1.0,
                custom_llm_provider="hosted_vllm",
                extra_headers={"x-test": "value"},
                include=("tokens",),
                add_special_tokens=False,
                atokenize=False,
                metadata={"purpose": "tokenize"},
                num_retries=0,
                user="caller",
            ),
            litellm.TokenizeResponse,
        )
        legacy: Coroutine[object, object, litellm.TokenizeResponse] = litellm.tokenize(
            "provider/model", "Hello", atokenize=True
        )
        assert_type(await legacy, litellm.TokenizeResponse)
        assert_type(
            await litellm.atokenize(
                model="provider/model",
                prompt="Hello",
                include=("tokens",),
                add_special_tokens=False,
            ),
            litellm.TokenizeResponse,
        )


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("include", "token_ids", "tokens"),
    [
        ((), None, None),
        (("token_ids",), [101, 102], None),
        (("tokens",), None, ["Hello", " world"]),
        (("token_ids", "tokens"), [101, 102], ["Hello", " world"]),
    ],
    ids=["count-only", "token-ids", "tokens", "both"],
)
async def test_vllm_returns_only_requested_representations(
    server: RecordingServer,
    asynchronous: bool,
    include: tuple[str, ...],
    token_ids: list[int] | None,
    tokens: list[str] | None,
) -> None:
    response: Final = await call_public(
        litellm.tokenize,
        litellm.atokenize,
        {
            "model": "hosted_vllm/acme/model",
            "prompt": "Hello world",
            "include": include,
            "user": "opaque-user",
        },
        server,
        asynchronous=asynchronous,
        metadata={"opaque": object()},
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "acme/model"
    assert response.count == 2
    assert response.token_ids == token_ids
    assert response.tokens == tokens
    assert_served_natively(server, response)
    sent: Final = server.requests[0]
    assert sent.path == "/tokenize"
    assert sent.body["model"] == "acme/model"
    assert sent.body["prompt"] == "Hello world"
    assert sent.headers["authorization"] == "Bearer test-key"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("include", "tokens"),
    [((), None), (("tokens",), ["Hello", " world"])],
    ids=["count-only", "tokens"],
)
async def test_watsonx_returns_only_requested_token_text(
    server: RecordingServer,
    asynchronous: bool,
    include: tuple[str, ...],
    tokens: list[str] | None,
) -> None:
    server.default_response = ResponseSpec(body={"result": {"token_count": 2, "tokens": ["Hello", " world"]}})
    response: Final = await call_public(
        litellm.tokenize,
        litellm.atokenize,
        {
            "model": "watsonx/ibm/granite-3-8b-instruct",
            "prompt": "Hello world",
            "include": include,
            "project_id": "project",
            "api_version": "2024-01-01",
            "token": "bearer-token",
        },
        server,
        asynchronous=asynchronous,
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "ibm/granite-3-8b-instruct"
    assert response.count == 2
    assert response.token_ids is None
    assert response.tokens == tokens
    assert_served_natively(server, response)
    sent: Final = server.requests[0]
    assert sent.path.startswith("/ml/v1/text/tokenization?")
    assert "version=2024-01-01" in sent.path
    assert sent.headers["authorization"] == "Bearer bearer-token"
    assert sent.body["input"] == "Hello world"
    assert sent.body["model_id"] == "ibm/granite-3-8b-instruct"
    assert sent.body["project_id"] == "project"


async def test_deployment_hook_rewrites_input_before_native_projection(
    server: RecordingServer,
) -> None:
    class Rewrite(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            assert call_type == "atokenize"
            return {**kwargs, "prompt": "rewritten"}

    with rebound(litellm, "callbacks", [Rewrite()]):
        response: Final = await litellm.atokenize(
            model="hosted_vllm/acme/model",
            prompt="original",
            api_base=server.base_url,
            api_key="test-key",
        )
    assert_served_natively(server, response)
    assert server.requests[0].body["prompt"] == "rewritten"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("include", "token_ids"),
    [((), None), (("token_ids",), [101, 102])],
    ids=["count-only", "token-ids"],
)
async def test_router_resolves_alias_before_tokenization(
    server: RecordingServer,
    asynchronous: bool,
    include: tuple[str, ...],
    token_ids: list[int] | None,
) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "tokenizer",
                "litellm_params": {
                    "model": "hosted_vllm/acme/model",
                    "api_base": server.base_url,
                    "api_key": "deployment-key",
                    "temperature": 0.7,
                    "max_tokens": 128,
                    "add_special_tokens": False,
                },
            }
        ]
    )
    response: Final = (
        await router.atokenize(model="tokenizer", prompt="Hello world", include=include)
        if asynchronous
        else await asyncio.to_thread(router.tokenize, model="tokenizer", prompt="Hello world", include=include)
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "acme/model"
    assert response.token_ids == token_ids
    assert response.tokens is None
    assert_served_natively(server, response)
    assert server.requests[0].body["model"] == "acme/model"
    assert server.requests[0].headers["authorization"] == "Bearer deployment-key"

    assert server.requests[0].body["add_special_tokens"] is False
    assert "temperature" not in server.requests[0].body
    assert "max_tokens" not in server.requests[0].body


async def test_caller_temperature_is_rejected_before_provider_io(server: RecordingServer) -> None:
    server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError):
        await litellm.atokenize(
            model="hosted_vllm/acme/model",
            prompt="Hello",
            api_base=server.base_url,
            api_key="test-key",
            temperature=0.7,
            num_retries=0,
        )
    assert_served_natively(server, count=0)


@pytest.mark.parametrize(
    "include",
    [None, ["unknown"], ["tokens", "tokens"]],
    ids=["null", "unknown-entry", "duplicate"],
)
async def test_invalid_include_is_rejected_before_provider_io(
    server: RecordingServer,
    include: object,
) -> None:
    server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError):
        await litellm.atokenize(
            model="hosted_vllm/acme/model",
            prompt="Hello",
            api_base=server.base_url,
            api_key="test-key",
            include=include,
        )
    assert_served_natively(server, count=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output", "count"),
        ("messages", []),
        ("tools", []),
        ("image", "encoded"),
        ("images", []),
        ("chat_template", "template"),
        ("add_generation_prompt", True),
    ],
)
async def test_unsupported_input_fields_are_rejected_before_provider_io(
    server: RecordingServer,
    field: str,
    value: object,
) -> None:
    server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError):
        await litellm.atokenize(
            model="hosted_vllm/acme/model",
            prompt="Hello",
            api_base=server.base_url,
            api_key="test-key",
            **{field: value},
        )
    assert_served_natively(server, count=0)


async def test_unknown_provider_option_is_rejected_before_provider_io(server: RecordingServer) -> None:
    server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError):
        await litellm.atokenize(
            model="hosted_vllm/acme/model",
            prompt="Hello",
            api_base=server.base_url,
            api_key="test-key",
            unsupported_option=True,
        )
    assert_served_natively(server, count=0)


@pytest.mark.parametrize("add_special_tokens", [True, False], ids=["enabled", "disabled"])
async def test_vllm_honors_explicit_special_token_policy(
    server: RecordingServer,
    add_special_tokens: bool,
) -> None:
    response: Final = await litellm.atokenize(
        model="hosted_vllm/acme/model",
        prompt="Hello",
        api_base=server.base_url,
        api_key="test-key",
        include=["token_ids"],
        add_special_tokens=add_special_tokens,
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "acme/model"
    assert server.requests[0].body["add_special_tokens"] is add_special_tokens
    assert_served_natively(server, response)


@pytest.mark.parametrize("add_special_tokens", [True, False], ids=["enabled", "disabled"])
async def test_watsonx_special_token_policy_is_a_bad_request_before_provider_io(
    server: RecordingServer,
    add_special_tokens: bool,
) -> None:
    server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError) as caught:
        await litellm.atokenize(
            model="watsonx/ibm/granite-3-8b-instruct",
            prompt="Hello",
            api_base=server.base_url,
            api_key="test-key",
            include=["tokens"],
            add_special_tokens=add_special_tokens,
        )
    assert caught.value.status_code == 400
    assert "add_special_tokens" in str(caught.value)
    assert_served_natively(server, count=0)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_router_uses_trusted_watsonx_deployment_options(
    server: RecordingServer,
    asynchronous: bool,
) -> None:
    server.default_response = ResponseSpec(body={"result": {"token_count": 2, "tokens": ["Hello", " world"]}})
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "watsonx-tokenizer",
                "litellm_params": {
                    "model": "watsonx/ibm/granite-3-8b-instruct",
                    "api_base": server.base_url,
                    "token": "deployment-token",
                    "project_id": "deployment-project",
                    "api_version": "2024-03-13",
                },
            }
        ]
    )
    response: Final = (
        await router.atokenize(model="watsonx-tokenizer", prompt="Hello world", include=["tokens"])
        if asynchronous
        else await asyncio.to_thread(
            router.tokenize, model="watsonx-tokenizer", prompt="Hello world", include=["tokens"]
        )
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "ibm/granite-3-8b-instruct"
    assert response.count == 2
    assert response.tokens == ["Hello", " world"]
    assert_served_natively(server, response)
    sent: Final = server.requests[0]
    assert "version=2024-03-13" in sent.path
    assert sent.headers["authorization"] == "Bearer deployment-token"
    assert sent.body["project_id"] == "deployment-project"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", "caller-project"),
        ("api_version", "2025-01-01"),
        ("_tokenize_deployment_params", {"temperature": 0.1}),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_router_rejects_request_sourced_watsonx_fields_before_provider_io(
    server: RecordingServer,
    field: str,
    value: object,
    asynchronous: bool,
) -> None:
    server.expected_requests = 0
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "watsonx-tokenizer",
                "litellm_params": {
                    "model": "watsonx/ibm/granite-3-8b-instruct",
                    "api_base": server.base_url,
                    "api_key": "deployment-key",
                },
            }
        ]
    )
    kwargs: Final = {"model": "watsonx-tokenizer", "prompt": "Hello", field: value}
    expected_error: Final = ValueError if field.startswith("_") else litellm.BadRequestError

    async def invoke_router() -> TokenizeResponse:
        if asynchronous:
            return await router.atokenize(**kwargs)
        return await asyncio.to_thread(router.tokenize, **kwargs)

    with pytest.raises(expected_error):
        await invoke_router()
    assert_served_natively(server, count=0)


async def test_default_include_returns_all_available_representations(server: RecordingServer) -> None:
    response: Final = await litellm.atokenize(
        model="hosted_vllm/acme/model",
        prompt="Hello",
        api_base=server.base_url,
        api_key="test-key",
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "acme/model"
    assert response.count == 2
    assert response.token_ids == [101, 102]
    assert response.tokens == ["Hello", " world"]
    assert_served_natively(server, response)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_watsonx_default_include_returns_available_token_text(
    server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
) -> None:
    monkeypatch.delenv("WATSONX_TOKEN", raising=False)
    monkeypatch.delenv("WATSONX_ZENAPIKEY", raising=False)
    monkeypatch.setenv("WATSONX_IAM_URL", f"{server.base_url}/identity/token")
    server.expected_requests = 2
    server.enqueue(ResponseSpec(body={"access_token": "deployment-token", "expires_in": 3600}))
    server.default_response = ResponseSpec(body={"result": {"token_count": 2, "tokens": ["Hello", " world"]}})
    response: Final = await call_public(
        litellm.tokenize,
        litellm.atokenize,
        {
            "model": "watsonx/ibm/granite-3-8b-instruct",
            "prompt": "Hello world",
            "project_id": "project",
        },
        server,
        asynchronous=asynchronous,
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "ibm/granite-3-8b-instruct"
    assert response.count == 2
    assert response.token_ids is None
    assert response.tokens == ["Hello", " world"]
    assert_served_natively(server, response, count=2)
    assert server.requests[0].path == "/identity/token"
    assert server.requests[1].headers["authorization"] == "Bearer deployment-token"


async def test_empty_prompt_is_forwarded_without_coercion(server: RecordingServer) -> None:
    server.default_response = ResponseSpec(body={"count": 0, "tokens": [], "token_strs": []})
    response: Final = await litellm.atokenize(
        model="hosted_vllm/acme/model",
        prompt="",
        include=[],
        api_base=server.base_url,
        api_key="test-key",
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "acme/model"
    assert response.count == 0
    assert response.token_ids is None
    assert response.tokens is None
    assert server.requests[0].body["prompt"] == ""
    assert_served_natively(server, response)


async def test_upstream_status_remains_a_public_error(server: RecordingServer) -> None:
    server.default_response = ResponseSpec(status=429, body={"error": "rate limited"})
    with pytest.raises(litellm.RateLimitError) as caught:
        await litellm.atokenize(
            model="hosted_vllm/acme/model",
            prompt="Hello",
            api_base=server.base_url,
            api_key="test-key",
            num_retries=0,
        )
    assert caught.value.status_code == 429
    assert_served_natively(server)


def test_omitted_special_token_setting_round_trips_as_omitted() -> None:
    request: Final = TokenizeRequest(model="provider/model", prompt="Hello")

    assert "add_special_tokens" not in request.model_dump()
    assert TokenizeRequest.model_validate_json(request.model_dump_json()) == request


@pytest.mark.parametrize("add_special_tokens", [False, True], ids=["disabled", "enabled"])
def test_explicit_special_token_setting_round_trips(add_special_tokens: bool) -> None:
    request: Final = TokenizeRequest(
        model="provider/model",
        prompt="Hello",
        add_special_tokens=add_special_tokens,
    )

    assert request.model_dump()["add_special_tokens"] is add_special_tokens
    assert TokenizeRequest.model_validate_json(request.model_dump_json()) == request


def test_explicit_null_special_token_setting_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TokenizeRequest.model_validate_json(
            '{"model":"provider/model","prompt":"Hello","add_special_tokens":null}'
        )
