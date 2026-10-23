from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.tokenize import TokenizeResponse
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import (
    RecordingServer,
    ResponseSpec,
)
from tests.test_litellm_rust.support.routing import assert_served_natively, call_public

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("prefix", ["hosted_vllm", "vllm"])
async def test_vllm_returns_token_ids_and_optional_text(
    server: RecordingServer, asynchronous: bool, prefix: str
) -> None:
    response: Final = await call_public(
        litellm.tokenize,
        litellm.atokenize,
        {
            "model": f"{prefix}/acme/model",
            "prompt": "Hello world",
        },
        server,
        asynchronous=asynchronous,
        metadata={"opaque": object()},
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "acme/model"
    assert response.count == 2
    assert response.token_ids == [101, 102]
    assert response.tokens == ["Hello", " world"]
    assert_served_natively(server, response)
    sent: Final = server.requests[0]
    assert sent.path == "/tokenize"
    assert sent.body == {
        "model": "acme/model",
        "prompt": "Hello world",
        "return_token_strs": True,
        "add_special_tokens": True,
    }
    assert sent.headers["authorization"] == "Bearer test-key"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_watsonx_preserves_token_text_without_inventing_ids(server: RecordingServer, asynchronous: bool) -> None:
    server.default_response = ResponseSpec(body={"result": {"token_count": 2, "tokens": ["Hello", " world"]}})
    response: Final = await call_public(
        litellm.tokenize,
        litellm.atokenize,
        {
            "model": "watsonx/ibm/granite-3-8b-instruct",
            "prompt": "Hello world",
            "project_id": "project",
            "token": "bearer-token",
        },
        server,
        asynchronous=asynchronous,
    )
    assert isinstance(response, TokenizeResponse)
    assert response.count == 2
    assert response.token_ids is None
    assert response.tokens == ["Hello", " world"]
    assert_served_natively(server, response)
    sent: Final = server.requests[0]
    assert sent.path.startswith("/ml/v1/text/tokenization?")
    assert sent.headers["authorization"] == "Bearer bearer-token"
    assert sent.body["input"] == "Hello world"
    assert sent.body["model_id"] == "ibm/granite-3-8b-instruct"
    assert sent.body["project_id"] == "project"
    assert sent.body["parameters"] == {"return_tokens": True}


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


async def test_router_resolves_alias_before_tokenization(
    server: RecordingServer,
) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "tokenizer",
                "litellm_params": {
                    "model": "hosted_vllm/acme/model",
                    "api_base": server.base_url,
                    "api_key": "deployment-key",
                },
            }
        ]
    )
    response: Final = await router.atokenize(model="tokenizer", prompt="Hello world")
    assert isinstance(response, TokenizeResponse)
    assert_served_natively(server, response)
    assert server.requests[0].body["model"] == "acme/model"
    assert server.requests[0].headers["authorization"] == "Bearer deployment-key"


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
