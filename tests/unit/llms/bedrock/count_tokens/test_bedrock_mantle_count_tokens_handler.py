import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest

from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.count_tokens.mantle_handler import BedrockMantleCountTokensHandler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from tests.unit.llms.bedrock.event_loop_probe import host_in_loop_probing_vault

MANTLE_COUNT_URL: Final = "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages/count_tokens"
PRIVATE_MANTLE: Final = "https://vpce-0e1f2a3b.bedrock-mantle.us-east-1.vpce.api.aws"
OPERATOR_MANTLE: Final = "https://vpce-9f8e7d6c.bedrock-mantle.us-east-1.vpce.api.aws"
LITELLM_PARAMS: Final = {
    "aws_access_key_id": "AKIATESTACCESSKEY",
    "aws_secret_access_key": "test-secret",
    "aws_region_name": "us-east-1",
}
REQUEST: Final[dict[str, object]] = {
    "model": "global.anthropic.claude-opus-4-8",
    "messages": [{"role": "user", "content": "The quick brown fox"}],
    "system": "You are a terse assistant.",
    "tools": [{"name": "get_weather", "input_schema": {"type": "object", "properties": {}}}],
}


class _MantleEndpoint:
    def __init__(self, status_code: int, body: Mapping[str, object]) -> None:
        self.status_code: Final = status_code
        self.body: Final = body
        self.requests: tuple[httpx.Request, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        return httpx.Response(self.status_code, json=dict(self.body), request=request)

    def only_request(self) -> httpx.Request:
        assert len(self.requests) == 1, self.requests
        return self.requests[0]


def _client(endpoint: _MantleEndpoint) -> AsyncHTTPHandler:
    return AsyncHTTPHandler(transport=httpx.MockTransport(endpoint))


@pytest.fixture(autouse=True)
def _sigv4_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
    monkeypatch.delenv("AWS_BEDROCK_RUNTIME_ENDPOINT", raising=False)


@pytest.mark.asyncio
async def test_counts_on_mantle_with_the_base_model_and_the_deployment_credentials() -> None:
    mantle: Final = _MantleEndpoint(200, {"input_tokens": 2177})

    result: Final = await BedrockMantleCountTokensHandler().handle_count_tokens_request(
        request_data=dict(REQUEST),
        litellm_params=dict(LITELLM_PARAMS),
        resolved_model="anthropic.claude-opus-4-8",
        client=_client(mantle),
    )

    assert result == {"input_tokens": 2177}
    posted: Final = mantle.only_request()
    assert str(posted.url) == MANTLE_COUNT_URL
    assert posted.headers["Authorization"].startswith("AWS4-HMAC-SHA256")
    assert "us-east-1/bedrock/aws4_request" in posted.headers["Authorization"]
    assert posted.headers["anthropic-version"] == "2023-06-01"
    assert json.loads(posted.content) == {
        "model": "anthropic.claude-opus-4-8",
        "messages": REQUEST["messages"],
        "system": REQUEST["system"],
        "tools": REQUEST["tools"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resolved_model",
    ["anthropic.claude-opus-4-8", "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"],
)
async def test_counts_tools_the_way_mantle_messages_sends_them(resolved_model: str) -> None:
    mantle: Final = _MantleEndpoint(200, {"input_tokens": 11})
    schema: Final = {"type": "object", "properties": {}}

    await BedrockMantleCountTokensHandler().handle_count_tokens_request(
        request_data={
            **REQUEST,
            "tools": [
                {"name": "eager", "input_schema": schema, "custom": {"eager_input_streaming": True}},
                {"name": "deferred", "input_schema": schema, "custom": {"defer_loading": True}},
            ],
        },
        litellm_params=dict(LITELLM_PARAMS),
        resolved_model=resolved_model,
        client=_client(mantle),
    )

    assert json.loads(mantle.only_request().content)["tools"] == [
        {"name": "eager", "input_schema": schema},
        {"name": "deferred", "input_schema": schema, "defer_loading": True},
    ]


@pytest.mark.asyncio
async def test_bedrock_mantle_api_base_env_names_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BEDROCK_MANTLE_API_BASE", "https://vpce-abc.bedrock-mantle.us-east-1.vpce.api.aws")
    mantle: Final = _MantleEndpoint(200, {"input_tokens": 3})

    await BedrockMantleCountTokensHandler().handle_count_tokens_request(
        request_data=dict(REQUEST),
        litellm_params=dict(LITELLM_PARAMS),
        resolved_model="anthropic.claude-opus-4-8",
        client=_client(mantle),
    )

    assert (
        str(mantle.only_request().url)
        == "https://vpce-abc.bedrock-mantle.us-east-1.vpce.api.aws/anthropic/v1/messages/count_tokens"
    )


@pytest.mark.asyncio
async def test_non_200_answers_raise_bedrock_error_with_mantle_status() -> None:
    mantle: Final = _MantleEndpoint(
        404, {"type": "error", "error": {"type": "not_found_error", "message": "does not exist"}}
    )

    with pytest.raises(BedrockError) as raised:
        await BedrockMantleCountTokensHandler().handle_count_tokens_request(
            request_data=dict(REQUEST),
            litellm_params=dict(LITELLM_PARAMS),
            resolved_model="anthropic.claude-sonnet-5-5",
            client=_client(mantle),
        )

    assert raised.value.status_code == 404
    assert "does not exist" in raised.value.message


@pytest.mark.asyncio
async def test_region_and_endpoint_lookups_leave_the_event_loop_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    vault: Final = host_in_loop_probing_vault(
        monkeypatch, "AWS_REGION_NAME", "BEDROCK_MANTLE_API_BASE", "AWS_BEDROCK_RUNTIME_ENDPOINT"
    )
    mantle: Final = _MantleEndpoint(200, {"input_tokens": 2177})

    result = await BedrockMantleCountTokensHandler().handle_count_tokens_request(
        request_data={**REQUEST, "model": "global.anthropic.claude-opus-5-5"},
        litellm_params={key: value for key, value in LITELLM_PARAMS.items() if key != "aws_region_name"},
        resolved_model="anthropic.claude-opus-5-5",
        client=_client(mantle),
    )

    assert vault.read.is_set()
    assert not vault.read_on_loop.is_set()
    assert result == {"input_tokens": 2177}
    assert str(mantle.only_request().url) == MANTLE_COUNT_URL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "environment"),
    (
        ({"api_base": PRIVATE_MANTLE}, {}),
        ({"api_base": f"{PRIVATE_MANTLE}/anthropic/v1/messages"}, {}),
        ({"aws_bedrock_runtime_endpoint": PRIVATE_MANTLE}, {}),
        ({"api_base": PRIVATE_MANTLE}, {"BEDROCK_MANTLE_API_BASE": OPERATOR_MANTLE}),
    ),
    ids=("api_base", "api_base_with_messages_path", "aws_bedrock_runtime_endpoint", "api_base_over_env"),
)
async def test_mantle_deployment_counts_on_the_endpoint_its_inference_uses(
    monkeypatch: pytest.MonkeyPatch, endpoint: Mapping[str, str], environment: Mapping[str, str]
) -> None:
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    mantle: Final = _MantleEndpoint(200, {"input_tokens": 3})

    await BedrockMantleCountTokensHandler().handle_count_tokens_request(
        request_data=dict(REQUEST),
        litellm_params={**LITELLM_PARAMS, "model": "bedrock/mantle/anthropic.claude-opus-5-5", **endpoint},
        resolved_model="anthropic.claude-opus-5-5",
        client=_client(mantle),
    )

    assert str(mantle.only_request().url) == f"{PRIVATE_MANTLE}/anthropic/v1/messages/count_tokens"


@pytest.mark.asyncio
async def test_runtime_deployment_with_its_own_endpoint_is_never_counted_on_public_mantle() -> None:
    mantle: Final = _MantleEndpoint(200, {"input_tokens": 3})

    with pytest.raises(BedrockError) as raised:
        await BedrockMantleCountTokensHandler().handle_count_tokens_request(
            request_data=dict(REQUEST),
            litellm_params={
                **LITELLM_PARAMS,
                "model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "api_base": "https://vpce-0a1b2c3d.bedrock-runtime.us-east-1.vpce.amazonaws.com",
            },
            resolved_model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            client=_client(mantle),
        )

    assert raised.value.status_code == 400
    assert "BEDROCK_MANTLE_API_BASE" in raised.value.message
    assert mantle.requests == ()
