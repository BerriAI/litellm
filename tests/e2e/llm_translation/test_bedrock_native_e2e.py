"""Vendor §9.12: Bedrock native converse/invoke passthrough (LIT-4778).

Model is path-scoped. Happy paths assert assistant-shaped bodies; negatives pin
missing messages and invalid model handling without crashing the proxy.
"""

from __future__ import annotations

import pytest
from e2e_config import unique_marker
from e2e_http import (
    assert_client_error,
    require_successful_call,
)
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from pydantic import BaseModel

pytestmark = pytest.mark.e2e

BEDROCK_BACKEND = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"


class ConverseContent(BaseModel):
    text: str


class ConverseMessage(BaseModel):
    role: str
    content: list[ConverseContent]


class ConverseInferenceConfig(BaseModel):
    maxTokens: int = 50
    temperature: float = 0.5


class ConverseBody(BaseModel):
    messages: list[ConverseMessage] | None = None
    system: list[ConverseContent] | None = None
    inferenceConfig: ConverseInferenceConfig | None = None


class InvokeBody(BaseModel):
    anthropic_version: str | None = None
    messages: list[InvokeMessage] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    system: str | None = None


class InvokeMessage(BaseModel):
    role: str
    content: str


class ConverseOutput(BaseModel):
    message: ConverseMessage


class ConverseResponse(BaseModel):
    output: ConverseOutput


class InvokeResponse(BaseModel):
    content: list[ConverseContent]


def _register(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str]:
    model = f"e2e-bedrock-native-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(
            model=BEDROCK_BACKEND,
            aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
            aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
            aws_region_name="os.environ/AWS_REGION",
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _default_converse() -> ConverseBody:
    return ConverseBody(
        messages=[ConverseMessage(role="user", content=[ConverseContent(text="Hello")])],
        inferenceConfig=ConverseInferenceConfig(),
    )


def _default_invoke() -> InvokeBody:
    return InvokeBody(
        anthropic_version="bedrock-2023-05-31",
        messages=[InvokeMessage(role="user", content="Hello")],
        max_tokens=50,
        temperature=0.7,
    )


class TestBedrockNative:
    @pytest.mark.covers("llm.bedrock_native.bedrock_converse.basic.nonstream.works")
    def test_converse_returns_assistant(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register(proxy, resources)
        result = proxy.transport.send(
            f"/bedrock/model/{model}/converse",
            headers=proxy.transport.bearer(key),
            json=_default_converse(),
        )
        require_successful_call(result)
        response = ConverseResponse.model_validate_json(result.body)
        assert response.output.message.role == "assistant"
        assert any(part.text.strip() for part in response.output.message.content)

    @pytest.mark.covers("llm.bedrock_native.bedrock_converse.basic.stream.works")
    def test_converse_stream_returns_chunks(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register(proxy, resources)
        result = proxy.transport.send(
            f"/bedrock/model/{model}/converse-stream",
            headers=proxy.transport.bearer(key),
            json=_default_converse(),
            stream=True,
        )
        require_successful_call(result)
        assert result.stream_error is None, result.stream_error
        assert result.chunks > 0, "converse-stream returned no events"

    @pytest.mark.covers("llm.bedrock_native.bedrock_invoke.basic.nonstream.works")
    def test_invoke_returns_message(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register(proxy, resources)
        result = proxy.transport.send(
            f"/bedrock/model/{model}/invoke",
            headers=proxy.transport.bearer(key),
            json=_default_invoke(),
        )
        require_successful_call(result)
        response = InvokeResponse.model_validate_json(result.body)
        assert any(part.text.strip() for part in response.content)

    @pytest.mark.covers("llm.bedrock_native.bedrock_invoke.basic.stream.works")
    def test_invoke_stream_returns_chunks(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register(proxy, resources)
        result = proxy.transport.send(
            f"/bedrock/model/{model}/invoke-with-response-stream",
            headers=proxy.transport.bearer(key),
            json=_default_invoke(),
            stream=True,
        )
        require_successful_call(result)
        assert result.stream_error is None, result.stream_error
        assert result.chunks > 0, "invoke stream returned no events"

    @pytest.mark.covers("llm.bedrock_native.bedrock_converse.input_validation.nonstream.works")
    def test_converse_missing_messages_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register(proxy, resources)
        result = proxy.transport.send(
            f"/bedrock/model/{model}/converse",
            headers=proxy.transport.bearer(key),
            json=ConverseBody(inferenceConfig=ConverseInferenceConfig()),
        )
        assert_client_error(result, "converse missing messages")

    @pytest.mark.covers("llm.bedrock_native.bedrock_converse.input_validation.nonstream.works")
    def test_converse_empty_messages_returns_client_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register(proxy, resources)
        result = proxy.transport.send(
            f"/bedrock/model/{model}/converse",
            headers=proxy.transport.bearer(key),
            json=ConverseBody(messages=[]),
        )
        assert_client_error(result, "converse empty messages")

    @pytest.mark.covers("llm.bedrock_native.bedrock_converse.input_validation.nonstream.works")
    def test_converse_invalid_model_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        _, key = _register(proxy, resources)
        result = proxy.transport.send(
            "/bedrock/model/does-not-exist/converse",
            headers=proxy.transport.bearer(key),
            json=_default_converse(),
        )
        assert result.status_code in (400, 404), (
            f"invalid model expected 400/404, got {result.status_code}: {result.body[:300]}"
        )

    @pytest.mark.covers("llm.bedrock_native.bedrock_invoke.input_validation.nonstream.works")
    def test_invoke_missing_messages_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register(proxy, resources)
        result = proxy.transport.send(
            f"/bedrock/model/{model}/invoke",
            headers=proxy.transport.bearer(key),
            json=InvokeBody(anthropic_version="bedrock-2023-05-31", max_tokens=50),
        )
        assert_client_error(result, "invoke missing messages")

    @pytest.mark.covers("llm.bedrock_native.bedrock_invoke.input_validation.nonstream.works")
    def test_invoke_missing_max_tokens_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register(proxy, resources)
        result = proxy.transport.send(
            f"/bedrock/model/{model}/invoke",
            headers=proxy.transport.bearer(key),
            json=InvokeBody(
                anthropic_version="bedrock-2023-05-31",
                messages=[InvokeMessage(role="user", content="Hello")],
            ),
        )
        assert_client_error(result, "invoke missing max_tokens")

    @pytest.mark.covers("llm.bedrock_native.bedrock_invoke.input_validation.nonstream.works")
    def test_invoke_invalid_temperature_returns_client_error(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model, key = _register(proxy, resources)
        result = proxy.transport.send(
            f"/bedrock/model/{model}/invoke",
            headers=proxy.transport.bearer(key),
            json=InvokeBody(
                anthropic_version="bedrock-2023-05-31",
                messages=[InvokeMessage(role="user", content="Hello")],
                max_tokens=50,
                temperature=5.0,
            ),
        )
        assert_client_error(result, "invoke invalid temperature")
