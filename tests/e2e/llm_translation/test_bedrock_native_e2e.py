"""Vendor §9.12: Bedrock native converse/invoke passthrough (LIT-4778).

Model is path-scoped. Happy paths assert assistant-shaped bodies; negatives pin
missing messages and invalid model handling without crashing the proxy.
"""

from __future__ import annotations

import pytest
from typing import Final

from openai.types.chat import ChatCompletionMessage
from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema
from pydantic import BaseModel, ConfigDict

from e2e_config import unique_marker
from e2e_http import (
    assert_client_error,
    require_successful_call,
)
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e

BEDROCK_BACKEND = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"


class CalendarEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    date: str
    participants: tuple[str, ...]


class EventsList(BaseModel):
    model_config = ConfigDict(frozen=True)

    events: tuple[CalendarEvent, ...]


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


class WarehouseAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    answer: int


class JsonModePreambleFields(BaseModel):
    model_config = ConfigDict(frozen=True)

    json_mode_preamble_text: str | None = None


class JsonModeAssistantMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str | None = None
    provider_specific_fields: JsonModePreambleFields | None = None


BEDROCK_JSON_SCHEMA_BACKEND: Final = "bedrock/us.anthropic.claude-opus-5-5"
WAREHOUSE_PROMPT: Final = (
    "A warehouse has 3 shelves. Shelf A holds 17 boxes of 24 items, shelf B holds 9 boxes of 36 items minus 13 "
    "damaged items, and shelf C holds twice as many items as shelf B. How many usable items are there in total? "
    "Work through it carefully, then answer."
)
WAREHOUSE_RESPONSE_FORMAT: Final[ResponseFormatJSONSchema] = {
    "type": "json_schema",
    "json_schema": {
        "name": "warehouse_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}


def _warehouse_message_has_preamble(message: ChatCompletionMessage) -> bool:
    structured_message: Final = JsonModeAssistantMessage.model_validate(message.model_dump())
    assert structured_message.content is not None
    WarehouseAnswer.model_validate_json(structured_message.content)
    provider_fields: Final = structured_message.provider_specific_fields
    preamble: Final = provider_fields.json_mode_preamble_text if provider_fields is not None else None
    return preamble is not None and bool(preamble.strip())


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
    @pytest.mark.provider_live
    @pytest.mark.covers("llm.chat_completions.bedrock_converse.structured_output.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_BACKEND,),
            capabilities=(Capability.RESPONSE_SCHEMA,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_bedrock_chat_completion_pydantic_response(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources)
        response: Final = sdk.openai(key).chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": "Create one event for the Apollo 11 launch in July 1969."}],
            response_format=EventsList,
        )
        parsed: Final = response.choices[0].message.parsed
        assert parsed is not None
        assert parsed.events
        assert all(event.name and event.date and event.participants for event in parsed.events)

    @pytest.mark.provider_live
    @pytest.mark.covers("llm.chat_completions.bedrock_converse.structured_output.preamble.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_JSON_SCHEMA_BACKEND,),
            capabilities=(Capability.RESPONSE_SCHEMA,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_bedrock_json_schema_response_exposes_text_before_json_tool_call(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = f"e2e-bedrock-json-mode-preamble-{unique_marker()}"
        model_id: Final = proxy.create_model(
            model,
            LiteLLMParamsBody(
                model=BEDROCK_JSON_SCHEMA_BACKEND,
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="us-east-1",
            ),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        key: Final = resources.key()

        assert any(
            _warehouse_message_has_preamble(
                sdk.openai(key)
                .chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": WAREHOUSE_PROMPT}],
                    response_format=WAREHOUSE_RESPONSE_FORMAT,
                    max_tokens=1500,
                )
                .choices[0]
                .message
            )
            for _ in range(3)
        ), "No model-generated preamble appeared in three Bedrock calls"

    @pytest.mark.covers("llm.bedrock_native.bedrock_converse.basic.nonstream.works")
    @meta(
        Subject(
            domain=Domain.PASSTHROUGH,
            route=Route.PASSTHROUGH,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
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
    @meta(
        Subject(
            domain=Domain.PASSTHROUGH,
            route=Route.PASSTHROUGH,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_BACKEND,),
            mode=Mode.STREAM,
        )
    )
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
    @meta(
        Subject(
            domain=Domain.PASSTHROUGH,
            route=Route.PASSTHROUGH,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
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
    @meta(
        Subject(
            domain=Domain.PASSTHROUGH,
            route=Route.PASSTHROUGH,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_BACKEND,),
            mode=Mode.STREAM,
        )
    )
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
    @meta(
        Subject(
            domain=Domain.PASSTHROUGH,
            route=Route.PASSTHROUGH,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_converse_missing_messages_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register(proxy, resources)
        result = proxy.transport.send(
            f"/bedrock/model/{model}/converse",
            headers=proxy.transport.bearer(key),
            json=ConverseBody(inferenceConfig=ConverseInferenceConfig()),
        )
        assert_client_error(result, "converse missing messages")

    @pytest.mark.covers("llm.bedrock_native.bedrock_converse.input_validation.nonstream.works")
    @meta(
        Subject(
            domain=Domain.PASSTHROUGH,
            route=Route.PASSTHROUGH,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_converse_empty_messages_returns_client_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register(proxy, resources)
        result = proxy.transport.send(
            f"/bedrock/model/{model}/converse",
            headers=proxy.transport.bearer(key),
            json=ConverseBody(messages=[]),
        )
        assert_client_error(result, "converse empty messages")

    @pytest.mark.covers("llm.bedrock_native.bedrock_converse.input_validation.nonstream.works")
    @meta(
        Subject(
            domain=Domain.PASSTHROUGH,
            route=Route.PASSTHROUGH,
            providers=(Provider.BEDROCK,),
            mode=Mode.NONSTREAM,
        )
    )
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
    @meta(
        Subject(
            domain=Domain.PASSTHROUGH,
            route=Route.PASSTHROUGH,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_invoke_missing_messages_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register(proxy, resources)
        result = proxy.transport.send(
            f"/bedrock/model/{model}/invoke",
            headers=proxy.transport.bearer(key),
            json=InvokeBody(anthropic_version="bedrock-2023-05-31", max_tokens=50),
        )
        assert_client_error(result, "invoke missing messages")

    @pytest.mark.covers("llm.bedrock_native.bedrock_invoke.input_validation.nonstream.works")
    @meta(
        Subject(
            domain=Domain.PASSTHROUGH,
            route=Route.PASSTHROUGH,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
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
    @meta(
        Subject(
            domain=Domain.PASSTHROUGH,
            route=Route.PASSTHROUGH,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
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
