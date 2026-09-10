"""Chat completions response, conversation, and validation contracts (LIT-4778).

Exercises the gateway against a live OpenAI deployment using customer request shapes.
"""

from __future__ import annotations

import pytest
from e2e_config import provider_edge_base, unique_marker
from e2e_http import StreamingResponse, assert_client_error, require_successful_call, unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, LiteLLMParamsBody
from proxy_client import ProxyClient
from pydantic import BaseModel

pytestmark = [pytest.mark.e2e, pytest.mark.replayable]

OPENAI_BACKEND = "openai/gpt-4o-mini"
CHAT_PATH = "/chat/completions"


class ChatMissingModelBody(BaseModel):
    messages: list[ChatMessage]


class ChatMissingMessagesBody(BaseModel):
    model: str


class ChatErrorBody(BaseModel):
    message: str | None = None
    type: str | None = None
    code: str | int | None = None


class ChatErrorEnvelope(BaseModel):
    error: ChatErrorBody | None = None


def _register_chat_model(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str]:
    base = provider_edge_base("openai")
    model = f"e2e-chat-sec-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(
            model=OPENAI_BACKEND,
            api_key="os.environ/OPENAI_API_KEY",
            api_base=None if base is None else f"{base}/v1",
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _chat_status(proxy: ProxyClient, key: str, body: BaseModel) -> StreamingResponse:
    return proxy.transport.send(
        CHAT_PATH,
        headers=proxy.transport.bearer(key),
        json=body,
    )


class TestChatCompletionsContract:
    @pytest.mark.covers("llm.chat_completions.openai.multi_turn.nonstream.works")
    def test_multi_turn_history_is_honored(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register_chat_model(proxy, resources)
        turn1 = unwrap(
            proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(role="system", content="You are a helpful math tutor."),
                        ChatMessage(role="user", content="What is 25 + 17? Reply with only the number."),
                    ],
                    temperature=0.1,
                    max_completion_tokens=32,
                ),
            )
        )
        assert turn1.choices and turn1.choices[0].message is not None
        assistant = turn1.choices[0].message.content or ""
        assert "42" in assistant, f"turn1 must answer 42, got: {assistant!r}"

        turn2 = unwrap(
            proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(role="system", content="You are a helpful math tutor."),
                        ChatMessage(role="user", content="What is 25 + 17? Reply with only the number."),
                        ChatMessage(role="assistant", content=assistant),
                        ChatMessage(
                            role="user",
                            content="Now multiply that result by 2. Reply with only the number.",
                        ),
                    ],
                    temperature=0.1,
                    max_completion_tokens=32,
                ),
            )
        )
        assert turn2.choices and turn2.choices[0].message is not None
        second = turn2.choices[0].message.content or ""
        assert "84" in second, f"turn2 must answer 84 from history, got: {second!r}"

    @pytest.mark.covers("llm.chat_completions.openai.basic.nonstream.works")
    def test_success_response_matches_chat_completion_contract(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model, key = _register_chat_model(proxy, resources)
        result = _chat_status(
            proxy,
            key,
            ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=f"Reply with a single word: confirmed. {unique_marker()}")],
                max_completion_tokens=32,
                temperature=0.2,
            ),
        )
        require_successful_call(result)
        parsed = ChatResponse.model_validate_json(result.body)
        assert parsed.id, f"chat completion must return id: {result.body[:300]}"
        assert parsed.object == "chat.completion", f"unexpected object: {parsed.object!r}"
        assert parsed.choices, f"choices must be non-empty: {result.body[:300]}"
        message = parsed.choices[0].message
        assert message is not None, f"choices[0].message required: {result.body[:300]}"
        assert message.role == "assistant", f"unexpected role: {message.role!r}"
        assert (message.content or "").strip(), f"content must be non-empty: {result.body[:300]}"

    @pytest.mark.covers("llm.chat_completions.openai.input_validation.nonstream.works")
    def test_missing_model_returns_client_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        _, key = _register_chat_model(proxy, resources)
        result = _chat_status(
            proxy,
            key,
            ChatMissingModelBody(messages=[ChatMessage(role="user", content="hi")]),
        )
        assert_client_error(result, "missing model")
        envelope = ChatErrorEnvelope.model_validate_json(result.body)
        assert envelope.error is not None and envelope.error.message, (
            f"error body must carry error.message: {result.body[:300]}"
        )

    @pytest.mark.covers("llm.chat_completions.openai.input_validation.nonstream.works")
    def test_missing_messages_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register_chat_model(proxy, resources)
        result = _chat_status(proxy, key, ChatMissingMessagesBody(model=model))
        assert_client_error(result, "missing messages")

    @pytest.mark.covers("llm.chat_completions.openai.input_validation.nonstream.works")
    def test_empty_messages_returns_client_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register_chat_model(proxy, resources)
        result = _chat_status(
            proxy,
            key,
            ChatBody(model=model, messages=[], max_completion_tokens=16),
        )
        assert_client_error(result, "empty messages")

    @pytest.mark.covers("llm.chat_completions.openai.input_validation.nonstream.works")
    def test_invalid_role_returns_client_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register_chat_model(proxy, resources)
        result = _chat_status(
            proxy,
            key,
            ChatBody(
                model=model,
                messages=[ChatMessage(role="invalid_role", content="hi")],
                max_completion_tokens=16,
            ),
        )
        assert_client_error(result, "invalid role")

    @pytest.mark.covers("llm.chat_completions.openai.input_validation.nonstream.works")
    def test_invalid_temperatures_return_client_errors(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register_chat_model(proxy, resources)
        for temperature in (-0.1, 2.1, 3.0, 100.0):
            result = _chat_status(
                proxy,
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content="hi")],
                    temperature=temperature,
                    max_completion_tokens=16,
                ),
            )
            assert_client_error(result, f"temperature={temperature}")

    @pytest.mark.covers("llm.chat_completions.openai.input_validation.nonstream.works")
    def test_invalid_max_completion_tokens_return_client_errors(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model, key = _register_chat_model(proxy, resources)
        for max_completion_tokens in (-100, -1, 0):
            result = _chat_status(
                proxy,
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content="hi")],
                    max_completion_tokens=max_completion_tokens,
                ),
            )
            assert_client_error(result, f"max_completion_tokens={max_completion_tokens}")

    @pytest.mark.covers("llm.chat_completions.openai.basic.nonstream.works")
    def test_temperature_boundaries_succeed(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register_chat_model(proxy, resources)
        for temperature in (0.0, 2.0):
            result = _chat_status(
                proxy,
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=f"Reply with ok. {unique_marker()}")],
                    temperature=temperature,
                    max_completion_tokens=16,
                ),
            )
            require_successful_call(result)
            parsed = ChatResponse.model_validate_json(result.body)
            assert parsed.choices, f"temperature={temperature} must return choices"
