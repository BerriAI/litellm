"""Live e2e: OpenAI-format /chat/completions translated to the Fireworks AI provider.

Registers a fireworks_ai deployment at runtime (deleted on teardown), drives a real
chat completion through the gateway, and asserts a completion came back and its cost
was logged. Streaming is exercised the same way.

This is the cell that would have caught the b3d05bd10b regression: making
FireworksAIConfig inherit FireworksAIMixin first put FireworksAIMixin.validate_environment
ahead of OpenAIGPTConfig.validate_environment in the MRO, and the mixin only sets
Authorization (plus x-session-affinity), never Content-Type. The proxy posts the body
as a pre-serialized string (data=json.dumps(...)), so with no Content-Type header httpx
sends application/octet-stream and Fireworks rejects it with 415 UNSUPPORTED_MEDIA_TYPE.
A mocked unit test bypasses that header/serialization path; only a live call through the
proxy proves it, so the assertion here is a served completion, not a 200 on a fake upstream.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import unwrap
from endpoints_client import EndpointsClient
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody

pytestmark = pytest.mark.e2e

BACKEND_MODEL = "fireworks_ai/accounts/fireworks/models/kimi-k2p6"
FIREWORKS_API_KEY = "os.environ/FIREWORKS_AI_API_KEY"


def _provision(endpoints_client: EndpointsClient, resources: ResourceManager) -> str:
    model_name = f"e2e-fireworks-{unique_marker()}"
    model_id = endpoints_client.create_model(
        model_name,
        LiteLLMParamsBody(model=BACKEND_MODEL, api_key=FIREWORKS_API_KEY),
    )
    resources.defer(lambda: endpoints_client.delete_model(model_id))
    return model_name


class TestFireworksAIChatCompletions:
    @pytest.mark.covers("llm.chat_completions.fireworks_ai.basic.nonstream.works")
    def test_basic_chat_completion_returns_content(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = _provision(endpoints_client, resources)
        key = resources.key()

        chat = unwrap(
            endpoints_client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user", content=f"reply with exactly one word {unique_marker()}"
                        )
                    ],
                    max_tokens=32,
                ),
            )
        )

        content = chat.choices[0].message.content if chat.choices and chat.choices[0].message else None
        assert content and content.strip(), f"fireworks chat returned no content: {chat}"

    @pytest.mark.covers("llm.chat_completions.fireworks_ai.basic.nonstream.cost_logged")
    def test_basic_chat_completion_logs_cost(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = _provision(endpoints_client, resources)
        key = resources.key()

        chat = unwrap(
            endpoints_client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user", content=f"reply with exactly one word {unique_marker()}"
                        )
                    ],
                    max_tokens=32,
                ),
            )
        )
        assert chat.id, f"chat completion carried no id to correlate a spend row: {chat}"

        rows = endpoints_client.proxy.poll_logs_for_request_id(
            chat.id,
            predicate=lambda logged_rows: any((row.spend or 0) > 0 for row in logged_rows),
        )
        row = next((logged_row for logged_row in rows if (logged_row.spend or 0) > 0), None)
        assert row is not None, f"no costed spend row for chat id {chat.id}"
        assert (row.custom_llm_provider or "") == "fireworks_ai", (
            f"spend row provider {row.custom_llm_provider!r} != fireworks_ai"
        )
        assert "kimi-k2p6" in (row.model or ""), f"unexpected spend row model: {row.model}"

    @pytest.mark.covers("llm.chat_completions.fireworks_ai.basic.stream.works")
    def test_streaming_chat_completion_returns_content(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = _provision(endpoints_client, resources)
        key = resources.key()

        result = endpoints_client.proxy.chat_stream(
            key,
            ChatBody(
                model=model,
                messages=[
                    ChatMessage(role="user", content=f"count to three {unique_marker()}")
                ],
                max_tokens=64,
                stream=True,
            ),
        )

        assert result.ok, f"fireworks stream failed (status {result.status_code}): {result.body[:300]}"
        assert result.is_streaming, (
            f"expected an SSE stream from /chat/completions, got content-type {result.content_type!r}"
        )
        assert result.chunks > 0, "no SSE events were consumed from the fireworks stream"
        assert result.stream_error is None, (
            f"the fireworks stream carried an error event: {result.stream_error}"
        )
