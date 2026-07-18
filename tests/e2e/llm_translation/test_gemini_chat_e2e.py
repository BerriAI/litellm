"""Live e2e: Gemini via the OpenAI-compatible /chat/completions path.

Registers a fresh gemini/gemini-2.5-flash deployment through /model/new (deleted
on teardown), drives a real completion with a virtual key, and asserts a usable
answer plus a costed SpendLogs row. Complements the native /gemini passthrough
suite by covering the translation path customers use when they keep the OpenAI
SDK.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

BACKEND_MODEL = "gemini/gemini-2.5-flash"


class TestGeminiChatCompletions:
    @pytest.mark.covers(
        "llm.chat_completions.gemini.basic.nonstream.works",
        "llm.chat_completions.gemini.basic.nonstream.cost_logged",
        exercised_on=["chat_completions"],
    )
    def test_gemini_chat_returns_content_and_logs_cost(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-gemini-chat-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=BACKEND_MODEL, api_key="os.environ/GEMINI_API_KEY"),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()
        tag = f"e2e-gemini-chat-{unique_marker()}"

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"Reply with the single word pong. marker={tag}",
                        )
                    ],
                    max_tokens=32,
                ),
            )
        )
        assert response.choices, f"gemini chat returned no choices: {response}"
        content = response.choices[0].message.content if response.choices[0].message else None
        assert content, f"gemini chat returned empty content: {response}"

        rows = client.proxy.poll_logs_for_key(
            key,
            min_rows=1,
            predicate=lambda rs: any((r.spend or 0) > 0 for r in rs),
        )
        assert rows, f"no SpendLogs row for gemini chat on key ending ...{key[-6:]}"
        row = rows[0]
        assert (row.spend or 0) > 0, f"gemini chat was not costed: {row}"
        assert row.status == "success", f"gemini chat spend status={row.status!r}"
