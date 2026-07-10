"""Live e2e: DeepSeek reasoner honors a request to turn reasoning OFF.

DeepSeek's reasoner defaults thinking ON and surfaces the chain as
``message.reasoning_content``. Two documented ways to disable it are
``reasoning_effort="none"`` and ``thinking={"type": "disabled"}``. The DeepSeek
param mapper (``litellm/llms/deepseek/chat/transformation.py``
``map_openai_params``) forwards both as ``thinking={"type": "disabled"}`` so the
outbound body carries a real disable signal and ``deepseek-reasoner`` returns no
``reasoning_content``. This is the behavior tracked by LIT-3686 / GH #27453.

The control case proves the model and path work (reasoning is returned when
nothing asks to disable it), so the two disable assertions are meaningful.

Requires DEEPSEEK_API_KEY on the proxy (tests/e2e/.env). No skip gate: once the
proxy is up, a failure here is real, per the suite's hard-fail contract.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, LiteLLMParamsBody, ThinkingParam
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

REASONER = "deepseek/deepseek-reasoner"
PROMPT = "What is 17 + 26? Answer with just the number."


def _register_reasoner(client: PassthroughClient, resources: ResourceManager) -> str:
    model = f"e2e-deepseek-reasoner-{unique_marker()}"
    model_id = client.gateway.create_model(
        model,
        LiteLLMParamsBody(model=REASONER, api_key="os.environ/DEEPSEEK_API_KEY"),
    )
    resources.defer(lambda: client.gateway.delete_model(model_id))
    return model


def _reasoning_content(response: ChatResponse) -> str | None:
    assert response.choices, f"reasoner returned no choices: {response}"
    message = response.choices[0].message
    assert message is not None, f"reasoner choice has no message: {response}"
    return message.reasoning_content


class TestDeepSeekReasoningDisable:
    def test_reasoner_returns_reasoning_by_default(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = _register_reasoner(client, resources)
        key = resources.key()

        response = unwrap(
            client.gateway.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=PROMPT)],
                    max_tokens=64,
                ),
            )
        )
        reasoning = _reasoning_content(response)
        assert reasoning, (
            "control case: deepseek-reasoner returned no reasoning_content with no "
            f"disable param, so the disable assertions below can't be trusted: {response}"
        )

    def test_reasoning_effort_none_disables_reasoning(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = _register_reasoner(client, resources)
        key = resources.key()

        response = unwrap(
            client.gateway.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=PROMPT)],
                    max_tokens=64,
                    reasoning_effort="none",
                ),
            )
        )
        assert not _reasoning_content(response), (
            "reasoning_effort='none' must disable reasoning, but reasoning_content "
            f"is still present: {response}"
        )

    def test_thinking_disabled_disables_reasoning(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = _register_reasoner(client, resources)
        key = resources.key()

        response = unwrap(
            client.gateway.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=PROMPT)],
                    max_tokens=64,
                    thinking=ThinkingParam(type="disabled"),
                ),
            )
        )
        assert not _reasoning_content(response), (
            "thinking={'type': 'disabled'} must disable reasoning, but "
            f"reasoning_content is still present: {response}"
        )
