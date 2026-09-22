from typing import Final

import pytest

from litellm.agent_sdk import PRRiskAgent, PRRiskAssessment, PRRiskFailure, PRRiskLevel, PullRequest
from litellm.agent_sdk.client import CompletionProvider
from litellm.agent_sdk.types import CompletionOutcome, CompletionSuccess, ConversationMessage


class RiskProvider(CompletionProvider):
    def __init__(self, response: str) -> None:
        self.response: Final = response
        self.models: tuple[str, ...] = ()
        self.messages: tuple[tuple[ConversationMessage, ...], ...] = ()

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: tuple[ConversationMessage, ...],
        max_tokens: int | None,
    ) -> CompletionOutcome:
        self.models = (*self.models, model)
        self.messages = (*self.messages, messages)
        return CompletionSuccess(content=self.response)


def _pull_request(diff: str) -> PullRequest:
    return PullRequest(
        title="Update request handling",
        body="Adds coverage for the new behavior",
        diff=diff,
        changed_files=1,
        additions=12,
        deletions=3,
    )


@pytest.mark.asyncio
async def test_classifies_pr_and_routes_routine_change_to_fast_model() -> None:
    provider: Final = RiskProvider(
        '{"risk":"low","summary":"Isolated change","reasons":["Small diff"],"recommended_checks":["unit tests"]}'
    )
    agent: Final = PRRiskAgent(
        routine_model="openai/gpt-5.4-mini",
        complex_model="anthropic/claude-opus-4-8",
        completion_provider=provider,
    )

    result: Final = await agent.classify(_pull_request("+ return validated_request"))

    assert result == PRRiskAssessment(
        risk=PRRiskLevel.LOW,
        summary="Isolated change",
        reasons=("Small diff",),
        recommended_checks=("unit tests",),
    )
    assert provider.models == ("openai/gpt-5.4-mini",)


@pytest.mark.asyncio
async def test_routes_sensitive_change_to_complex_model() -> None:
    provider: Final = RiskProvider(
        '```json\n{"risk":"high","summary":"Auth change","reasons":["Permission logic"],'
        '"recommended_checks":["security review"]}\n```'
    )
    agent: Final = PRRiskAgent(
        routine_model="openai/gpt-5.4-mini",
        complex_model="anthropic/claude-opus-4-8",
        completion_provider=provider,
    )

    result: Final = await agent.classify(_pull_request("+ def authorize_admin():"))

    assert isinstance(result, PRRiskAssessment)
    assert result.risk is PRRiskLevel.HIGH
    assert provider.models == ("anthropic/claude-opus-4-8",)


@pytest.mark.asyncio
async def test_invalid_model_output_is_a_typed_failure() -> None:
    provider: Final = RiskProvider("The risk is probably low")
    agent: Final = PRRiskAgent(
        routine_model="openai/gpt-5.4-mini",
        complex_model="anthropic/claude-opus-4-8",
        completion_provider=provider,
    )

    result: Final = await agent.classify(_pull_request("+ docs update"))

    assert isinstance(result, PRRiskFailure)
    assert result.error.startswith("The model returned an invalid risk assessment")


@pytest.mark.asyncio
async def test_each_pr_review_starts_with_fresh_context() -> None:
    provider: Final = RiskProvider(
        '{"risk":"low","summary":"Isolated change","reasons":["Small diff"],"recommended_checks":[]}'
    )
    agent: Final = PRRiskAgent(
        routine_model="openai/gpt-5.4-mini",
        complex_model="anthropic/claude-opus-4-8",
        completion_provider=provider,
    )

    await agent.classify(_pull_request("+ first_change"))
    await agent.classify(_pull_request("+ second_change"))

    assert tuple(len(messages) for messages in provider.messages) == (1, 1)
    assert "first_change" not in provider.messages[1][0].content
