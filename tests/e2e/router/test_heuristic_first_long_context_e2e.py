"""Live e2e repros for heuristic-first classification of short turns with context."""

from __future__ import annotations

from typing import Final

import pytest
from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call
from lifecycle import ResourceManager
from models import (
    ChatAssistantTurn,
    ChatBody,
    ChatMessage,
    ChatToolResultTurn,
    KeyGenerateBody,
    LiteLLMParamsBody,
    ToolCall,
    ToolCallFunction,
)
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

ROUTER_BACKENDS: Final = ("gpt-5.5", "claude-haiku-4-5")
SIMPLE_MODELS: Final = frozenset(("openai/gpt-5.5", "gpt-5.5"))


@pytest.fixture(scope="module")
def heuristic_first_router(proxy: ProxyClient, request: pytest.FixtureRequest) -> str:
    router_name: Final = f"e2e-heuristic-first-router-{unique_marker()}"
    model_id: Final = proxy.create_model(
        router_name,
        LiteLLMParamsBody(
            model="auto_router/complexity_router",
            complexity_router_config={
                "classifier_type": "heuristic_first",
                "heuristic_first_max_tier": "MEDIUM",
                "heuristic_first_max_context_tokens": 8000,
                "classifier_fallback": "heuristic",
                "classifier_llm_config": {"model": "gpt-5.5", "timeout_ms": 30000},
                "tiers": {
                    "SIMPLE": "gpt-5.5",
                    "MEDIUM": "claude-haiku-4-5",
                    "COMPLEX": "claude-haiku-4-5",
                    "REASONING": "claude-haiku-4-5",
                },
            },
        ),
    )
    request.addfinalizer(lambda: proxy.delete_model(model_id))
    return router_name


@pytest.fixture
def heuristic_first_key(
    resources: ResourceManager,
    client: ComplexityRouterClient,
    heuristic_first_router: str,
) -> str:
    key: Final = client.proxy.generate_key(
        KeyGenerateBody(
            models=[heuristic_first_router, *ROUTER_BACKENDS],
            user_id=f"e2e-heuristic-first-{unique_marker()}",
        )
    )
    resources.defer(lambda: client.proxy.delete_key(key))
    return key


def _agentic_messages(marker: str) -> tuple[ChatMessage | ChatAssistantTurn | ChatToolResultTurn, ...]:
    system: Final = ChatMessage(
        role="system",
        content=(
            f"You are a coding agent operating on a repository. Preserve the marker {marker}. "
            "Use tools to inspect files, run tests, and diagnose failures. Keep track of prior "
            "commands and their outputs before proposing a fix. Never discard relevant logs or "
            "assume that a failed command was unrelated to the current change."
        ),
    )
    rounds: Final = tuple(
        turn
        for round_index in range(5)
        for turn in (
            ChatMessage(
                role="user",
                content=(
                    f"Inspect the repository state for debugging round {round_index} using marker {marker}. "
                    "Run the relevant checks and report every warning, traceback, and changed file."
                ),
            ),
            ChatAssistantTurn(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"{marker}-call-{round_index}",
                        type="function",
                        function=ToolCallFunction(
                            name="run_tests",
                            arguments=f'{{"round": {round_index}, "marker": "{marker}"}}',
                        ),
                    )
                ],
            ),
            ChatToolResultTurn(
                tool_call_id=f"{marker}-call-{round_index}",
                content="\n".join(
                    f"{marker} round={round_index} line={line_index} "
                    "synthetic test output records a failing assertion, a retry, a provider "
                    "response, a stack frame, and the captured repository state for diagnosis"
                    for line_index in range(120)
                ),
            ),
        )
    )
    return (system, *rounds, ChatMessage(role="user", content=f"why did that fail? {marker}"))


def _send(
    client: ComplexityRouterClient,
    key: str,
    body: ChatBody,
) -> StreamingResponse:
    response: Final = client.proxy.transport.send(
        "/chat/completions",
        headers=client.proxy.transport.bearer(key),
        json=body,
        stream=False,
    )
    require_successful_call(response)
    return response


def _assert_classifier_consulted(response: StreamingResponse, context: str) -> None:
    assert response.headers.get("x-litellm-complexity-router-cause") == "llm_classifier", (
        f"{context}: expected a successful classifier decision; observed headers={response.headers!r}"
    )
    classifier_cost: Final = response.headers.get("x-litellm-classifier-cost")
    assert classifier_cost is not None, (
        f"{context}: classifier header missing; observed headers={response.headers!r}; body={response.body[:300]!r}"
    )
    try:
        parsed_cost: Final = float(classifier_cost)
    except ValueError as exc:
        raise AssertionError(
            f"{context}: classifier header was not parseable as float: {classifier_cost!r}; "
            f"observed headers={response.headers!r}"
        ) from exc
    assert parsed_cost >= 0, f"{context}: classifier cost was negative: {parsed_cost}; headers={response.headers!r}"


@pytest.mark.covers("reliability.routing.complexity_heuristic.scores_current_ask_only")
class TestHeuristicFirstLongContext:
    def test_short_turn_in_long_agentic_conversation_consults_classifier(
        self,
        client: ComplexityRouterClient,
        heuristic_first_key: str,
        heuristic_first_router: str,
    ) -> None:
        marker: Final = unique_marker()
        response: Final = _send(
            client,
            heuristic_first_key,
            ChatBody(
                model=heuristic_first_router,
                messages=_agentic_messages(marker),
                max_tokens=16,
            ),
        )
        _assert_classifier_consulted(
            response,
            f"long agentic context marker={marker}",
        )

    def test_short_single_turn_stays_on_heuristic_path(
        self,
        client: ComplexityRouterClient,
        heuristic_first_key: str,
        heuristic_first_router: str,
    ) -> None:
        marker: Final = unique_marker()
        response: Final = _send(
            client,
            heuristic_first_key,
            ChatBody(
                model=heuristic_first_router,
                messages=[ChatMessage(role="user", content=f"why did that fail? {marker}")],
                max_tokens=16,
            ),
        )
        assert "x-litellm-classifier-cost" not in response.headers, (
            f"single-turn heuristic path unexpectedly consulted classifier; "
            f"observed headers={response.headers!r}; body={response.body[:300]!r}"
        )
        assert response.headers.get("x-litellm-complexity-router-cause") == "heuristic_first_short_circuit", (
            f"single-turn request should bypass the classifier; observed headers={response.headers!r}"
        )
        rows: Final = client.proxy.poll_logs_for_key(heuristic_first_key, min_rows=1)
        served: Final = tuple(row.model for row in rows if row.model is not None)
        assert len(served) == 1 and served[0] in SIMPLE_MODELS, (
            f"single-turn heuristic path should serve SIMPLE backend {sorted(SIMPLE_MODELS)!r}; "
            f"observed spend-log models={served!r}; headers={response.headers!r}"
        )
