"""Live proof that one complexity-router alias can change tiers between tasks while holding its tier inside a tool loop"""

import os
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import (
    ChatAssistantTurn,
    ChatBody,
    ChatMessage,
    ChatMetadata,
    ChatTool,
    ChatToolFunction,
    ChatToolResultTurn,
    ComplexityRouterConfig,
    ComplexityRouterTierMap,
    KeyGenerateBody,
    LiteLLMParamsBody,
    SpendLogRow,
)
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

CHEAP_PROVIDER_MODEL: Final = "together_ai/deepseek-ai/DeepSeek-V4-Flash-0731"
STRONG_PROVIDER_MODEL: Final = "together_ai/deepseek-ai/DeepSeek-V4-Pro-0813"
RESEARCH_ASK: Final = (
    "Think step by step. Research the agent architecture and competing MCP integrations. "
    "First compare the API, database, distributed system, authentication, encryption, "
    "algorithm, function-calling and microservices tradeoffs; then reconcile conflicting "
    "technical documentation and produce an evidence-backed recommendation."
)
EMAIL_ASK: Final = "Write a short follow-up email thanking the customer for today's call."


class TestComplexityRouterAgenticWorkflow:
    @pytest.mark.covers("reliability.routing.complexity_user_turn.routes_agentic_tasks_by_tier")
    def test_research_tool_loop_then_email_reclassifies_between_tiers(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        """One alias routes a research ask to the strong tier, replays that decision for its tool result, then reclassifies a new email ask in the same session onto the cheap tier"""

        marker: Final = unique_marker()
        router_alias: Final = f"e2e-agentic-router-{marker}"
        cheap_tier: Final = f"e2e-agentic-cheap-{marker}"
        strong_tier: Final = f"e2e-agentic-strong-{marker}"
        session_id: Final = f"e2e-agentic-session-{marker}"

        cheap_model_id: Final = proxy.create_model(
            cheap_tier,
            LiteLLMParamsBody(
                model=CHEAP_PROVIDER_MODEL,
                api_key=os.environ.get("TOGETHERAI_API_KEY") or "os.environ/TOGETHERAI_API_KEY",
            ),
        )
        resources.defer(lambda: proxy.delete_model(cheap_model_id))

        strong_model_id: Final = proxy.create_model(
            strong_tier,
            LiteLLMParamsBody(
                model=STRONG_PROVIDER_MODEL,
                api_key=os.environ.get("TOGETHERAI_API_KEY") or "os.environ/TOGETHERAI_API_KEY",
            ),
        )
        resources.defer(lambda: proxy.delete_model(strong_model_id))

        router_config: Final = ComplexityRouterConfig(
            classifier_type="heuristic",
            classification_mode="user_turn",
            session_affinity=False,
            tiers=ComplexityRouterTierMap(
                {
                    "SIMPLE": cheap_tier,
                    "MEDIUM": cheap_tier,
                    "COMPLEX": strong_tier,
                    "REASONING": strong_tier,
                }
            ),
        )
        router_model_id: Final = proxy.create_model(
            router_alias,
            LiteLLMParamsBody(
                model="auto_router/complexity_router",
                complexity_router_config=router_config.model_dump(),
            ),
        )
        resources.defer(lambda: proxy.delete_model(router_model_id))

        router_rows: Final = tuple(row for row in proxy.model_info() if row.model_name == router_alias)
        assert len(router_rows) == 1, f"expected one /model/info row for {router_alias}, got {len(router_rows)}"
        recorded_config: Final = router_rows[0].litellm_params.complexity_router_config
        assert recorded_config is not None, f"/model/info dropped complexity_router_config for {router_alias}"
        assert recorded_config.classifier_type == "heuristic"
        assert recorded_config.classification_mode == "user_turn"
        assert recorded_config.session_affinity is False
        assert recorded_config.tiers.root == router_config.tiers.root

        key: Final = proxy.generate_key(
            KeyGenerateBody(
                models=[router_alias, cheap_tier, strong_tier],
                user_id=f"e2e-agentic-user-{marker}",
            )
        )
        resources.defer(lambda: proxy.delete_key(key))

        search_tool: Final = ChatTool(
            function=ChatToolFunction(
                name="search_docs",
                description="Search technical documentation",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        )
        research_user_turn: Final = ChatMessage(role="user", content=f"{RESEARCH_ASK} Reference {marker}")
        request_metadata: Final = ChatMetadata(session_id=session_id)
        research: Final = unwrap(
            proxy.chat(
                key,
                ChatBody(
                    model=router_alias,
                    messages=(research_user_turn,),
                    metadata=request_metadata,
                    tools=(search_tool,),
                    tool_choice="required",
                    max_tokens=128,
                ),
            )
        )
        assert research.id, f"research completion returned no request id: {research}"
        assert len(research.choices) == 1 and research.choices[0].message is not None
        research_message: Final = research.choices[0].message
        assert research_message.tool_calls is not None and len(research_message.tool_calls) == 1
        search_call: Final = research_message.tool_calls[0]
        assert search_call.id is not None
        assert search_call.function.name == "search_docs"

        research_rows: Final = proxy.poll_logs_for_request_id(research.id, min_rows=1)
        assert len(research_rows) == 1, f"expected one spend row for research, got {len(research_rows)}"
        research_row: Final[SpendLogRow] = research_rows[0]
        assert research_row.model in {
            STRONG_PROVIDER_MODEL,
            "deepseek-ai/DeepSeek-V4-Pro-0813",
            strong_tier,
        }
        assert research_row.metadata is not None and research_row.metadata.routing_decision is not None
        research_decision: Final = research_row.metadata.routing_decision
        assert research_decision.router_model_name == router_alias
        assert research_decision.routed_model == strong_tier
        assert research_decision.tier == "REASONING"
        assert research_decision.cause == "heuristic_scorer"
        assert research_decision.conversation_continuing is False

        research_assistant_turn: Final = ChatAssistantTurn(
            content=research_message.content,
            reasoning_content=research_message.reasoning_content,
            tool_calls=research_message.tool_calls,
        )
        tool_result_turn: Final = ChatToolResultTurn(
            tool_call_id=search_call.id,
            content=f"Evidence {marker}: the architecture comparison and source reconciliation are complete",
        )
        continuation: Final = unwrap(
            proxy.chat(
                key,
                ChatBody(
                    model=router_alias,
                    messages=(research_user_turn, research_assistant_turn, tool_result_turn),
                    metadata=request_metadata,
                    max_tokens=128,
                ),
            )
        )
        assert continuation.id, f"tool continuation returned no request id: {continuation}"
        assert len(continuation.choices) == 1 and continuation.choices[0].message is not None
        continuation_message: Final = continuation.choices[0].message

        continuation_rows: Final = proxy.poll_logs_for_request_id(continuation.id, min_rows=1)
        assert len(continuation_rows) == 1, (
            f"expected one spend row for tool continuation, got {len(continuation_rows)}"
        )
        continuation_row: Final[SpendLogRow] = continuation_rows[0]
        assert continuation_row.model in {
            STRONG_PROVIDER_MODEL,
            "deepseek-ai/DeepSeek-V4-Pro-0813",
            strong_tier,
        }
        assert continuation_row.metadata is not None and continuation_row.metadata.routing_decision is not None
        continuation_decision: Final = continuation_row.metadata.routing_decision
        assert continuation_decision.router_model_name == router_alias
        assert continuation_decision.routed_model == strong_tier
        assert continuation_decision.tier == "REASONING"
        assert continuation_decision.cause == "user_turn_continuation"
        assert continuation_decision.conversation_continuing is True

        completed_research_turn: Final = ChatAssistantTurn(
            content=continuation_message.content or "Research complete",
            reasoning_content=continuation_message.reasoning_content,
            tool_calls=continuation_message.tool_calls,
        )
        email_user_turn: Final = ChatMessage(role="user", content=f"{EMAIL_ASK} Reference {marker}")
        email: Final = unwrap(
            proxy.chat(
                key,
                ChatBody(
                    model=router_alias,
                    messages=(
                        research_user_turn,
                        research_assistant_turn,
                        tool_result_turn,
                        completed_research_turn,
                        email_user_turn,
                    ),
                    metadata=request_metadata,
                    max_tokens=128,
                ),
            )
        )
        assert email.id, f"email completion returned no request id: {email}"
        assert len(email.choices) == 1 and email.choices[0].message is not None
        assert email.choices[0].message.content or email.choices[0].message.reasoning_content

        email_rows: Final = proxy.poll_logs_for_request_id(email.id, min_rows=1)
        assert len(email_rows) == 1, f"expected one spend row for email, got {len(email_rows)}"
        email_row: Final[SpendLogRow] = email_rows[0]
        assert email_row.model in {
            CHEAP_PROVIDER_MODEL,
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            cheap_tier,
        }
        assert email_row.metadata is not None and email_row.metadata.routing_decision is not None
        email_decision: Final = email_row.metadata.routing_decision
        assert email_decision.router_model_name == router_alias
        assert email_decision.routed_model == cheap_tier
        assert email_decision.tier == "SIMPLE"
        assert email_decision.cause == "heuristic_scorer"
        assert email_decision.conversation_continuing is True
