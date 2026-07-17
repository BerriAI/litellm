"""Live e2e: the v2 auto-router's LLM complexity classifier actually runs over the
proxy and drives routing, instead of silently crashing and falling back to the
local heuristic scorer.

The regression this guards (complexity_router.py `_classifier_call_metadata`
returning None when the request carries no `litellm_metadata`, which the classifier
sub-call then fed into a `.update`, raising `'NoneType' object has no attribute
'update'`) was invisible from the outside: the router caught the error and answered
from heuristic scoring, so every request still returned 200. The only tell is which
tier, and therefore which backend, served the request.

`complexity-smart-router` (see the inline config in docker-compose.yml) pins SIMPLE
to the openai backend and every higher tier to the anthropic backend. "Is P equal
to NP?" is lexically trivial, so the heuristic scorer lands it in SIMPLE (openai),
but any competent LLM classifier reads it as a hard reasoning question and lands it
above SIMPLE (anthropic). The served deployment is read back from the spend log's
`model`, so anthropic proves the classifier ran and openai proves it silently fell
back - the exact failure before the fix.
"""

import pytest

from complexity_router_client import ComplexityRouterClient
from e2e_http import unwrap
from models import ChatBody, ChatMessage

pytestmark = pytest.mark.e2e

ROUTER_MODEL = "complexity-smart-router"
# Lexically simple (heuristic -> SIMPLE) but a hard reasoning question (LLM -> above SIMPLE).
LEXICALLY_SIMPLE_HARD_PROMPT = "Is P equal to NP?"
# SIMPLE tier backend; served only when the classifier silently falls back to heuristic.
# Spend logs may store the alias (gpt-5.5) or the provider-prefixed form depending on
# how the deployment is registered (compose vs /model/new).
HEURISTIC_TIER_MODELS = frozenset({"openai/gpt-5.5", "gpt-5.5"})
# MEDIUM/COMPLEX/REASONING tier backend; served only when the LLM classifier runs.
LLM_TIER_MODELS = frozenset({"anthropic/claude-haiku-4-5", "claude-haiku-4-5"})


class TestComplexityRouterLlmClassifier:
    @pytest.mark.covers("reliability.routing.complexity_llm_classifier.routes_by_llm_tier")
    def test_llm_classifier_runs_and_routes_by_semantic_tier(
        self, client: ComplexityRouterClient, complexity_key: str
    ) -> None:
        chat = unwrap(
            client.gateway.chat(
                complexity_key,
                ChatBody(
                    model=ROUTER_MODEL,
                    messages=[ChatMessage(role="user", content=LEXICALLY_SIMPLE_HARD_PROMPT)],
                    max_tokens=16,
                ),
            )
        )
        assert chat.choices, f"router returned no choices: {chat}"

        rows = client.gateway.poll_logs_for_key(complexity_key, min_rows=1)
        served = [row.model for row in rows]
        assert served and all(model in LLM_TIER_MODELS for model in served), (
            f"expected the request to be served by one of {sorted(LLM_TIER_MODELS)!r} "
            f"(higher-tier backend the LLM classifier picks for a hard prompt), but the "
            f"spend log shows {served!r}. One of {sorted(HEURISTIC_TIER_MODELS)!r} means "
            f"the LLM classifier silently failed or scored SIMPLE (heuristic/fallback path)"
        )
