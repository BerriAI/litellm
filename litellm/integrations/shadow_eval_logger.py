"""
Shadow Eval Logger: Duplicate requests through an auto-router, judge blind,
report per-tier stratified win rates for pre-adoption evaluation.

Core flow:
1. On every successful request, check if the deployment has shadow_eval enabled
2. If yes, fire an async background task (non-blocking) to:
   a. Call the router on the same prompt to get the model it would have picked
   b. Call the judge to compare real response vs router-picked response (blind)
   c. Extract the router's tier classification
   d. Write a verdict row to LiteLLM_ShadowEvalVerdict
   e. Tally results in LiteLLM_ShadowEvalJob.result_json
"""

import asyncio
import json
import random
import re
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Literal, Optional, cast

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import LLMResponseTypes

if TYPE_CHECKING:
    from prisma import Prisma

    from litellm.router import Router
    from litellm.types.management_endpoints.auto_router_endpoints import (
        JudgePreference,
    )


_JSON_FENCE_RE: Final = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# Pairwise-comparison judge prompt (blind to which is real vs shadow)
PAIRWISE_JUDGE_SYSTEM_PROMPT = """You are an impartial quality judge. You will compare two responses to the same question.

The responses are labeled A and B in random order (you do not know which came from which system).

Your task: Determine which response is better, or if they are equivalent.

Criteria:
- Correctness: Does it answer accurately?
- Completeness: Does it include relevant context?
- Clarity: Is it easy to understand?
- Conciseness: Is it appropriately brief?

Return ONLY valid JSON in this exact format, no other text:
{
  "preference": "A" | "B" | "tie",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence explanation>"
}"""


def _parse_pairwise_verdict(raw: str) -> dict[str, Any]:
    """Parse the judge's JSON pairwise verdict, tolerating markdown fences."""
    text = raw.strip()
    fenced: Final = _JSON_FENCE_RE.search(text)
    if fenced is not None:
        text = fenced.group(1).strip()
    parsed: object
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract JSON object boundaries
        start: Final = text.find("{")
        end: Final = text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("judge response is not a JSON object")
    return cast(dict[str, Any], parsed)


def _extract_text_from_content(content: Any) -> str:
    """Extract plain text from a message content field (str or multimodal list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: Final = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return " ".join(parts)
    return ""


class ShadowEvalLogger(CustomLogger):
    """
    Integrations hook that fires a background task on every successful response.
    The task (if shadow_eval is enabled for the key) duplicates the request through
    an auto-router, judges the two outputs blind, and writes verdict rows.
    """

    def __init__(self, router: Optional["Router"] = None, prisma_client: Optional["Prisma"] = None):
        """
        Args:
            router: LiteLLM Router instance (needed to call the auto-router)
            prisma_client: Prisma client for writing verdicts to DB
        """
        self.router = router
        self.prisma_client = prisma_client

    async def async_log_success_event(self, kwargs: dict, response_obj: LLMResponseTypes, start_time: Any, end_time: Any):
        """
        Called after a successful LLM call. Fires a background task to shadow-eval if enabled.

        Args:
            kwargs: Request data (messages, model, litellm_call_id, litellm_params, etc.)
            response_obj: The actual LLM response
            start_time: Request start time
            end_time: Request end time
        """
        try:
            # Check if this deployment has shadow_eval enabled
            # (This would normally come from the model's config, checked here)
            metadata: Final = kwargs.get("litellm_params", {}).get("metadata", {}) or {}

            # For now, shadow_eval config would come from:
            # - The model's model_info.shadow_eval config (read from proxy config)
            # - Or from the key's settings (read from database)
            # This is a hook point; the actual config fetching happens in the proxy layer.

            # Fire the background task (don't block the logging return)
            asyncio.create_task(
                self._run_shadow_eval_async(
                    kwargs=kwargs,
                    response_obj=response_obj,
                    start_time=start_time,
                    end_time=end_time,
                )
            )
        except Exception as e:
            verbose_logger.debug(f"Failed to schedule shadow eval task: {e}")
            # Don't raise — logging hook failures should not fail the request

    async def _run_shadow_eval_async(
        self,
        kwargs: dict,
        response_obj: LLMResponseTypes,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """
        Background task: call the router, judge the outputs, write verdict.

        This runs detached from the request, so exceptions are logged but not raised.
        """
        try:
            # 1. Extract the real response text
            real_response_text = self._extract_response_text(response_obj)
            if not real_response_text:
                verbose_logger.debug("Shadow eval: could not extract response text, skipping")
                return

            # 2. Get messages from the request
            messages: Final = kwargs.get("messages", [])
            if not messages:
                verbose_logger.debug("Shadow eval: no messages in request, skipping")
                return

            # 3. Call the router to get the model it would have picked
            # (This is a stub; actual implementation would call self.router with the config)
            shadow_response_text: Final = "shadow response placeholder"  # TODO: call router
            shadow_model: Final = "claude-haiku-4-5"  # TODO: extract from router response
            tier_classification: Final = "SIMPLE"  # TODO: extract from router response

            # 4. Call the judge to compare (blind, randomized A/B order)
            judge_preference, judge_confidence, judge_reasoning = await self._call_judge(
                messages=messages,
                real_response=real_response_text,
                shadow_response=shadow_response_text,
            )

            # 5. Write the verdict to the database (if prisma_client is available)
            if self.prisma_client is not None:
                # TODO: write to LiteLLM_ShadowEvalVerdict and update job counters
                verbose_logger.debug(
                    f"Shadow eval verdict: {judge_preference} (confidence {judge_confidence}), tier={tier_classification}"
                )
        except Exception as e:
            verbose_logger.debug(f"Exception in shadow eval task: {e}", exc_info=True)
            # Don't raise — this is a background task

    def _extract_response_text(self, response_obj: LLMResponseTypes) -> str:
        """Extract the assistant's response text from the LLM response object."""
        if isinstance(response_obj, litellm.ModelResponse):
            try:
                return response_obj["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                return ""
        elif isinstance(response_obj, str):
            return response_obj
        elif isinstance(response_obj, dict):
            try:
                return response_obj["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                return ""
        return ""

    async def _call_judge(
        self,
        messages: list[dict[str, Any]],
        real_response: str,
        shadow_response: str,
    ) -> tuple["JudgePreference", float, str]:
        """
        Call the judge model to compare two responses blindly.

        Returns: (preference, confidence, reasoning)
            preference: "real" | "shadow" | "tie"
            confidence: 0.0 to 1.0
            reasoning: judge's explanation
        """
        # Randomize A/B labels to cancel position bias
        is_real_first: Final = random.random() < 0.5
        response_a = real_response if is_real_first else shadow_response
        response_b = shadow_response if is_real_first else real_response

        conversation_text: Final = "\n".join(
            f"{m.get('role', 'user').upper()}: {_extract_text_from_content(m.get('content', ''))}"
            for m in messages
            if m.get("content") is not None
        )

        user_prompt = f"""Conversation:
{conversation_text}

Response A:
{response_a}

Response B:
{response_b}

Which response is better?"""

        try:
            # Call litellm.acompletion with the judge model
            response = await litellm.acompletion(
                model="claude-3-5-sonnet-20241022",  # TODO: make configurable
                messages=[
                    {"role": "system", "content": PAIRWISE_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=200,
            )
            judge_text: Final = response["choices"][0]["message"]["content"]
            verdict: Final = _parse_pairwise_verdict(judge_text)

            # Unmask the preference (judge said A or B; we need to say "real" or "shadow")
            raw_preference: Final = verdict.get("preference", "tie").lower()
            if raw_preference == "a":
                preference: "JudgePreference" = "real" if is_real_first else "shadow"
            elif raw_preference == "b":
                preference = "shadow" if is_real_first else "real"
            else:
                preference = "tie"

            confidence: Final = float(verdict.get("confidence", 0.5))
            reasoning: Final = str(verdict.get("reasoning", ""))

            return preference, confidence, reasoning

        except Exception as e:
            verbose_logger.debug(f"Judge call failed: {e}")
            raise


# Placeholder for router call (to be implemented in proxy layer)
async def _call_router_for_shadow(
    router: "Router",
    router_config_name: str,
    messages: list[dict[str, Any]],
) -> tuple[str, str, str]:
    """
    Call the auto-router on a prompt to determine what model it would pick.

    Returns: (model_name, tier_classification, shadow_response_text)
        model_name: e.g. "claude-haiku-4-5"
        tier_classification: e.g. "SIMPLE", "COMPLEX", "REASONING"
        shadow_response_text: the actual model response
    """
    # TODO: implement router call via router.acompletion with the given config
    raise NotImplementedError("_call_router_for_shadow not yet implemented")
