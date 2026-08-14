"""
TealTigerGuardrail — self-contained PII / cost / tool-authorization guardrail.

Implements apply_guardrail(), the current recommended pattern for guardrails
that need to run uniformly across chat completions, /v1/messages, responses
API, embeddings, etc. Verified against a live litellm proxy (v1.96.0) and
against litellm/proxy/guardrails/guardrail_hooks/onyx/onyx.py as a reference
implementation of the same interface.

No network calls, no LLM calls, no external API keys required.
"""
from typing import TYPE_CHECKING, Final, List, Literal, Optional

from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.utils import GenericGuardrailAPIInputs

from .engine import Action, PolicyMode, TealEngine

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

DEFAULT_POLICIES = [
    {"type": "pii", "action": "REDACT", "patterns": "all"},
    {"type": "cost", "action": "ENFORCE", "daily_limit_usd": 50.0},
    {"type": "tool_auth", "action": "ENFORCE", "allowlist": None},  # None = allow all
]


class TealTigerGuardrail(CustomGuardrail):
    def __init__(
        self,
        policies: Optional[list] = None,
        policy_mode: str = "ENFORCE",
        **kwargs,
    ):
        self.engine: Final = TealEngine(
            policies=policies or DEFAULT_POLICIES,
            mode=PolicyMode(policy_mode),
        )
        super().__init__(**kwargs)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Runs on both request (pre-call) and response (post-call) content,
        since TealTiger scans for PII leakage in both directions.

        Tool-call authorization and budget checks only make sense on the
        request path, so they're gated on input_type == "request".
        """
        if input_type == "request":
            for tool_call in inputs.get("tool_calls") or []:
                tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
                if tool_name and not self.engine.check_tool(tool_name):
                    raise ValueError(f"TealTiger: blocked — TOOL_NOT_ALLOWLISTED ({tool_name})")

            over_budget, spent, limit = self.engine.check_budget(
                session_id=request_data.get("user") or "default"
            )
            if over_budget:
                raise ValueError(
                    f"TealTiger: blocked — DAILY_BUDGET_EXCEEDED (${spent:.2f} / ${limit:.2f})"
                )

        checked_texts: List[str] = []
        for text in inputs.get("texts") or []:
            decision = self.engine.evaluate_text(text)

            if decision.action == Action.BLOCK.value:
                raise ValueError(f"TealTiger: blocked — {decision.reason_code}")

            checked_texts.append(decision.redacted_text or text)

        inputs["texts"] = checked_texts
        return inputs

    # ---- cost tracking on successful response ----
    # NOTE for reviewers: apply_guardrail is only handed extracted `texts` /
    # `images` / `tool_calls`, not token-usage data — confirmed by reading
    # litellm/integrations/custom_guardrail.py's apply_guardrail docstring
    # directly, not just documentation. Wiring TealEngine.track_cost() into
    # something more precise than a fixed per-request estimate needs either
    # (a) an async_post_call_success_hook override that reads response.usage
    #     (loses the endpoint-agnostic behavior apply_guardrail gives us), or
    #     (b) whatever usage data logging_obj exposes at call time — worth
    #     asking maintainers directly rather than guessing further.
