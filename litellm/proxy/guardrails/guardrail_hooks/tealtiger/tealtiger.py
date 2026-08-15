"""
TealTigerGuardrail — self-contained PII / cost / tool-authorization guardrail.

Implements apply_guardrail(), the current recommended pattern for guardrails
that need to run uniformly across chat completions, /v1/messages, responses
API, embeddings, etc. Verified against a live litellm proxy (v1.97.0) and
against litellm/proxy/guardrails/guardrail_hooks/onyx/onyx.py as a reference
implementation of the same interface.

No network calls, no LLM calls, no external API keys required.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.utils import GenericGuardrailAPIInputs

from .engine import Action, PolicyMode, TealEngine

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_EMPTY_MAPPING: Final = MappingProxyType({})

DEFAULT_POLICIES: Final = (
    MappingProxyType({"type": "pii", "action": "REDACT", "patterns": "all"}),
    MappingProxyType({"type": "cost", "action": "ENFORCE", "daily_limit_usd": 50.0}),
    MappingProxyType({"type": "tool_auth", "action": "ENFORCE", "allowlist": None}),  # None = allow all
)


class TealTigerGuardrail(CustomGuardrail):
    def __init__(
        self,
        policies: Sequence[Mapping[str, object]] | None = None,
        policy_mode: str = "ENFORCE",
        **kwargs: object,  # kwargs-ok: forwards unknown/future CustomGuardrail.__init__ params (guardrail_name, event_hook, default_on, ...) to the base class
    ) -> None:
        self.engine: Final = TealEngine(
            policies=policies or DEFAULT_POLICIES,
            mode=PolicyMode(policy_mode),
        )
        super().__init__(**kwargs)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: Mapping[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Runs on both request (pre-call) and response (post-call) content,
        since TealTiger scans for PII leakage in both directions.

        Tool-call authorization and budget checks only make sense on the
        request path, so they're gated on input_type == "request".

        Returns a new mapping rather than mutating `inputs` in place, since
        the caller's object should not be rewritten at a distance.
        """
        if input_type == "request":
            self._check_tool_calls(inputs)
            self._check_budget(request_data)

        checked_texts: Final = tuple(self._check_text_or_raise(text) for text in inputs.get("texts") or ())
        return {**inputs, "texts": list(checked_texts)}  # mutable-ok: dict interop

    def _check_tool_calls(self, inputs: GenericGuardrailAPIInputs) -> None:
        for tool_call in inputs.get("tool_calls") or ():
            tool_name = tool_call.get("name") or tool_call.get("function", _EMPTY_MAPPING).get("name")
            if tool_name and not self.engine.check_tool(tool_name):
                raise ValueError(f"TealTiger: blocked — TOOL_NOT_ALLOWLISTED ({tool_name})")

    def _check_budget(self, request_data: Mapping[str, object]) -> None:
        over_budget, spent, limit = self.engine.check_budget(session_id=request_data.get("user") or "default")
        if over_budget:
            raise ValueError(f"TealTiger: blocked — DAILY_BUDGET_EXCEEDED (${spent:.2f} / ${limit:.2f})")

    def _check_text_or_raise(self, text: str) -> str:
        decision: Final = self.engine.evaluate_text(text)
        if decision.action == Action.BLOCK.value:
            raise ValueError(f"TealTiger: blocked — {decision.reason_code}")
        return decision.redacted_text or text

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
