"""
LiteLLM + VERONICA: Runtime Budget Enforcement
================================================

veronica-core is a runtime policy enforcement library for LLM calls.
It adds budget caps and kill switches that work alongside LiteLLM's
existing routing and fallback features.

No LiteLLM core changes required -- just register a callback instance.

Install
-------
    pip install litellm veronica-core

Usage
-----
    export OPENAI_API_KEY=sk-...
    python cookbook/community/veronica_budget_enforcement.py
"""

# -- 1. Build a budget enforcer ------------------------------------------------

from veronica_core import BudgetEnforcer, PolicyPipeline
from veronica_core.integrations.litellm import VeronicaLiteLLMCallback

budget = BudgetEnforcer(limit_usd=0.50)          # Hard ceiling: $0.50
pipeline = PolicyPipeline([budget])
callback = VeronicaLiteLLMCallback(pipeline=pipeline)

# -- 2. Register as a LiteLLM callback ----------------------------------------

import litellm

litellm.callbacks = [callback]                    # Zero core changes needed

# -- 3. Make LLM calls -- enforcement is automatic ----------------------------

for i in range(20):
    try:
        resp = litellm.completion(
            model="gpt-4o-mini",                  # Any litellm-supported model
            messages=[{"role": "user", "content": f"Say 'hello {i}'."}],
            max_tokens=10,
        )
        print(
            f"[{i}] {resp.choices[0].message.content.strip()}"
            f"  (total: ${budget.spent_usd:.4f} / ${budget.limit_usd:.2f})"
        )
    except litellm.BudgetExceededError as e:
        # Raised by VERONICA's PolicyPipeline check, not by LiteLLM's
        # built-in proxy budget. The two mechanisms are independent.
        print(f"\n[BLOCKED] {e}")
        break

# -- 4. Final report ----------------------------------------------------------

print(f"\nSpent ${budget.spent_usd:.4f} of ${budget.limit_usd:.2f} budget")
print(f"Calls made: {budget.call_count}")

# -- How it works --------------------------------------------------------------
#
# VeronicaLiteLLMCallback extends litellm's CustomLogger (public API).
#
# On each LLM call:
#   1. log_pre_api_call  -> evaluates PolicyPipeline -> raises if denied
#   2. log_success_event -> reads kwargs["response_cost"] -> BudgetEnforcer.spend()
#      (response_cost may be absent for some providers; treated as $0.00)
#   3. log_failure_event -> CircuitBreaker.record_failure() (if configured)
#
# Proxy mode (LiteLLM Proxy Server):
#   async_pre_call_hook can hard-block requests before they reach the provider.
#
# Only numeric metadata (cost, call count) is recorded.
# No prompts, completions, or PII are logged or stored.
#
# More info: https://github.com/amabito/veronica-core
