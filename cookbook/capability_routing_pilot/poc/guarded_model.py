import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path

import litellm
from minisweagent.exceptions import LimitsExceeded
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

COST_HEADER = "llm_provider-x-litellm-response-cost"


class GuardedModelConfig(LitellmModelConfig):
    run_budget_usd: float
    worst_input_usd_per_token: float
    output_usd_per_token: float
    max_output_tokens: int
    input_token_safety_factor: float = 1.25
    fixed_overhead_tokens: int = 1024
    call_log_path: str


def reserve_usd(input_tokens: int, cfg: GuardedModelConfig) -> float:
    return (
        input_tokens * cfg.input_token_safety_factor * cfg.worst_input_usd_per_token
        + cfg.max_output_tokens * cfg.output_usd_per_token
    )


def estimate_input_tokens(counted: int, prev_counted: int | None, prev_actual: int | None,
                          cfg: GuardedModelConfig) -> int:
    floor = counted + cfg.fixed_overhead_tokens
    if prev_counted is None or prev_actual is None:
        return floor
    return max(floor, prev_actual + max(counted - prev_counted, 0))


def budget_allows(spent_usd: float, reservation_usd: float, budget_usd: float) -> bool:
    return spent_usd + reservation_usd <= budget_usd


def gateway_cost(hidden_params: Mapping[str, object]) -> float | None:
    headers = hidden_params.get("additional_headers")
    raw = headers.get(COST_HEADER) if isinstance(headers, Mapping) else None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError, OverflowError):
        return None


def valid_cost(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value)) and value >= 0
    except OverflowError:
        return False


def _cost_accounting_error() -> LimitsExceeded:
    return LimitsExceeded(
        {"role": "exit", "content": "CostAccountingError", "extra": {
            "exit_status": "CostAccountingError", "submission": ""
        }}
    )


class GuardedLitellmModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=GuardedModelConfig, **kwargs)
        self.spent_usd = 0.0
        self.accounting_incomplete = False
        self._pending_reservation = 0.0
        self._pending_input_estimate = 0
        self._pending_counted = 0
        self._prev_counted: int | None = None
        self._prev_actual: int | None = None

    def query(self, messages: list[dict], **kwargs) -> dict:
        if self.accounting_incomplete:
            self._log({"event": "accounting_incomplete_refused", "exit_status": "CostAccountingError"})
            raise _cost_accounting_error()
        os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"
        prepared = self._prepare_messages_for_api(messages)
        n_in = litellm.token_counter(
            model=self.config.model_name.removeprefix("litellm_proxy/"), messages=prepared, tools=[BASH_TOOL]
        )
        n_est = estimate_input_tokens(n_in, self._prev_counted, self._prev_actual, self.config)
        reservation = reserve_usd(n_est, self.config)
        if not budget_allows(self.spent_usd, reservation, self.config.run_budget_usd):
            self._log({"event": "budget_guard_refused", "spent_usd": self.spent_usd, "reservation_usd": reservation,
                       "counted_input_tokens": n_in, "est_input_tokens": n_est})
            raise LimitsExceeded(
                {"role": "exit", "content": "BudgetGuard", "extra": {"exit_status": "BudgetGuard", "submission": ""}}
            )
        self._pending_reservation = reservation
        self._pending_input_estimate = n_est
        self._pending_counted = n_in
        return super().query(messages, **kwargs)

    def _query(self, messages, **kwargs):
        return super()._query(
            messages,
            **(
                kwargs
                | {
                    "max_tokens": self.config.max_output_tokens,
                    "timeout": 120.0,
                    "num_retries": 0,
                }
            ),
        )

    def _calculate_cost(self, response) -> dict[str, float]:
        hidden_params = response._hidden_params
        headers = hidden_params.get("additional_headers")
        raw_cost = headers.get(COST_HEADER) if isinstance(headers, Mapping) else None
        cost = gateway_cost(hidden_params)
        cost_valid = valid_cost(cost)
        if not cost_valid:
            self.accounting_incomplete = True
        else:
            self.spent_usd += cost
        usage = response.usage.model_dump() if response.usage else {}
        self._prev_counted = self._pending_counted
        self._prev_actual = usage.get("prompt_tokens")
        self._log({
            "event": "call",
            "call_id": hidden_params.get("litellm_call_id"),
            "response_id": response.id,
            "returned_model": response.model,
            "gateway_model_id": (headers or {}).get("llm_provider-x-litellm-model-id")
            if isinstance(headers, Mapping) else None,
            "cost_usd": cost if cost_valid else raw_cost,
            "cost_source": "gateway_header" if cost_valid else "invalid",
            "reservation_usd": self._pending_reservation,
            "est_input_tokens": self._pending_input_estimate,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "latency_ms": hidden_params.get("_response_ms"),
            "finish_reason": response.choices[0].finish_reason,
            "spent_usd_after": self.spent_usd,
            "reservation_covered_actual": cost_valid and cost <= self._pending_reservation,
        })
        if not cost_valid:
            raise _cost_accounting_error()
        return {"cost": cost}

    def _log(self, row: dict) -> None:
        with Path(self.config.call_log_path).open("a") as f:
            f.write(json.dumps({"ts": time.time(), **row}) + "\n")
