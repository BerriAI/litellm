"""Export the accounting contracts for other implementations (the Rust cost crate first).

Writes two files next to the Rust cost crate's reference fixtures:
- ``accounting_schema.json``: JSON Schema for every public accounting record type.
- ``accounting_fixtures.tsv``: one semantic case per line (success, failure, billed retry,
  batch, cache hit, MCP, internal call, unknown price) as validated events in wire form.

``--check`` exits non-zero when the checked-in files differ from the models.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Final

from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from litellm.accounting import (
    AdmissionDecision,
    AggregateQuery,
    AggregateRow,
    CostEvent,
    DeliveryState,
    DeliveryTarget,
    Dimension,
    DimensionName,
    EventKind,
    InboundUsageEvent,
    Measurement,
    Metric,
    MetricName,
    OutboundUsageEvent,
    Outcome,
    PricingSnapshot,
    Rate,
    Unpriced,
)
from litellm.types.utils import CallTypes

OUTPUT_DIR: Final = Path(__file__).resolve().parents[1] / "litellm-rust" / "crates" / "cost" / "tests"
SCHEMA_PATH: Final = OUTPUT_DIR / "accounting_schema.json"
FIXTURES_PATH: Final = OUTPUT_DIR / "accounting_fixtures.tsv"
RECORD_TYPES: Final[tuple[type[BaseModel], ...]] = (
    InboundUsageEvent,
    OutboundUsageEvent,
    CostEvent,
    Unpriced,
    Measurement,
    AggregateQuery,
    AggregateRow,
    AdmissionDecision,
)

_START: Final = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
_END: Final = _START + timedelta(seconds=2)
_KEY: Final = Dimension(name=DimensionName.KEY, value="sk-hash")
_TEAM: Final = Dimension(name=DimensionName.TEAM, value="team-1")
_MODEL: Final = Dimension(name=DimensionName.MODEL, value="gpt-4o")
_PROVIDER: Final = Dimension(name=DimensionName.PROVIDER, value="openai")
_CALLER: Final = (_KEY, _TEAM)
_ATTEMPT: Final = (_KEY, _TEAM, _MODEL, _PROVIDER)


def _count(name: MetricName, value: int) -> Metric:
    return Metric(name=name, value=Decimal(value))


def _spend(value: str) -> Metric:
    return Metric(name=MetricName.SPEND, value=Decimal(value), unit="USD")


def _inbound(request_id: str, outcome: Outcome, call_type: CallTypes = CallTypes.acompletion) -> InboundUsageEvent:
    outcome_metric: Final = (
        _count(MetricName.SUCCESSFUL_REQUESTS, 1)
        if outcome is Outcome.SUCCEEDED
        else _count(MetricName.FAILED_REQUESTS, 1)
    )
    return InboundUsageEvent(
        event_id=f"inbound:{request_id}",
        request_id=request_id,
        call_type=call_type,
        started_at=_START,
        ended_at=_END,
        outcome=outcome,
        dimensions=_CALLER,
        metrics=(_count(MetricName.REQUESTS, 1), outcome_metric),
    )


def _outbound(
    attempt_id: str,
    request_id: str | None,
    outcome: Outcome,
    metrics: tuple[Metric, ...] | None,
    *,
    retry_of: str | None = None,
    call_type: CallTypes = CallTypes.acompletion,
    dimensions: tuple[Dimension, ...] = _ATTEMPT,
) -> OutboundUsageEvent:
    return OutboundUsageEvent(
        event_id=f"outbound:{attempt_id}",
        attempt_id=attempt_id,
        request_id=request_id,
        retry_of_attempt_id=retry_of,
        call_type=call_type,
        started_at=_START,
        ended_at=_END,
        outcome=outcome,
        dimensions=dimensions,
        metrics=metrics,
    )


def _tokens(prompt: int, completion: int, cache_read: int = 0) -> tuple[Metric, ...]:
    usage = [
        _count(MetricName.PROMPT_TOKENS, prompt),
        _count(MetricName.COMPLETION_TOKENS, completion),
        _count(MetricName.TOTAL_TOKENS, prompt + completion),
    ]
    if cache_read:
        usage.append(_count(MetricName.CACHE_READ_INPUT_TOKENS, cache_read))
    return tuple(usage)


def _cost(attempt_id: str, total: str, usage: tuple[Metric, ...], **snapshot: object) -> CostEvent:
    return CostEvent(
        event_id=f"cost:{attempt_id}",
        usage_event_id=f"outbound:{attempt_id}",
        occurred_at=_END,
        dimensions=_ATTEMPT,
        charges=(_spend(total),),
        snapshot=PricingSnapshot.model_validate(
            {
                "calculator_version": "fixture",
                "model": "gpt-4o",
                "provider": "openai",
                "pricing_source": "cost_map",
                "priced_model_key": "openai/gpt-4o",
                "rates": (
                    Rate(token_type="input", usd_per_token=Decimal("0.0000025")),
                    Rate(token_type="output", usd_per_token=Decimal("0.00001")),
                ),
                "usage": tuple(metric for metric in usage if metric.name is not MetricName.TOTAL_TOKENS),
                "total_cost": Decimal(total),
                **snapshot,
            }
        ),
    )


def fixture_cases() -> dict[str, tuple[BaseModel, ...]]:
    success_usage: Final = _tokens(100, 20)
    retry_usage: Final = _tokens(100, 20)
    cache_usage: Final = _tokens(100, 20, cache_read=80)
    return {
        "success": (
            _inbound("req-success", Outcome.SUCCEEDED),
            _outbound("req-success", "req-success", Outcome.SUCCEEDED, success_usage),
            _cost(
                "req-success", "0.00045", success_usage, input_cost=Decimal("0.00025"), output_cost=Decimal("0.0002")
            ),
        ),
        "failure": (
            _inbound("req-failure", Outcome.FAILED),
            _outbound("req-failure", "req-failure", Outcome.FAILED, ()),
        ),
        "billed_retry": (
            _inbound("req-retry", Outcome.SUCCEEDED),
            _outbound("req-retry-attempt-0", "req-retry", Outcome.FAILED, _tokens(100, 5)),
            _cost("req-retry-attempt-0", "0.0003", _tokens(100, 5)),
            _outbound(
                "req-retry-attempt-1", "req-retry", Outcome.SUCCEEDED, retry_usage, retry_of="req-retry-attempt-0"
            ),
            _cost("req-retry-attempt-1", "0.00045", retry_usage),
        ),
        "batch": (
            _inbound("req-batch", Outcome.SUCCEEDED, call_type=CallTypes.acreate_batch),
            _outbound(
                "req-batch_batch_cost",
                "req-batch",
                Outcome.SUCCEEDED,
                _tokens(5000, 1000),
                call_type=CallTypes.acreate_batch,
            ),
            _cost("req-batch_batch_cost", "0.0225", _tokens(5000, 1000)),
        ),
        "cache_hit": (
            _inbound("req-cache", Outcome.SUCCEEDED),
            _outbound("req-cache", "req-cache", Outcome.SUCCEEDED, cache_usage),
            _cost(
                "req-cache",
                "0.00031",
                cache_usage,
                rates=(
                    Rate(token_type="input", usd_per_token=Decimal("0.0000025")),
                    Rate(token_type="output", usd_per_token=Decimal("0.00001")),
                    Rate(token_type="cache_read", usd_per_token=Decimal("0.00000125")),
                ),
            ),
        ),
        "mcp": (
            _inbound("req-mcp", Outcome.SUCCEEDED, call_type=CallTypes.call_mcp_tool),
            _outbound(
                "req-mcp",
                "req-mcp",
                Outcome.SUCCEEDED,
                (),
                call_type=CallTypes.call_mcp_tool,
                dimensions=(_KEY, _TEAM, Dimension(name=DimensionName.TOOL, value="github/search_issues")),
            ),
        ),
        "internal": (
            _outbound(
                "req-internal_background_cost",
                None,
                Outcome.SUCCEEDED,
                success_usage,
                call_type=CallTypes.aembedding,
            ),
            _cost("req-internal_background_cost", "0.00045", success_usage),
        ),
        "unknown_price": (
            _inbound("req-unpriced", Outcome.SUCCEEDED),
            _outbound("req-unpriced", "req-unpriced", Outcome.SUCCEEDED, success_usage),
            Unpriced(usage_event_id="outbound:req-unpriced", reason="model_not_in_cost_map"),
        ),
    }


def render_schema() -> str:
    _, definitions = models_json_schema(
        [(record, "validation") for record in RECORD_TYPES], ref_template="#/$defs/{model}"
    )
    document: Final = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "litellm.accounting",
        "schema_version": 1,
        "records": [record.__name__ for record in RECORD_TYPES],
        "event_kinds": [kind.value for kind in EventKind],
        "delivery_states": [state.value for state in DeliveryState],
        "delivery_targets": [target.value for target in DeliveryTarget],
        **definitions,
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def render_fixtures() -> str:
    lines: Final = [
        "# Accounting contract fixtures. Regenerate with: python scripts/export_accounting_schema.py",
        "case\trecord\tjson",
    ]
    for case, records in fixture_cases().items():
        for record in records:
            lines.append(f"{case}\t{type(record).__name__}\t{record.model_dump_json()}")
    return "\n".join(lines) + "\n"


class _Options(argparse.Namespace):
    check: bool = False


def main() -> int:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when the checked-in files are stale")
    options: Final = parser.parse_args(namespace=_Options())
    check: Final = options.check
    outputs: Final = {SCHEMA_PATH: render_schema(), FIXTURES_PATH: render_fixtures()}
    stale: Final = tuple(
        path for path, content in outputs.items() if not path.exists() or path.read_text(encoding="utf-8") != content
    )
    if check:
        for path in stale:
            sys.stderr.write(f"stale: {path}\n")
        return 1 if stale else 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
        sys.stdout.write(f"wrote {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
