import importlib.util
import json
import sys
from pathlib import Path
from typing import Final

from litellm.accounting import CostEvent, InboundUsageEvent, OutboundUsageEvent, Unpriced, measurements_for

_SCRIPT: Final = Path(__file__).resolve().parents[3] / "scripts" / "export_accounting_schema.py"
_SPEC: Final = importlib.util.spec_from_file_location("export_accounting_schema", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
export = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = export
_SPEC.loader.exec_module(export)


def test_checked_in_schema_and_fixtures_match_the_models():
    assert export.SCHEMA_PATH.read_text(encoding="utf-8") == export.render_schema()
    assert export.FIXTURES_PATH.read_text(encoding="utf-8") == export.render_fixtures()


def test_schema_lists_every_record_type_and_delivery_target():
    document = json.loads(export.render_schema())
    assert document["records"] == [record.__name__ for record in export.RECORD_TYPES]
    assert all(name in document["$defs"] for name in document["records"])
    assert "LiteLLM_DailyTagSpend" in document["delivery_targets"]
    assert document["$defs"]["CostEvent"]["properties"]["snapshot"]


def test_fixtures_cover_the_semantic_cases_and_round_trip():
    cases = export.fixture_cases()
    assert tuple(cases) == (
        "success",
        "failure",
        "billed_retry",
        "batch",
        "cache_hit",
        "mcp",
        "internal",
        "unknown_price",
    )
    for line in export.render_fixtures().splitlines()[2:]:
        case, record, payload = line.split("\t")
        model = (
            getattr(export, record, None)
            or {
                "InboundUsageEvent": InboundUsageEvent,
                "OutboundUsageEvent": OutboundUsageEvent,
                "CostEvent": CostEvent,
                "Unpriced": Unpriced,
            }[record]
        )
        parsed = model.model_validate_json(payload)
        assert parsed in cases[case]


def test_fixture_identities_follow_the_spend_log_suffix_rules():
    cases = export.fixture_cases()
    retry = [event for event in cases["billed_retry"] if isinstance(event, OutboundUsageEvent)]
    assert retry[1].retry_of_attempt_id == retry[0].attempt_id
    assert (
        sum(len(measurements_for(event)) for event in cases["billed_retry"] if isinstance(event, InboundUsageEvent))
        == 2
    )
    assert all(len(measurements_for(event)) == 1 for event in cases["billed_retry"] if isinstance(event, CostEvent))
    batch = next(event for event in cases["batch"] if isinstance(event, OutboundUsageEvent))
    assert batch.attempt_id.endswith("_batch_cost")
    internal = next(event for event in cases["internal"] if isinstance(event, OutboundUsageEvent))
    assert internal.request_id is None and internal.attempt_id.endswith("_background_cost")
    unpriced = cases["unknown_price"][-1]
    assert isinstance(unpriced, Unpriced) and unpriced.usage_event_id == "outbound:req-unpriced"
