from collections.abc import Mapping
from datetime import timedelta, timezone
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.models.alert import AlertDetection

DETECTION: Final = {
    "type": "spend",
    "resource_type": "team",
    "resource_id": "example-team",
    "window_start": "2026-01-08T00:00:00Z",
    "window_end": "2026-01-09T00:00:00Z",
    "evidence": {"value": 20, "baseline": 2, "unit": "usd", "sample_size": 50},
}


@pytest.mark.parametrize(
    "change",
    (
        {"resource_id": "x" * 1025},
        {"resource_id": ""},
        {"resource_type": "key", "resource_id": "sk-example-not-a-real-key"},
        {"resource_type": "gateway", "resource_id": "another-gateway"},
        {
            "resource_type": "gateway",
            "resource_id": "gateway",
            "type": "model_deprecation",
            "evidence": {"deprecation_date": "2026-02-01"},
        },
        {"window_start": "2026-01-08T00:00:00"},
        {"window_start": "2026-01-08T00:00:00.123456Z", "window_end": "2026-01-08T00:00:00.123457Z"},
        {"window_end": "2026-01-08T00:00:00Z"},
        {"window_end": "2026-01-07T00:00:00Z"},
        {"type": "custom"},
        {"type": "latency"},
        {"type": "model_deprecation"},
        {"evidence": {"value": float("inf"), "baseline": 2, "unit": "usd", "sample_size": 1}},
        {"evidence": {"value": 2, "baseline": float("nan"), "unit": "usd", "sample_size": 1}},
        {"evidence": {"value": -1, "baseline": 2, "unit": "usd", "sample_size": 1}},
        {"evidence": {"value": 2, "baseline": 2, "unit": "usd", "sample_size": -1}},
        {"evidence": {"value": 2, "baseline": 2, "unit": "usd", "sample_size": 1, "unbounded": "extra"}},
        {"type": "reliability", "evidence": {"value": 1.1, "baseline": 0, "unit": "ratio", "sample_size": 1}},
        {"type": "reliability", "evidence": {"value": 0, "baseline": 1.1, "unit": "ratio", "sample_size": 1}},
        {
            "resource_type": "model",
            "type": "model_deprecation",
            "evidence": {"deprecation_date": "2026-02-01", "replacement_model": "x" * 1025},
        },
    ),
)
def test_invalid_or_unbounded_detection_is_rejected(change: Mapping[str, object]) -> None:
    with pytest.raises(ValidationError):
        AlertDetection.model_validate({**DETECTION, **change})


def test_identity_uses_instants_not_timezone_or_evidence() -> None:
    original: Final = AlertDetection.model_validate(DETECTION)
    revised: Final = original.model_copy(
        update={
            "window_start": original.window_start.astimezone(timezone(timedelta(hours=8))),
            "window_end": original.window_end.astimezone(timezone(timedelta(hours=8))),
            "evidence": original.evidence.model_copy(update={"value": 30}),
        }
    )
    assert original.identity() == revised.identity()
