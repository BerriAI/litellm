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


CACHE_EVIDENCE: Final[Mapping[str, object]] = {
    "config_fingerprint": "config-1",
    "prefix_fingerprint": "prefix-1",
    "provider": "anthropic",
    "provider_model": "claude-test",
    "consecutive_misses": 3,
    "proven_write_tokens": 2048,
    "ttl_seconds": 300,
}
PROMPT_CACHE: Final[Mapping[str, object]] = {
    "type": "prompt_cache",
    "resource_type": "deployment",
    "resource_id": "claude-east",
    "window_start": "2026-01-08T00:00:00Z",
    "window_end": "2026-01-08T00:00:03Z",
    "evidence": CACHE_EVIDENCE,
}


@pytest.mark.parametrize(
    "change",
    (
        {"resource_type": "model"},
        {"evidence": DETECTION["evidence"]},
        {"type": "spend"},
        {"evidence": {**CACHE_EVIDENCE, "consecutive_misses": 0}},
        {"evidence": {**CACHE_EVIDENCE, "provider": "bedrock"}},
        {"evidence": {**CACHE_EVIDENCE, "ttl_seconds": 3600}},
        {"evidence": {**CACHE_EVIDENCE, "prompt": "never retained"}},
    ),
)
def test_prompt_cache_detection_accepts_only_its_deployment_resource_and_supported_evidence(
    change: Mapping[str, object],
) -> None:
    with pytest.raises(ValidationError):
        AlertDetection.model_validate({**PROMPT_CACHE, **change})


def test_prompt_cache_identity_is_one_per_deployment_configuration_whatever_window_found_it() -> None:
    first: Final = AlertDetection.model_validate(PROMPT_CACHE)
    later: Final = AlertDetection.model_validate(
        {
            **PROMPT_CACHE,
            "window_start": "2026-02-08T00:00:00Z",
            "window_end": "2026-02-08T00:00:09Z",
            "evidence": {**CACHE_EVIDENCE, "prefix_fingerprint": "prefix-2", "consecutive_misses": 4},
        }
    )
    upgraded: Final = AlertDetection.model_validate(
        {**PROMPT_CACHE, "evidence": {**CACHE_EVIDENCE, "config_fingerprint": "config-2"}}
    )
    assert first.identity() == later.identity()
    assert first.identity() != upgraded.identity()
