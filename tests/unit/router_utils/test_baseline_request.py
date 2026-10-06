from typing import Final

from litellm.router_utils.baseline_request import baseline_request, capture_baseline_parameters


def test_baseline_snapshot_owns_nested_caller_settings_and_overrides_routed_settings() -> None:
    reasoning: Final = {"effort": "medium"}
    snapshot: Final = capture_baseline_parameters({"reasoning": reasoning, "verbosity": "low"})
    assert snapshot is not None
    reasoning["effort"] = "high"
    projected: Final = baseline_request(
        {"messages": [{"role": "user", "content": "hello"}], "reasoning": reasoning, "verbosity": "high"},
        snapshot,
        {"verbosity": "medium"},
    )
    assert projected == {
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"effort": "medium"},
        "verbosity": "medium",
    }


def test_oversized_snapshot_fails_closed_before_json_validation() -> None:
    assert capture_baseline_parameters({"output_config": {"format": "x" * 5_000_000}}) is None


def test_snapshot_retains_extra_body_settings_but_no_credentials() -> None:
    assert capture_baseline_parameters({"api_key": "private", "extra_body": {"verbosity": "low"}}) == {
        "verbosity": "low"
    }
