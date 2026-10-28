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
        "verbosity": "low",
    }


def test_oversized_snapshot_fails_closed_before_json_validation() -> None:
    assert capture_baseline_parameters({"output_config": {"format": "x" * 5_000_000}}) is None


def test_snapshot_retains_extra_body_settings_but_no_credentials() -> None:
    assert capture_baseline_parameters({"api_key": "private", "extra_body": {"verbosity": "low"}}) == {
        "extra_body": {"verbosity": "low"}
    }


def test_chat_projection_applies_extra_body_after_top_level_parameters() -> None:
    snapshot: Final = capture_baseline_parameters({"verbosity": "high", "extra_body": {"verbosity": "low"}})
    assert snapshot is not None
    assert baseline_request({}, snapshot, {}) == {"verbosity": "low"}


def test_baseline_projection_keeps_caller_tools_and_request_parameter_precedence() -> None:
    from litellm.router import Router

    deployment: Final = {
        "tools": [{"type": "function", "function": {"name": "configured"}}],
        "tool_choice": "required",
        "max_tokens": 64,
    }
    caller: Final = {
        "tools": [{"type": "function", "function": {"name": "caller"}}],
        "tool_choice": "auto",
        "max_tokens": 128,
    }
    actual_request: Final = dict(caller)
    Router._merge_tools_from_deployment({"litellm_params": deployment}, actual_request)
    snapshot: Final = capture_baseline_parameters(caller)
    assert snapshot is not None
    assert baseline_request({"tools": [{"name": "routed-only"}], "max_tokens": 4}, snapshot, deployment) == {
        **deployment,
        **actual_request,
    }
