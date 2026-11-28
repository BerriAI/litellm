from collections.abc import Mapping
from typing import Final

from litellm.litellm_core_utils.agentic_followup_kwargs import build_agentic_followup_kwargs


def _build(
    *,
    request_kwargs: dict[str, object],
    patch_kwargs: dict[str, object],
    request_params: set[str],
    fingerprints: list[str] | None = None,
) -> Mapping[str, object]:
    return build_agentic_followup_kwargs(
        request_kwargs=request_kwargs,
        patch_kwargs=patch_kwargs,
        request_params=request_params,
        depth=0,
        max_loops=3,
        fingerprints=fingerprints if fingerprints is not None else [],
        fingerprint="fp",
    )


def test_followup_kwargs_never_repeat_a_request_param():
    """Neither source may re-add a key the caller already sends as a request param, or the follow-up call raises a duplicate keyword"""
    followup: Final = _build(
        request_kwargs={"prompt_cache_key": "thread-1", "api_base": "https://a"},
        patch_kwargs={"prompt_cache_key": "thread-1", "metadata": {"user": "u1"}},
        request_params={"prompt_cache_key", "model", "input"},
    )

    assert followup.keys().isdisjoint({"prompt_cache_key", "model", "input"})
    assert followup["api_base"] == "https://a"
    assert followup["metadata"] == {"user": "u1"}


def test_followup_kwargs_let_the_plan_override_the_request():
    followup: Final = _build(
        request_kwargs={"api_base": "https://request", "timeout": 5},
        patch_kwargs={"api_base": "https://plan"},
        request_params=set(),
    )

    assert followup["api_base"] == "https://plan"
    assert followup["timeout"] == 5


def test_followup_kwargs_carry_the_loop_bookkeeping_without_touching_the_inputs():
    fingerprints: Final = ["earlier"]
    request_kwargs: Final = {"_agentic_loop_depth": 0, "max_agentic_loops": 9}
    patch_kwargs: Final = {"_agentic_loop_fingerprints": ["stale"]}

    followup: Final = _build(
        request_kwargs=request_kwargs,
        patch_kwargs=patch_kwargs,
        request_params=set(),
        fingerprints=fingerprints,
    )

    assert followup["_agentic_loop_depth"] == 1
    assert followup["max_agentic_loops"] == 3
    assert followup["_agentic_loop_fingerprints"] == ["earlier", "fp"]
    assert fingerprints == ["earlier"]
    assert request_kwargs == {"_agentic_loop_depth": 0, "max_agentic_loops": 9}
    assert patch_kwargs == {"_agentic_loop_fingerprints": ["stale"]}


def test_followup_kwargs_copies_metadata_and_isolates_otel_internal():
    """Metadata must be copied so mutations by the follow-up do not bleed back into the caller,
    and internal OTel dedupe state (_otel_internal) must be stripped from the child call."""
    parent_meta = {
        "user_id": "usr_123",
        "custom_tag": "test",
        "_otel_internal": {"spans_logged": {("litellm_request", "success"): True}},
    }
    followup = _build(
        request_kwargs={"metadata": parent_meta},
        patch_kwargs={},
        request_params=set(),
    )

    child_meta = followup["metadata"]
    assert child_meta is not parent_meta
    assert child_meta == {"user_id": "usr_123", "custom_tag": "test"}
    assert "_otel_internal" not in child_meta

    # Mutating child metadata must not affect parent
    child_meta["_otel_internal"] = {"spans_logged": {("followup", "success"): True}}
    assert "_otel_internal" in parent_meta
    assert ("followup", "success") not in parent_meta["_otel_internal"]["spans_logged"]


def test_followup_kwargs_copies_litellm_params_metadata():
    """When metadata is nested inside litellm_params, it should also be copied and cleaned."""
    parent_params = {
        "api_key": "sk-123",
        "metadata": {
            "session_id": "sess_abc",
            "_otel_internal": {"spans_logged": {}},
        },
    }
    followup = _build(
        request_kwargs={"litellm_params": parent_params},
        patch_kwargs={},
        request_params=set(),
    )

    child_params = followup["litellm_params"]
    assert child_params is not parent_params
    assert child_params["metadata"] is not parent_params["metadata"]
    assert "_otel_internal" not in child_params["metadata"]
    assert child_params["metadata"]["session_id"] == "sess_abc"
