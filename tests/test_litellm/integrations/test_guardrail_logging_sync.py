"""
Regression test for _sync_guardrail_info_to_logging_obj.

Ensures that when the @log_guardrail_information decorator writes guardrail info
to request_data["litellm_metadata"] (as it does for /v1/messages passthrough
routes that have no "metadata" key), the helper propagates it into
logging_obj.litellm_params["metadata"] so merge_litellm_metadata surfaces it in
spend logs.
"""

import pytest
from litellm.integrations.custom_guardrail import _sync_guardrail_info_to_logging_obj


def _make_slg_entry(name: str = "headroom-test") -> dict:
    return {
        "guardrail_name": name,
        "guardrail_response": "mask",
        "guardrail_status": "success",
        "duration": 0.1,
    }


class _FakeLogging:
    """Minimal stand-in for litellm.litellm_core_utils.litellm_logging.Logging."""

    def __init__(self, lp_metadata: dict | None = None):
        self.litellm_params: dict = {"metadata": lp_metadata or {}}
        self.model_call_details: dict = {"litellm_params": self.litellm_params}


def test_syncs_from_litellm_metadata_key():
    """When guardrail info is in request_data["litellm_metadata"], it is copied."""
    entry = _make_slg_entry()
    request_data = {
        "litellm_metadata": {"standard_logging_guardrail_information": [entry]}
    }
    logging_obj = _FakeLogging()

    _sync_guardrail_info_to_logging_obj(request_data, logging_obj)

    result = logging_obj.litellm_params["metadata"].get(
        "standard_logging_guardrail_information"
    )
    assert result == [entry]


def test_syncs_from_metadata_key():
    """When guardrail info is in request_data["metadata"], it is also copied."""
    entry = _make_slg_entry()
    request_data = {"metadata": {"standard_logging_guardrail_information": [entry]}}
    logging_obj = _FakeLogging()

    _sync_guardrail_info_to_logging_obj(request_data, logging_obj)

    result = logging_obj.litellm_params["metadata"].get(
        "standard_logging_guardrail_information"
    )
    assert result == [entry]


def test_metadata_wins_over_litellm_metadata():
    """metadata key takes precedence over litellm_metadata when both are present."""
    entry_meta = _make_slg_entry("from-metadata")
    entry_lm = _make_slg_entry("from-litellm_metadata")
    request_data = {
        "metadata": {"standard_logging_guardrail_information": [entry_meta]},
        "litellm_metadata": {"standard_logging_guardrail_information": [entry_lm]},
    }
    logging_obj = _FakeLogging()

    _sync_guardrail_info_to_logging_obj(request_data, logging_obj)

    result = logging_obj.litellm_params["metadata"].get(
        "standard_logging_guardrail_information"
    )
    assert result == [entry_meta]


def test_noop_when_no_guardrail_info():
    """Does nothing when standard_logging_guardrail_information is absent."""
    request_data = {"litellm_metadata": {"other_key": "value"}}
    logging_obj = _FakeLogging()

    _sync_guardrail_info_to_logging_obj(request_data, logging_obj)

    assert (
        logging_obj.litellm_params["metadata"].get(
            "standard_logging_guardrail_information"
        )
        is None
    )


def test_noop_when_logging_obj_is_none():
    """Does nothing when logging_obj is None."""
    entry = _make_slg_entry()
    request_data = {
        "litellm_metadata": {"standard_logging_guardrail_information": [entry]}
    }
    _sync_guardrail_info_to_logging_obj(request_data, None)


def test_writes_to_model_call_details_too():
    """Also writes into model_call_details["litellm_params"]["metadata"]."""
    entry = _make_slg_entry()
    request_data = {
        "litellm_metadata": {"standard_logging_guardrail_information": [entry]}
    }

    logging_obj = _FakeLogging()
    # Simulate litellm_params reassignment (creating a new dict) — model_call_details
    # then points to the OLD dict while litellm_params points to the new one.
    old_lp = logging_obj.litellm_params
    logging_obj.litellm_params = {**old_lp, "extra": "added"}
    logging_obj.model_call_details["litellm_params"] = old_lp  # diverged

    _sync_guardrail_info_to_logging_obj(request_data, logging_obj)

    # Both dicts should have the info.
    assert logging_obj.litellm_params["metadata"].get(
        "standard_logging_guardrail_information"
    ) == [entry]
    assert old_lp["metadata"].get("standard_logging_guardrail_information") == [entry]
