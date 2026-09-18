"""Unit tests for ``_carry_guardrail_logging_info``.

This is the helper that lets a passthrough guardrail block still surface its otel
span: it copies ``standard_logging_guardrail_information`` from the post-call
guardrail's (otherwise discarded) ``hook_data`` onto the dict the failure handler
forwards to ``post_call_failure_hook``. No otel dependency here, so these run
everywhere and pin the helper's contract directly.
"""

from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    _carry_guardrail_logging_info,
)

_ENTRY = {"guardrail_name": "block-demo", "guardrail_status": "guardrail_intervened"}


def _source(entries):
    return {"metadata": {"standard_logging_guardrail_information": entries}}


def test_carries_entries_onto_request_without_metadata():
    request_data: dict = {}
    _carry_guardrail_logging_info(request_data, _source([_ENTRY]))
    assert request_data["metadata"]["standard_logging_guardrail_information"] == [
        _ENTRY
    ]


def test_carried_list_is_copied_not_shared():
    source = _source([_ENTRY])
    request_data: dict = {}
    _carry_guardrail_logging_info(request_data, source)
    carried = request_data["metadata"]["standard_logging_guardrail_information"]
    assert carried is not source["metadata"]["standard_logging_guardrail_information"]
    carried.append({"guardrail_name": "other"})
    assert source["metadata"]["standard_logging_guardrail_information"] == [_ENTRY]


def test_existing_metadata_without_guardrail_key_is_populated():
    request_data: dict = {"metadata": {"user_api_key": "sk-x"}}
    _carry_guardrail_logging_info(request_data, _source([_ENTRY]))
    assert request_data["metadata"]["user_api_key"] == "sk-x"
    assert request_data["metadata"]["standard_logging_guardrail_information"] == [
        _ENTRY
    ]


def test_existing_guardrail_entries_are_not_clobbered():
    existing = [{"guardrail_name": "already-logged"}]
    request_data = {"metadata": {"standard_logging_guardrail_information": existing}}
    _carry_guardrail_logging_info(request_data, _source([_ENTRY]))
    assert (
        request_data["metadata"]["standard_logging_guardrail_information"] is existing
    )


def test_noop_when_guardrail_data_is_none():
    request_data: dict = {}
    _carry_guardrail_logging_info(request_data, None)
    assert request_data == {}


def test_noop_when_no_guardrail_entries():
    request_data: dict = {}
    _carry_guardrail_logging_info(request_data, {"metadata": {}})
    _carry_guardrail_logging_info(request_data, _source([]))
    _carry_guardrail_logging_info(request_data, {})
    assert request_data == {}
