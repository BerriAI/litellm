import logging
from typing import Final

import pytest

from litellm.litellm_core_utils.internal_key_emission_guard import (
    internal_key_leak_counter,
    litellm_owned_key_paths,
    observe_internal_keys,
)


def test_nested_owned_key_reports_dotted_path() -> None:
    body: Final = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"litellm_call_id": "x", "user_tag": 1},
    }
    assert litellm_owned_key_paths(body) == ("metadata.litellm_call_id",)


def test_top_level_owned_keys_report_in_insertion_order() -> None:
    body: Final = {"model": "m", "litellm_params": {"api_base": "x"}, "_litellm_trace": 1}
    assert litellm_owned_key_paths(body) == ("litellm_params", "_litellm_trace")


def test_sequence_index_owned_key_reports_indexed_path() -> None:
    body: Final = {"model": "m", "tools": [{"_litellm_x": 1}]}
    assert litellm_owned_key_paths(body) == ("tools[0]._litellm_x",)


def test_tool_call_history_ids_are_not_flagged() -> None:
    body: Final = {
        "model": "m",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"ttl": "5m"}}]},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ],
    }
    assert litellm_owned_key_paths(body) == ()


def test_json_schema_property_names_are_not_flagged() -> None:
    schema: Final = {"type": "object", "properties": {"id": {"type": "string"}, "tags": {"type": "array"}}}
    body: Final = {
        "model": "m",
        "tools": [{"type": "function", "function": {"name": "f", "parameters": schema}}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}},
    }
    assert litellm_owned_key_paths(body) == ()


def test_owned_key_below_metadata_is_scoped_to_metadata_itself() -> None:
    body: Final = {"model": "m", "metadata": {"user_tag": 1, "nested": {"litellm_call_id": "x"}}}
    assert litellm_owned_key_paths(body) == ()


def test_bedrock_passthrough_container_owned_key_reports_full_path() -> None:
    body: Final = {
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        "additionalModelRequestFields": {"top_k": 5, "extra_body": {"litellm_call_id": "x", "vendor_flag": 1}},
    }
    assert litellm_owned_key_paths(body) == ("additionalModelRequestFields.extra_body.litellm_call_id",)


def test_bedrock_passthrough_container_top_level_owned_key_reports_path() -> None:
    body: Final = {
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        "additionalModelRequestFields": {"litellm_call_id": "x"},
    }
    assert litellm_owned_key_paths(body) == ("additionalModelRequestFields.litellm_call_id",)


def test_clean_body_reports_no_paths() -> None:
    body: Final = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
    }
    assert litellm_owned_key_paths(body) == ()


def test_user_supplied_extra_body_keys_are_not_flagged() -> None:
    transformed: Final = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    extra_body: Final = {"custom_flag": True, "nested": {"vendor_option": 1}}
    data: Final = {**transformed, **extra_body}
    assert litellm_owned_key_paths(data) == ()


def test_observe_internal_keys_counts_and_warns_on_dirty_body(caplog: pytest.LogCaptureFixture) -> None:
    before: Final = internal_key_leak_counter.value
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        paths: Final = observe_internal_keys({"litellm_call_id": "x", "model": "m"}, "deepseek")
    assert paths == ("litellm_call_id",)
    assert internal_key_leak_counter.value == before + 1
    assert [record.getMessage() for record in caplog.records] == [
        "LiteLLM internal keys reached the deepseek provider request body: litellm_call_id"
    ]


def test_observe_internal_keys_does_not_count_or_warn_on_clean_body(caplog: pytest.LogCaptureFixture) -> None:
    before: Final = internal_key_leak_counter.value
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        paths: Final = observe_internal_keys({"model": "m", "temperature": 0.2}, "deepseek")
    assert paths == ()
    assert internal_key_leak_counter.value == before
    assert not caplog.records
