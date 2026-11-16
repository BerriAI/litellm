from collections import Counter
from enum import Enum

import pytest

from litellm.integrations.custom_logger_v2 import Activation, Field, FieldTier, Hook, OptOutBy


@pytest.mark.parametrize("enum_type", [Hook, Field, Activation, OptOutBy])
def test_enum_values_are_their_lowercase_names_so_config_and_manifest_share_one_string(enum_type: type[Enum]) -> None:
    mismatched = [(member.name, member.value) for member in enum_type if member.value != member.name.lower()]
    assert mismatched == []
    assert all(isinstance(member, str) for member in enum_type)


def test_field_round_trips_from_its_config_string() -> None:
    assert Field("provider_request_body") is Field.provider_request_body
    assert Field.messages == "messages"


def test_field_bits_are_distinct_and_fit_a_u64_bitset() -> None:
    bits = [member.bit for member in Field]
    assert len(set(bits)) == len(bits) == 47
    assert all(0 <= bit < 64 for bit in bits)


def test_fields_fall_in_the_four_cost_tiers_of_the_design() -> None:
    assert Counter(member.tier for member in Field) == {
        FieldTier.small_record: 27,
        FieldTier.content: 12,
        FieldTier.wire: 4,
        FieldTier.compatibility: 4,
    }
    assert {member for member in Field if member.tier is FieldTier.compatibility} == {
        Field.standard_logging_payload,
        Field.deprecated_kwargs,
        Field.deprecated_response_obj,
        Field.standard_logging_payload_ref,
    }


def test_field_bits_are_the_wire_contract_shared_with_the_rust_manifest() -> None:
    assert {member.value: member.bit for member in Field} == {
        "model": 0,
        "key_identity": 1,
        "deployment": 2,
        "candidate_deployments": 3,
        "request_tags": 4,
        "dynamic_params": 5,
        "guardrail_info": 6,
        "outcome": 7,
        "error": 8,
        "usage": 9,
        "cost": 10,
        "metadata": 11,
        "messages": 12,
        "input": 13,
        "tools": 14,
        "optional_params": 15,
        "request_headers": 16,
        "response": 17,
        "response_headers": 18,
        "stream_chunk": 19,
        "assembled_stream": 20,
        "tool_result": 21,
        "model_list": 22,
        "request_body": 23,
        "provider_request_body": 24,
        "provider_response_raw": 25,
        "provider_response_headers": 26,
        "standard_logging_payload": 27,
        "deprecated_kwargs": 28,
        "deprecated_response_obj": 29,
        "key_limits": 30,
        "budget_plan": 31,
        "budget_reservation": 32,
        "rate_limit_statuses": 33,
        "output_cap": 34,
        "attempt_timing": 35,
        "rate_limit_state": 36,
        "logged_request": 37,
        "mcp_tool_call": 38,
        "log_id": 39,
        "cold_storage_object_key": 40,
        "standard_logging_payload_ref": 41,
        "logged_key_identity": 42,
        "logged_model_group": 43,
        "metadata_labels": 44,
        "input_sequence_length": 45,
        "limiter_request": 46,
    }
