import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.types.llms.bedrock_invoke import (
    assert_no_control_params,
    assert_no_control_params_in_payload,
    parse_invoke_inference_params,
)


def test_passthrough_keeps_unknown_keys():
    body = parse_invoke_inference_params(
        provider="mistral",
        model="mistral.mistral-7b-instruct-v0:2",
        params={"temperature": 0.5, "unknown_key": "x"},
        drop_params=False,
    )
    assert body == {"temperature": 0.5, "unknown_key": "x"}


def test_drop_params_strips_only_unknown_keys():
    body = parse_invoke_inference_params(
        provider="mistral",
        model="mistral.mistral-7b-instruct-v0:2",
        params={"temperature": 0.5, "unknown_key": "x"},
        drop_params=True,
    )
    assert body == {"temperature": 0.5}


def test_command_r_and_legacy_resolve_to_different_bodies():
    """command-r exposes k/p; the legacy text model does not, so the same key
    must be classified differently per model id."""
    command_r = parse_invoke_inference_params(
        provider="cohere",
        model="cohere.command-r-v1:0",
        params={"k": 2},
        drop_params=True,
    )
    legacy = parse_invoke_inference_params(
        provider="cohere",
        model="cohere.command-text-v14",
        params={"k": 2},
        drop_params=True,
    )
    assert command_r == {"k": 2.0}
    assert legacy == {}


def test_unmodeled_provider_passes_through_untouched():
    params = {"anything": 1, "stream_chunk_size": 4}
    assert (
        parse_invoke_inference_params(
            provider="anthropic",
            model="anthropic.claude-sonnet-4-6",
            params=params,
            drop_params=True,
        )
        == params
    )


def test_guard_raises_on_leaked_control_param():
    with pytest.raises(ValueError, match="stream_chunk_size"):
        assert_no_control_params({"temperature": 0.5, "stream_chunk_size": 2048})


def test_guard_payload_ignores_non_dict_and_invalid_json():
    assert_no_control_params_in_payload("not json")
    assert_no_control_params_in_payload("[1, 2, 3]")
    assert_no_control_params_in_payload('{"temperature": 0.5}')

    with pytest.raises(ValueError, match="stream_chunk_size"):
        assert_no_control_params_in_payload('{"stream_chunk_size": 2048}')
