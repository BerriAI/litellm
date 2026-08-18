"""
Pure-logic contract tests for litellm/proxy/video_endpoints/utils.py

Three helpers the video proxy endpoints lean on:
  - extract_model_from_target_model_names: first model from a comma string / list
  - get_custom_provider_from_data: provider precedence (top-level > extra_body)
  - encode_character_id_in_response: re-encode a response id in place

Every test asserts the exact result (or identity), so a mutation that flips a
branch, drops a strip/filter, or changes precedence fails. The only collaborator
is encode_character_id_with_provider, which runs for real; encoding assertions
are checked by the genuine decode round-trip.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.video_endpoints.utils import (
    encode_character_id_in_response,
    extract_model_from_target_model_names,
    get_custom_provider_from_data,
)
from litellm.types.videos.utils import (
    decode_character_id_with_provider,
    encode_character_id_with_provider,
)

# =========================================================================== #
# extract_model_from_target_model_names
# =========================================================================== #


@pytest.mark.parametrize(
    "value,expected",
    [
        ("m1,m2,m3", "m1"),
        ("  a , b ", "a"),  # leading/trailing whitespace stripped
        (",, m1 ,,", "m1"),  # empty tokens filtered out
        ("solo", "solo"),  # single token, no comma
        ("", None),  # empty string -> no tokens
        (" , , ", None),  # only separators/whitespace -> no tokens
        (["x", "y"], "x"),  # list -> first element
        ([], None),  # empty list
    ],
)
def test_extract_model__str_and_list(value, expected):
    assert extract_model_from_target_model_names(value) == expected


@pytest.mark.parametrize("value", [None, 123, {"a": 1}, 4.5])
def test_extract_model__non_str_non_list_is_none(value):
    assert extract_model_from_target_model_names(value) is None


# =========================================================================== #
# get_custom_provider_from_data
# =========================================================================== #


def test_provider__top_level_wins_over_extra_body():
    data = {
        "custom_llm_provider": "azure",
        "extra_body": {"custom_llm_provider": "openai"},
    }
    assert get_custom_provider_from_data(data) == "azure"


@pytest.mark.parametrize("falsy", ["", None])
def test_provider__falsy_top_level_falls_through_to_extra_body(falsy):
    data = {
        "custom_llm_provider": falsy,
        "extra_body": {"custom_llm_provider": "vertex_ai"},
    }
    assert get_custom_provider_from_data(data) == "vertex_ai"


def test_provider__from_extra_body_dict():
    assert (
        get_custom_provider_from_data(
            {"extra_body": {"custom_llm_provider": "bedrock"}}
        )
        == "bedrock"
    )


def test_provider__from_extra_body_json_string():
    data = {"extra_body": '{"custom_llm_provider": "gemini"}'}
    assert get_custom_provider_from_data(data) == "gemini"


def test_provider__invalid_json_string_is_none():
    assert get_custom_provider_from_data({"extra_body": "not-json{"}) is None


def test_provider__json_string_parsing_to_non_dict_is_none():
    # parses to a list, not a dict -> no provider extracted.
    assert get_custom_provider_from_data({"extra_body": "[1, 2]"}) is None


def test_provider__extra_body_provider_not_a_string_is_none():
    assert (
        get_custom_provider_from_data({"extra_body": {"custom_llm_provider": 123}})
        is None
    )


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"extra_body": {}},
        {"extra_body": 5},  # non-dict, non-str
        {"extra_body": {"other": "x"}},  # dict without provider key
    ],
)
def test_provider__no_provider_anywhere_is_none(data):
    assert get_custom_provider_from_data(data) is None


# =========================================================================== #
# encode_character_id_in_response
# =========================================================================== #


class _Resp:
    """Minimal response object exposing an `id` attribute."""


def test_encode__dict_with_id_mutates_in_place_and_preserves_other_keys():
    response = {"id": "char_raw", "object": "character", "name": "hero"}

    out = encode_character_id_in_response(response, "azure", "model-1")

    assert out is response  # same dict, mutated in place
    assert out["object"] == "character" and out["name"] == "hero"
    assert out["id"] == encode_character_id_with_provider(
        "char_raw", "azure", "model-1"
    )
    decoded = decode_character_id_with_provider(out["id"])
    assert decoded["custom_llm_provider"] == "azure"
    assert decoded["model_id"] == "model-1"
    assert decoded["character_id"] == "char_raw"


@pytest.mark.parametrize("response", [{}, {"id": ""}, {"id": None}])
def test_encode__dict_without_usable_id_unchanged(response):
    snapshot = dict(response)
    out = encode_character_id_in_response(response, "azure", "model-1")
    assert out == snapshot


def test_encode__object_with_str_id():
    resp = _Resp()
    resp.id = "char_raw"

    out = encode_character_id_in_response(resp, "openai", None)

    assert out is resp
    assert resp.id == encode_character_id_with_provider("char_raw", "openai", None)
    decoded = decode_character_id_with_provider(resp.id)
    assert decoded["custom_llm_provider"] == "openai"
    assert decoded["character_id"] == "char_raw"


@pytest.mark.parametrize("bad_id", [None, 123, ""])
def test_encode__object_non_str_or_empty_id_unchanged(bad_id):
    resp = _Resp()
    resp.id = bad_id

    out = encode_character_id_in_response(resp, "azure", "model-1")

    assert out is resp
    assert resp.id == bad_id  # untouched


def test_encode__object_without_id_attr_returned_unchanged():
    resp = _Resp()
    out = encode_character_id_in_response(resp, "azure", "model-1")
    assert out is resp
    assert not hasattr(resp, "id")
