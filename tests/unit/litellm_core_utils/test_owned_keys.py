import copy
import json
from collections.abc import Mapping
from typing import Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from litellm.litellm_core_utils.owned_keys import (
    OWNED_KEYS,
    is_owned_key,
    loggable_owned_keys,
    owned_keys_in,
    parse_request_body,
    request_body_view,
)

_JSON_SCALAR: Final = st.none() | st.booleans() | st.integers() | st.text()
_JSON_VALUE: Final = st.recursive(
    _JSON_SCALAR,
    lambda children: st.lists(children, max_size=5) | st.dictionaries(st.text(), children, max_size=5),
    max_leaves=30,
)
_KEY: Final = st.one_of(
    st.sampled_from(sorted(OWNED_KEYS)),
    st.text().map(lambda value: "_litellm_" + value),
    st.text(),
)
_BODY: Final = st.dictionaries(_KEY, _JSON_VALUE, max_size=10)


def test_top_level_owned_key_is_flagged() -> None:
    body: Final = {"model": "gpt-5.4", "model_info": {"id": "x"}}

    assert owned_keys_in(body) == ("model_info",)


@pytest.mark.parametrize("key", ("_litellm_probe", "_litellm_zzz"))
def test_prefixed_key_is_flagged(key: str) -> None:
    assert owned_keys_in({"model": "gpt-5.4", key: 1}) == (key,)


@pytest.mark.parametrize(
    ("container", "inner", "expected"),
    (
        ("metadata", {"user_api_key_hash": "h"}, ("metadata.user_api_key_hash",)),
        ("extra_body", {"_litellm_probe": 1}, ("_litellm_probe",)),
        ("litellm_metadata", {"user_api_key_hash": "h"}, ("litellm_metadata", "litellm_metadata.user_api_key_hash")),
        (
            "additionalModelRequestFields",
            {"_litellm_probe": 1},
            ("additionalModelRequestFields._litellm_probe",),
        ),
    ),
)
def test_owned_keys_one_level_under_any_mapping_are_flagged(
    container: str, inner: Mapping[str, object], expected: tuple[str, ...]
) -> None:
    assert owned_keys_in({"model": "gpt-5.4", container: inner}) == expected


def test_extra_body_is_flattened_before_walk() -> None:
    body: Final = {"model": "gpt-5.4", "extra_body": {"metadata": {"user_api_key_hash": "h"}}}

    assert owned_keys_in(body) == ("metadata.user_api_key_hash",)


@pytest.mark.parametrize("extra_body", ("user_api_key_hash", ["user_api_key_hash"], [{"user_api_key_hash": "h"}], None))
def test_non_mapping_extra_body_is_left_alone(extra_body: object) -> None:
    body: Final = {"model": "gpt-5.4", "extra_body": extra_body}

    assert request_body_view(body) is body
    assert owned_keys_in(body) == ()


def test_extra_body_flatten_reports_every_owned_key_once() -> None:
    body: Final = {
        "model": "gpt-5.4",
        "model_info": {},
        "extra_body": {"_litellm_probe": 1, "metadata": {"user_api_key_hash": "h", "litellm_call_id": "c"}},
    }

    assert owned_keys_in(body) == (
        "_litellm_probe",
        "metadata.litellm_call_id",
        "metadata.user_api_key_hash",
        "model_info",
    )


def test_key_two_levels_deep_is_not_flagged() -> None:
    body: Final = {
        "model": "gpt-5.4",
        "additionalModelRequestFields": {"extra_body": {"_litellm_probe": 1}},
    }

    assert owned_keys_in(body) == ()


def test_owned_key_inside_a_list_held_mapping_is_not_flagged() -> None:
    body: Final = {
        "model": "gpt-5.4",
        "messages": [{"_litellm_probe": 1, "user_api_key_hash": "h"}],
    }

    assert owned_keys_in(body) == ()


def test_owned_names_inside_tool_schemas_and_tool_arguments_are_not_flagged() -> None:
    arguments: Final = json.dumps({"model_info": {}, "rpm": 1, "tpm": 1, "user_api_key_hash": "h"})
    body: Final = {
        "model": "gpt-5.4",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": arguments}}],
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_info": {"type": "string"},
                            "rpm": {"type": "integer"},
                            "tpm": {"type": "integer"},
                            "user_api_key_hash": {"type": "string"},
                            "_litellm_probe": {"type": "string"},
                        },
                    },
                },
            }
        ],
    }

    assert owned_keys_in(body) == ()


def test_plain_metadata_is_not_flagged() -> None:
    assert owned_keys_in({"model": "gpt-5.4", "metadata": {"foo": "bar"}}) == ()


@pytest.mark.parametrize("key", tuple(sorted(OWNED_KEYS)))
def test_every_owned_key_is_flagged_at_top_level(key: str) -> None:
    assert owned_keys_in({"model": "gpt-5.4", key: 1}) == (key,)


@pytest.mark.parametrize(
    "key", ("user", "stream_chunk_size", "litellm_probe", "_litellmx", "_litellm", "x_litellm_y", "x_litellm_")
)
def test_non_owned_and_prefix_boundary_keys_are_not_flagged(key: str) -> None:
    assert owned_keys_in({"model": "gpt-5.4", key: 1}) == ()


def test_result_is_sorted_regardless_of_insertion_order() -> None:
    expected: Final = (
        "_litellm_probe",
        "metadata.litellm_call_id",
        "metadata.user_api_key_hash",
        "model_info",
    )
    forward: Final = {
        "extra_body": {"_litellm_probe": 1},
        "metadata": {"user_api_key_hash": "h", "litellm_call_id": "c"},
        "model_info": {},
    }
    reversed_order: Final = {
        "model_info": {},
        "metadata": {"litellm_call_id": "c", "user_api_key_hash": "h"},
        "extra_body": {"_litellm_probe": 1},
    }

    assert owned_keys_in(forward) == expected
    assert owned_keys_in(reversed_order) == expected
    assert owned_keys_in(forward) == owned_keys_in(reversed_order)


@settings(max_examples=40, deadline=None)
@given(_BODY)
def test_owned_keys_in_does_not_mutate_and_never_raises(body: dict[str, object]) -> None:
    before: Final = copy.deepcopy(body)
    serialized_before: Final = json.dumps(body, sort_keys=True, default=str)
    flagged: Final = owned_keys_in(body)

    assert body == before
    assert json.dumps(body, sort_keys=True, default=str) == serialized_before
    assert flagged == tuple(sorted(set(flagged)))
    assert all(is_owned_key(path) or is_owned_key(path.rsplit(".", 1)[-1]) for path in flagged)
    view: Final = request_body_view(body)
    has_owned_key: Final = any(
        is_owned_key(key)
        or (isinstance(value, Mapping) and any(is_owned_key(inner_key) for inner_key in value))
        for key, value in view.items()
        if isinstance(key, str)
    )
    if not has_owned_key:
        assert flagged == ()


@settings(max_examples=30, deadline=None)
@given(_BODY)
def test_parse_request_body_json_objects_round_trip(body: dict[str, object]) -> None:
    parsed: Final = parse_request_body(json.dumps(body))

    assert parsed == body


@settings(max_examples=30, deadline=None)
@given(st.text())
def test_parse_request_body_arbitrary_text_never_raises(text: str) -> None:
    parsed: Final = parse_request_body(text)

    assert parsed is None or isinstance(parsed, Mapping)


@pytest.mark.parametrize("text", ("[1,2]", "not json"))
def test_parse_request_body_non_object_json_is_none(text: str) -> None:
    assert parse_request_body(text) is None


@settings(max_examples=30, deadline=None)
@given(st.lists(_KEY, max_size=10).map(tuple))
def test_loggable_owned_keys_uses_only_safe_names(flagged: tuple[str, ...]) -> None:
    result: Final = loggable_owned_keys(flagged)

    assert set(result) <= OWNED_KEYS | {"_litellm_*"}
    assert result == tuple(sorted(set(result)))


def test_loggable_owned_keys_preserves_known_leaves_and_redacts_prefixes() -> None:
    assert loggable_owned_keys(("metadata.user_api_key_hash", "_litellm_foo")) == (
        "_litellm_*",
        "user_api_key_hash",
    )
