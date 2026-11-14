from typing import Final

import pytest

from litellm.litellm_core_utils.prompt_templates.prompt_cache_breakpoint_utils import (
    cache_control_as_prompt_cache_breakpoint,
    prompt_cache_breakpoint_for_wire,
    with_prompt_cache_breakpoint,
)

_MALFORMED_PROMPT_CACHE_BREAKPOINTS: Final = (
    "yes",
    1,
    ["explicit"],
    "",
    {},
    {"mode": "bogus"},
    {"mode": "bogus", "ttl": "30m"},
    {1: "explicit"},
)


_MALFORMED_PROMPT_CACHE_BREAKPOINT_IDS: Final = (
    "string",
    "int",
    "list",
    "empty-string",
    "empty-object",
    "bogus-mode",
    "bogus-mode-with-ttl",
    "non-string-key",
)


_VALID_PROMPT_CACHE_BREAKPOINTS: Final = ({"mode": "explicit"}, {"mode": "explicit", "ttl": "30m"})


@pytest.mark.parametrize("drop_params", (True, False), ids=("drop", "keep"))
def test_prompt_cache_breakpoint_for_wire_passes_a_missing_marker_through(drop_params: bool) -> None:
    assert prompt_cache_breakpoint_for_wire(None, drop_params) is None


@pytest.mark.parametrize("marker", _MALFORMED_PROMPT_CACHE_BREAKPOINTS, ids=_MALFORMED_PROMPT_CACHE_BREAKPOINT_IDS)
def test_prompt_cache_breakpoint_for_wire_drops_a_malformed_marker_under_drop_params(marker: object) -> None:
    assert prompt_cache_breakpoint_for_wire(marker, True) is None


@pytest.mark.parametrize("marker", _MALFORMED_PROMPT_CACHE_BREAKPOINTS, ids=_MALFORMED_PROMPT_CACHE_BREAKPOINT_IDS)
def test_prompt_cache_breakpoint_for_wire_forwards_a_malformed_marker_without_drop_params(marker: object) -> None:
    assert prompt_cache_breakpoint_for_wire(marker, False) is marker


@pytest.mark.parametrize("drop_params", (True, False), ids=("drop", "keep"))
@pytest.mark.parametrize("marker", _VALID_PROMPT_CACHE_BREAKPOINTS, ids=("explicit", "explicit-30m"))
def test_prompt_cache_breakpoint_for_wire_keeps_a_valid_marker(marker: dict[str, str], drop_params: bool) -> None:
    assert prompt_cache_breakpoint_for_wire(marker, drop_params) == marker


def test_prompt_cache_breakpoint_for_wire_strips_an_unknown_key_under_drop_params() -> None:
    assert prompt_cache_breakpoint_for_wire({"mode": "explicit", "note": "kept"}, True) == {"mode": "explicit"}


@pytest.mark.parametrize(
    "marker",
    ({"mode": "explicit", "ttl": "1h"}, {"mode": "explicit", "ttl": "1h", "note": "kept"}),
    ids=("bad-ttl", "bad-ttl-and-unknown-key"),
)
def test_prompt_cache_breakpoint_for_wire_drops_an_unsupported_ttl_and_keeps_the_mode_under_drop_params(
    marker: dict[str, str],
) -> None:
    assert prompt_cache_breakpoint_for_wire(marker, True) == {"mode": "explicit"}
    assert prompt_cache_breakpoint_for_wire(marker, False) is marker


def test_cache_control_is_replaced_by_an_explicit_prompt_cache_breakpoint_on_the_same_block() -> None:
    marked: Final = {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral", "ttl": "1h"}}

    assert cache_control_as_prompt_cache_breakpoint(marked) == {
        "type": "text",
        "text": "rules",
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }


def test_a_block_without_cache_control_is_returned_untouched() -> None:
    unmarked: Final = {"type": "text", "text": "rules"}

    assert cache_control_as_prompt_cache_breakpoint(unmarked) is unmarked


def test_a_breakpoint_the_client_wrote_survives_a_message_level_mark_ttl_included() -> None:
    block: Final = {
        "type": "input_text",
        "text": "rules",
        "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "30m"},
    }

    assert with_prompt_cache_breakpoint(block, {"mode": "explicit"}) is block


def test_a_breakpoint_the_client_wrote_survives_a_per_part_cache_control() -> None:
    both: Final = {
        "type": "input_text",
        "text": "rules",
        "cache_control": {"type": "ephemeral"},
        "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "30m"},
    }

    assert cache_control_as_prompt_cache_breakpoint(both) == {
        "type": "input_text",
        "text": "rules",
        "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "30m"},
    }
