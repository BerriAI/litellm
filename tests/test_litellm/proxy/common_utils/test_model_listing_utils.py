"""Model identity survives Claude Code presentation, filtering and configured alias precedence."""

from itertools import combinations

import pytest

import litellm
from litellm import Router
from litellm.proxy.common_utils.model_listing_utils import (
    ClaudeCodeRoutingNames,
    alias_listing_entries,
    alias_target,
    caller_alias_maps,
    claude_code_group_name,
    claude_code_model_id,
    claude_code_requested_group,
    claude_code_view_ids,
)


def _encoded(name):
    return "claude-router-" + name.encode().hex()


def _marked(name):
    return f"{_encoded(name)}[1m]"


def _row(name, limit=1000000):
    return {"id": name, "object": "model", "created": 0, "owned_by": "openai", "max_input_tokens": limit}


def _router(*names, aliases=None):
    return Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"}} for name in names
        ],
        model_group_alias=aliases,
    )


@pytest.mark.parametrize("limit", [None, 999999, 1000000])
@pytest.mark.parametrize(
    "name", ["foo", "foo[1m]", "foo[1M]", "a/b: 世界", "claude-router-foo", "claude-opus-5", "claude-opus-5[1m]"]
)
def test_listing_round_trips_entire_source_name(name, limit):
    names = frozenset({name})
    view = claude_code_model_id(name, limit, names)
    assert (claude_code_group_name(view, names) or view) == name
    if "claude" not in name:
        assert view.startswith(_encoded(name))
    assert ("[1m]" in view.lower()) == (limit == 1000000 or "[1m]" in name.lower() and "claude" in name)


def test_collision_matrix_round_trips_without_duplicate_ids():
    universe = (
        "foo",
        "foo[1m]",
        "claude-router-foo",
        _encoded("foo"),
        _encoded("foo") + "[1m]",
        "claude-opus-5",
        "claude-opus-5[1m]",
    )
    for pair in combinations(universe, 2):
        for visible in (pair, pair[:1], pair[1:]):
            names = frozenset(pair)
            view = claude_code_view_ids(tuple(_row(n) for n in visible), {"user-agent": "claude-code/2.1.267"}, names)
            assert len(set(view.values())) == len(visible)
            assert all((claude_code_group_name(shown, names) or shown) == source for source, shown in view.items())


@pytest.mark.parametrize(
    "spelling",
    [
        "claude-router-foo",
        "claude-router-ff",
        "claude-router-66 6f6f",
        "claude-router-666F6F",
        "claude-router-",
        _encoded("missing"),
    ],
)
def test_unknown_or_noncanonical_ids_are_never_guessed(spelling):
    assert claude_code_group_name(spelling, frozenset({"foo"})) is None


@pytest.mark.parametrize(
    "headers,enabled",
    [
        ({"user-agent": "claude-code/2.1.267"}, True),
        ({"user-agent": "claude-cli/2.1.267 (external, sdk-cli)"}, True),
        ({"x-gateway-client": "Claude-Code"}, True),
        ({"user-agent": "anthropic-sdk-python/0.40"}, False),
        ({}, False),
    ],
)
def test_only_claude_code_gets_the_view(headers, enabled):
    rows = (_row("foo"), _row("claude-opus-5"))
    view = claude_code_view_ids(rows, headers, frozenset(row["id"] for row in rows))
    assert dict(view) == ({"foo": _encoded("foo") + "[1m]", "claude-opus-5": "claude-opus-5[1m]"} if enabled else {})


@pytest.mark.parametrize("layer", ["literal", "global", "router", "key", "team", "wildcard"])
def test_configured_names_outrank_generated_ids_even_when_hidden_from_listing(monkeypatch, layer):
    encoded = _encoded("foo")
    alias = {encoded: "other"}
    monkeypatch.setattr(litellm, "model_alias_map", alias if layer == "global" else {})
    router = _router(
        "foo",
        "other",
        *((encoded,) if layer == "literal" else ("*",) if layer == "wildcard" else ()),
        aliases=alias if layer == "router" else None,
    )
    maps = (alias,) if layer in ("key", "team") else ()
    names = ClaudeCodeRoutingNames(router, None, maps)
    assert claude_code_requested_group(encoded, router, None, maps) is None
    assert claude_code_view_ids((_row("foo", None),), {"x-gateway-client": "claude-code"}, names)["foo"] == "foo"


@pytest.mark.parametrize("source", ["foo", "foo[1m]", "世界"])
def test_mutation_breaking_the_hex_name_cannot_route_to_the_source(source):
    router = _router(source)
    encoded = _encoded(source)
    malformed = encoded[:-1] + ("0" if encoded[-1] != "0" else "1")
    assert claude_code_requested_group(malformed, router, None) is None
    assert claude_code_requested_group(_marked(source), router, None) == source


def test_team_alias_is_listed_under_its_target_metadata_and_only_when_the_target_is_accessible():
    entries = [("gpt-4.1-mini", "gpt-4.1-mini"), ("team-public", "model_name_team_1_abc")]
    aliases = (
        {"gpt-4.1-mini": "team-public"},
        None,
        {"claude-sonnet-4-5": "gpt-4.1-mini", "via-public": "team-public", "not-granted": "gpt-4.1"},
    )
    assert alias_listing_entries(entries, aliases) == (
        *entries,
        ("claude-sonnet-4-5", "gpt-4.1-mini"),
        ("via-public", "model_name_team_1_abc"),
    )
    assert alias_listing_entries(entries, (None, {})) == tuple(entries)


def test_alias_target_resolves_the_requested_alias_across_key_and_team_maps():
    assert (
        alias_target("claude-sonnet-4-5", ({"o": "gpt-4.1"}, {"claude-sonnet-4-5": "gpt-4.1-mini"})) == "gpt-4.1-mini"
    )
    assert alias_target("gpt-4.1-mini", (None, {"claude-sonnet-4-5": "gpt-4.1-mini"})) is None


def test_alias_maps_apply_in_the_order_chat_completions_applies_them():
    team_then_key = ({"fast": "gpt-4.1-mini", "hop": "mid"}, {"fast": "gpt-4.1", "mid": "gpt-4.1"})
    entries = [("gpt-4.1-mini", "gpt-4.1-mini"), ("gpt-4.1", "gpt-4.1")]

    assert alias_target("fast", team_then_key) == "gpt-4.1-mini"
    assert alias_target("hop", team_then_key) == "gpt-4.1"
    assert alias_listing_entries(entries, team_then_key) == (
        *entries,
        ("fast", "gpt-4.1-mini"),
        ("hop", "gpt-4.1"),
        ("mid", "gpt-4.1"),
    )


def test_one_bad_alias_entry_hides_only_itself():
    aliases = {"fast": "gpt-4.1-mini", "broken": 5, 7: "gpt-4.1-mini"}
    entries = [("gpt-4.1-mini", "gpt-4.1-mini")]

    assert alias_listing_entries(entries, (aliases,)) == (*entries, ("fast", "gpt-4.1-mini"))
    assert alias_target("fast", (aliases,)) == "gpt-4.1-mini"


def test_chained_key_alias_is_listed_only_when_its_final_target_is_listable():
    key_aliases = {"a": "b", "b": "hidden"}
    entries = [("b", "b")]

    assert alias_listing_entries(entries, caller_alias_maps(key_aliases, None, "team-a", None)) == (*entries,)
    assert alias_target("a", caller_alias_maps(key_aliases, None, "team-a", None)) == "hidden"


def test_team_aliases_only_apply_when_listing_the_team_the_key_authenticated_as(monkeypatch):
    key_aliases, team_aliases, global_aliases = {"k": "gpt-4.1"}, {"t": "gpt-4.1-mini"}, {"g": "gpt-4.1"}
    monkeypatch.setattr(litellm, "model_alias_map", global_aliases)
    own_team = (team_aliases, key_aliases, global_aliases, key_aliases)
    assert caller_alias_maps(key_aliases, team_aliases, "team-a", None) == own_team
    assert caller_alias_maps(key_aliases, team_aliases, "team-a", "team-a") == own_team
    assert caller_alias_maps(key_aliases, team_aliases, "team-a", "team-b") == (
        key_aliases,
        global_aliases,
        key_aliases,
    )


def test_global_alias_rewrites_between_the_two_key_passes_like_chat_completions(monkeypatch):
    monkeypatch.setattr(litellm, "model_alias_map", {"b": "d"})
    key_aliases = {"a": "b", "b": "c"}
    entries = [("c", "c"), ("d", "d")]
    maps = caller_alias_maps(key_aliases, None, "team-a", None)

    assert alias_target("a", maps) == "d"
    assert alias_listing_entries(entries, maps) == (*entries, ("a", "d"), ("b", "c"))


def test_team_public_name_uses_the_same_scope_at_list_and_request():
    router = Router(
        model_list=[
            {
                "model_name": "model_name_team-a_id",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"},
                "model_info": {"team_id": "team-a", "team_public_model_name": "shared"},
            }
        ]
    )
    shown = claude_code_view_ids(
        (_row("shared"),), {"user-agent": "claude-code/2.1.267"}, ClaudeCodeRoutingNames(router, "team-a")
    )["shared"]
    assert claude_code_requested_group(shown, router, "team-a") == "shared"
    assert claude_code_requested_group(shown, router, "team-b") is None
