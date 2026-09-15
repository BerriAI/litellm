"""Model identity survives Claude Code presentation, filtering and configured alias precedence."""

from itertools import combinations

import pytest

from litellm import Router
from litellm.proxy.common_utils.model_listing_utils import (
    ClaudeCodeRoutingNames,
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
            {"model_name": name, "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"}}
            for name in names
        ],
        model_group_alias=aliases,
    )


@pytest.mark.parametrize("limit", [None, 999999, 1000000])
@pytest.mark.parametrize("name", ["foo", "foo[1m]", "foo[1M]", "a/b: 世界", "claude-router-foo", "claude-opus-5", "claude-opus-5[1m]"])
def test_listing_round_trips_entire_source_name(name, limit):
    names = frozenset({name})
    view = claude_code_model_id(name, limit, names)
    assert (claude_code_group_name(view, names) or view) == name
    if "claude" not in name:
        assert view.startswith(_encoded(name))
    assert ("[1m]" in view.lower()) == (limit == 1000000 or "[1m]" in name.lower() and "claude" in name)


def test_collision_matrix_round_trips_without_duplicate_ids():
    universe = ("foo", "foo[1m]", "claude-router-foo", _encoded("foo"), _encoded("foo") + "[1m]", "claude-opus-5", "claude-opus-5[1m]")
    for pair in combinations(universe, 2):
        for visible in (pair, pair[:1], pair[1:]):
            names = frozenset(pair)
            view = claude_code_view_ids(tuple(_row(n) for n in visible), {"user-agent": "claude-code/2.1.267"}, names)
            assert len(set(view.values())) == len(visible)
            assert all((claude_code_group_name(shown, names) or shown) == source for source, shown in view.items())


@pytest.mark.parametrize("spelling", ["claude-router-foo", "claude-router-ff", "claude-router-66 6f6f", "claude-router-666F6F", "claude-router-", _encoded("missing")])
def test_unknown_or_noncanonical_ids_are_never_guessed(spelling):
    assert claude_code_group_name(spelling, frozenset({"foo"})) is None


@pytest.mark.parametrize("headers,enabled", [
    ({"user-agent": "claude-code/2.1.267"}, True),
    ({"user-agent": "claude-cli/2.1.267 (external, sdk-cli)"}, True),
    ({"x-gateway-client": "Claude-Code"}, True),
    ({"user-agent": "anthropic-sdk-python/0.40"}, False),
    ({}, False),
])
def test_only_claude_code_gets_the_view(headers, enabled):
    rows = (_row("foo"), _row("claude-opus-5"))
    view = claude_code_view_ids(rows, headers, frozenset(row["id"] for row in rows))
    assert dict(view) == ({"foo": _encoded("foo") + "[1m]", "claude-opus-5": "claude-opus-5[1m]"} if enabled else {})


@pytest.mark.parametrize("layer", ["literal", "global", "router", "key", "team", "wildcard"])
def test_configured_names_outrank_generated_ids_even_when_hidden_from_listing(monkeypatch, layer):
    import litellm

    encoded = _encoded("foo")
    alias = {encoded: "other"}
    monkeypatch.setattr(litellm, "model_alias_map", alias if layer == "global" else {})
    router = _router("foo", "other", *( (encoded,) if layer == "literal" else ("*",) if layer == "wildcard" else ()), aliases=alias if layer == "router" else None)
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


def test_team_public_name_uses_the_same_scope_at_list_and_request():
    router = Router(model_list=[{
        "model_name": "model_name_team-a_id",
        "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"},
        "model_info": {"team_id": "team-a", "team_public_model_name": "shared"},
    }])
    shown = claude_code_view_ids((_row("shared"),), {"user-agent": "claude-code/2.1.267"}, ClaudeCodeRoutingNames(router, "team-a"))["shared"]
    assert claude_code_requested_group(shown, router, "team-a") == "shared"
    assert claude_code_requested_group(shown, router, "team-b") is None
