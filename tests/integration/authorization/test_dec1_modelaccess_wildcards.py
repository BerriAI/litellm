"""Wildcard entries in key, team and user model lists, and how they decide access and listings.

Contract (litellm/proxy/auth/auth_checks.py `check_model_access_helper`, `is_model_allowed_by_pattern`,
`model_matches_any_wildcard_pattern_in_list`, `_model_custom_llm_provider_matches_wildcard_pattern`;
litellm/proxy/auth/model_checks.py `_get_wildcard_models`):

An entry that contains `*` in a key, team or user `models` list is a pattern. The requested model name is
allowed when the regex `^<entry with each * replaced by .*>$` matches it. Only `*` is translated: every other
character, the dot included, is used as regex syntax as written, so a dot in a pattern matches any one
character. A bare model name that names no provider also matches `<provider>/*` when the provider inferred
from the bare name is that provider, but a name that already carries an unknown namespace never does. The
pattern is checked against the name the caller sent, before routing, so a denial is the usual 403 whether or
not a deployment serves the name.

In listings a pattern entry is never shown as itself unless `return_wildcard_routes=true` is passed (or
`general_settings.return_wildcard_routes` is on); it is replaced by the provider's known models instead.
These rules move into the Rust auth core with the key and team stores, so each one is pinned here.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration.authorization.dec1_modelaccess_support import (
    KEY_DENIED,
    TEAM_DENIED,
    USER_DENIED,
    assert_model_denied,
    assert_not_model_denied,
    assert_served,
    chat,
    marker,
    model_ids,
    unique_name,
    upstream_requests,
)


def test_a_provider_wildcard_on_a_key_serves_every_group_named_under_that_provider(gateway: Gateway) -> None:
    """Two deployments are named `openai/<unique>-a` and `openai/<unique>-b`, and a key has `models: ["openai/*"]`.

    Both names must be served with 200, because `^openai/.*$` matches each requested name. The key never
    names either group, so a port that only does exact list membership would refuse them.
    """
    with gateway.scenario() as scenario:
        stem: Final = unique_name()
        first: Final = scenario.model(name=f"openai/{stem}-a")
        second: Final = scenario.model(name=f"openai/{stem}-b")
        key: Final = scenario.key(models=["openai/*"])
        assert_served(chat(gateway, key, first))
        assert_served(chat(gateway, key, second))


def test_a_provider_wildcard_on_a_key_refuses_a_group_under_another_provider_prefix(gateway: Gateway) -> None:
    """A deployment is named `anthropic/<unique>` and a key has `models: ["openai/*"]`.

    The request must be refused with the exact 403 `key_model_access_denied` body naming
    `anthropic/<unique>`, and the upstream must not see it.
    """
    with gateway.scenario() as scenario:
        other: Final = scenario.model(name=f"anthropic/{unique_name()}")
        key: Final = scenario.key(models=["openai/*"])
        text: Final = marker()
        assert_model_denied(chat(gateway, key, other, text), other, KEY_DENIED)
        assert upstream_requests(gateway, text) == []


def test_a_name_prefix_wildcard_on_a_key_serves_matching_groups_and_refuses_the_rest(gateway: Gateway) -> None:
    """Deployments `<stem>-mini` and `<other stem>-mini` exist and a key has `models: ["<stem>-*"]`.

    The first must be served and the second refused with 403 `key_model_access_denied`. Patterns are not
    limited to provider prefixes: any name prefix works, which is how teams scope model families such as
    `anthropic/claude-*`.
    """
    with gateway.scenario() as scenario:
        stem: Final = unique_name()
        matching: Final = scenario.model(name=f"{stem}-mini")
        other: Final = scenario.model(name=f"{unique_name()}-mini")
        key: Final = scenario.key(models=[f"{stem}-*"])
        assert_served(chat(gateway, key, matching))
        assert_model_denied(chat(gateway, key, other), other, KEY_DENIED)


def test_a_family_wildcard_like_anthropic_claude_star_matches_only_that_family(gateway: Gateway) -> None:
    """Deployments `anthropic/claude-<unique>` and `anthropic/haiku-<unique>` exist and a key has
    `models: ["anthropic/claude-*"]`.

    The `claude-` deployment must be served and the other refused with 403 `key_model_access_denied`.
    """
    with gateway.scenario() as scenario:
        suffix: Final = unique_name()
        claude: Final = scenario.model(name=f"anthropic/claude-{suffix}")
        other: Final = scenario.model(name=f"anthropic/haiku-{suffix}")
        key: Final = scenario.key(models=["anthropic/claude-*"])
        assert_served(chat(gateway, key, claude))
        assert_model_denied(chat(gateway, key, other), other, KEY_DENIED)


def test_a_wildcard_is_anchored_at_both_ends(gateway: Gateway) -> None:
    """A key has `models: ["<stem>-*"]` and a deployment is named `prefix-<stem>-x`.

    The name contains the pattern's literal part but does not start with it, so the request must be refused
    with 403 `key_model_access_denied` (the regex is `^...$` through `re.match`, not a search).
    """
    with gateway.scenario() as scenario:
        stem: Final = unique_name()
        embedded: Final = scenario.model(name=f"prefix-{stem}-x")
        key: Final = scenario.key(models=[f"{stem}-*"])
        assert_model_denied(chat(gateway, key, embedded), embedded, KEY_DENIED)


def test_a_wildcard_in_the_middle_of_an_entry_matches_any_infix(gateway: Gateway) -> None:
    """A key has `models: ["<stem>-*-mini"]` and deployments `<stem>-fast-mini` and `<stem>-fast-large` exist.

    The first must be served and the second refused with 403 `key_model_access_denied`, because `*` may
    appear anywhere in the entry and the rest of the entry must still match to the end.
    """
    with gateway.scenario() as scenario:
        stem: Final = unique_name()
        mini: Final = scenario.model(name=f"{stem}-fast-mini")
        large: Final = scenario.model(name=f"{stem}-fast-large")
        key: Final = scenario.key(models=[f"{stem}-*-mini"])
        assert_served(chat(gateway, key, mini))
        assert_model_denied(chat(gateway, key, large), large, KEY_DENIED)


def test_a_dot_in_a_wildcard_entry_matches_any_character(gateway: Gateway) -> None:
    """A key has `models: ["<stem>.v1-*"]` and a deployment is named `<stem>Xv1-preview`.

    The request is served with 200 today: only `*` is translated and the dot is left as the regex
    "any character". Pinned as current behaviour; a port that escapes the pattern (the safer reading) changes
    which names a key with a dotted family pattern such as `gpt-4.1-*` can reach, and must do so on purpose.
    """
    with gateway.scenario() as scenario:
        stem: Final = unique_name()
        lookalike: Final = scenario.model(name=f"{stem}Xv1-preview")
        key: Final = scenario.key(models=[f"{stem}.v1-*"])
        assert_served(chat(gateway, key, lookalike))


def test_a_team_wildcard_serves_its_key_and_refuses_with_the_team_denial(gateway: Gateway) -> None:
    """A team has `models: ["<stem>-*"]` and its key has no list of its own.

    `<stem>-a` must be served and `<other>-a` refused with 403 `team_model_access_denied`: team lists
    use the same pattern rules as key lists.
    """
    with gateway.scenario() as scenario:
        stem: Final = unique_name()
        matching: Final = scenario.model(name=f"{stem}-a")
        other: Final = scenario.model(name=f"{unique_name()}-a")
        team: Final = scenario.team(models=[f"{stem}-*"])
        key: Final = scenario.key(team_id=team)
        assert_served(chat(gateway, key, matching))
        assert_model_denied(chat(gateway, key, other), other, TEAM_DENIED)


def test_an_exact_key_entry_inside_a_team_wildcard_narrows_the_key(gateway: Gateway) -> None:
    """A team has `models: ["<stem>-*"]` and its key has `models: ["<stem>-a"]`; `<stem>-b` also exists.

    `<stem>-a` must be served and `<stem>-b` refused with 403 `key_model_access_denied`: the key list is
    checked first and the team's wider pattern does not widen the key.
    """
    with gateway.scenario() as scenario:
        stem: Final = unique_name()
        first: Final = scenario.model(name=f"{stem}-a")
        second: Final = scenario.model(name=f"{stem}-b")
        team: Final = scenario.team(models=[f"{stem}-*"])
        key: Final = scenario.key(team_id=team, models=[first])
        assert_served(chat(gateway, key, first))
        assert_model_denied(chat(gateway, key, second), second, KEY_DENIED)


def test_a_key_wildcard_cannot_exceed_an_exact_team_list(gateway: Gateway) -> None:
    """A team allows only `<stem>-a` and its key has `models: ["<stem>-*"]`; `<stem>-b` also exists.

    `<stem>-b` passes the key pattern but must be refused with 403 `team_model_access_denied`.
    """
    with gateway.scenario() as scenario:
        stem: Final = unique_name()
        first: Final = scenario.model(name=f"{stem}-a")
        second: Final = scenario.model(name=f"{stem}-b")
        team: Final = scenario.team(models=[first])
        key: Final = scenario.key(team_id=team, models=[f"{stem}-*"])
        assert_served(chat(gateway, key, first))
        assert_model_denied(chat(gateway, key, second), second, TEAM_DENIED)


def test_a_user_wildcard_limits_that_users_personal_key(gateway: Gateway) -> None:
    """A user has `models: ["<stem>-*"]` and owns a personal key (no team, no list).

    `<stem>-a` must be served and `<other>-a` refused with 403 `user_model_access_denied`.
    """
    with gateway.scenario() as scenario:
        stem: Final = unique_name()
        matching: Final = scenario.model(name=f"{stem}-a")
        other: Final = scenario.model(name=f"{unique_name()}-a")
        user: Final = scenario.user(models=[f"{stem}-*"])
        key: Final = scenario.key(user_id=user)
        assert_served(chat(gateway, key, matching))
        assert_model_denied(chat(gateway, key, other), other, USER_DENIED)


def test_a_wildcard_denial_names_the_requested_model_and_not_the_pattern(gateway: Gateway) -> None:
    """A key with `models: ["<stem>-*"]` asks for an unrelated deployment.

    The 403 body must be exactly the fixed denial naming the requested model, and the key's pattern must
    not appear anywhere in the response text.
    """
    with gateway.scenario() as scenario:
        stem: Final = unique_name()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[f"{stem}-*"])
        refused: Final = chat(gateway, key, other)
        assert_model_denied(refused, other, KEY_DENIED)
        assert stem not in refused.text


def test_a_wildcard_denial_does_not_depend_on_a_deployment_existing(gateway: Gateway) -> None:
    """A key with `models: ["<stem>-*"]` asks for `<other>-ghost`, a name no deployment serves.

    The answer must be the same 403 `key_model_access_denied` body, not a 400 "invalid model" from the
    router: model access is decided before routing.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key(models=[f"{unique_name()}-*"])
        ghost: Final = f"{unique_name()}-ghost"
        assert_model_denied(chat(gateway, key, ghost), ghost, KEY_DENIED)


def test_a_provider_wildcard_admits_a_bare_name_whose_inferred_provider_matches(gateway: Gateway) -> None:
    """A key has `models: ["openai/*"]` and asks for the bare name `gpt-4o-mini`.

    `get_llm_provider("gpt-4o-mini")` infers `openai`, so the key check rebuilds `openai/gpt-4o-mini`, which
    the pattern matches. The request must not be a model access denial (whatever the router answers for the
    bare name on this proxy). A port that matches only the literal string would refuse it.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key(models=["openai/*"])
        assert_not_model_denied(chat(gateway, key, "gpt-4o-mini"))


def test_a_name_under_an_unknown_namespace_is_never_rebuilt_into_a_provider_wildcard(gateway: Gateway) -> None:
    """A key has `models: ["bedrock/*"]` and asks for `bedrockz/anthropic.claude-<unique>`.

    The name already carries a namespace the provider lookup did not consume, so it must not be rebuilt as
    `bedrock/bedrockz/...`; the answer must be 403 `key_model_access_denied` naming the requested model.
    Losing this guard lets a `bedrock/*` key reach any lookalike namespace.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key(models=["bedrock/*"])
        lookalike: Final = f"bedrockz/anthropic.claude-{unique_name()}"
        assert_model_denied(chat(gateway, key, lookalike), lookalike, KEY_DENIED)


def test_a_pattern_entry_is_not_listed_as_itself_by_default(gateway: Gateway) -> None:
    """A key has `models: ["<stem>-*", <exact model>]` and calls `GET /v1/models` without parameters.

    The exact model must be listed and the literal `<stem>-*` must not be: `_get_wildcard_models` removes
    every pattern entry from the listing unless wildcard routes were requested.
    """
    with gateway.scenario() as scenario:
        stem: Final = unique_name()
        exact: Final = scenario.model()
        key: Final = scenario.key(models=[f"{stem}-*", exact])
        listed: Final = model_ids(gateway, key)
        assert exact in listed and f"{stem}-*" not in listed, listed


def test_return_wildcard_routes_lists_the_pattern_entry_itself(gateway: Gateway) -> None:
    """The same key calls `GET /v1/models?return_wildcard_routes=true`.

    The literal `<stem>-*` must now be listed next to the exact model, which is how UIs show "any model
    starting with" grants.
    """
    with gateway.scenario() as scenario:
        stem: Final = unique_name()
        exact: Final = scenario.model()
        key: Final = scenario.key(models=[f"{stem}-*", exact])
        listed: Final = model_ids(gateway, key, {"return_wildcard_routes": "true"})
        assert f"{stem}-*" in listed and exact in listed, listed


def test_a_name_prefix_pattern_does_not_list_the_concrete_groups_it_grants(gateway: Gateway) -> None:
    """A key has `models: ["<stem>-*"]`, a deployment `<stem>-a` exists, and the key calls `GET /v1/models`.

    `<stem>-a` is callable (see the prefix test above) but is not listed today: a pattern with no
    deployment of its own and no provider prefix expands to no known models. Pinned as current behaviour;
    clients that build pickers from `/v1/models` do not see these groups. UNVERIFIED: expansion goes through
    `_known_models_without_a_deployment`, which may list provider catalog ids for some prefixes.
    """
    with gateway.scenario() as scenario:
        stem: Final = unique_name()
        concrete: Final = scenario.model(name=f"{stem}-a")
        key: Final = scenario.key(models=[f"{stem}-*"])
        assert_served(chat(gateway, key, concrete))
        assert concrete not in model_ids(gateway, key)


@pytest.mark.parametrize("entry", ["*", "all-proxy-models"])
def test_a_team_everything_entry_serves_any_model(gateway: Gateway, entry: str) -> None:
    """A team has `models: ["*"]` or `models: ["all-proxy-models"]` and its key has no list.

    A freshly added deployment must be served with 200: both team entries mean every model on the proxy.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[entry])
        key: Final = scenario.key(team_id=team)
        assert_served(chat(gateway, key, model))
