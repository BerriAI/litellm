"""
Regression tests for `_build_key_filter_conditions` (issue #32062).

Explicit `user_id` / `key_alias` query filters must narrow the result set
across ALL visibility branches (own keys, admin-team keys, member service
accounts) instead of being unioned away by an unfiltered team-visibility
branch. At the same time, the implicit self-scope (`user_id` defaulted for a
non-admin) must NOT be applied as a filter, or a team admin who omits
`user_id` would be wrongly restricted to their own keys.
"""

from litellm.proxy.management_endpoints.key_management_endpoints import (
    _build_key_filter_conditions,
)


def _all_and_clauses(where):
    """Flatten every clause combined via (possibly nested) AND.

    Each global filter wraps the prior ``where`` in a new ``{"AND": [...]}``,
    so filters end up at different nesting depths but are all AND-combined.
    Nested ORs are treated as opaque leaves (not descended into), so a
    ``user_id`` that only appears inside a visibility OR-branch is NOT counted
    as a top-level filter.
    """
    out = []

    def walk(node):
        if isinstance(node, dict) and isinstance(node.get("AND"), list):
            for clause in node["AND"]:
                walk(clause)
        else:
            out.append(node)

    walk(where)
    return out


def test_explicit_user_id_narrows_across_team_visibility():
    # Caller has team-wide visibility (admin_team_ids) AND explicitly filters by
    # user_id -> the filter must be AND'd globally so only that user's keys show.
    where = _build_key_filter_conditions(
        user_id="alice",
        team_id=None,
        organization_id=None,
        key_alias=None,
        key_hash=None,
        exclude_team_id=None,
        admin_team_ids=["team-1"],
        apply_user_id_filter=True,
    )
    assert {"user_id": "alice"} in _all_and_clauses(where)


def test_explicit_key_alias_narrows_across_team_visibility():
    where = _build_key_filter_conditions(
        user_id=None,
        team_id=None,
        organization_id=None,
        key_alias="my-alias",
        key_hash=None,
        exclude_team_id=None,
        admin_team_ids=["team-1"],
    )
    assert {"key_alias": "my-alias"} in _all_and_clauses(where)


def test_implicit_self_scope_does_not_filter_team_keys():
    # Team admin omitted user_id -> it was defaulted to self (apply_user_id_filter
    # is False). It must NOT be applied as a global AND, otherwise the admin would
    # only see their own keys instead of ALL team keys.
    where = _build_key_filter_conditions(
        user_id="admin-user",
        team_id=None,
        organization_id=None,
        key_alias=None,
        key_hash=None,
        exclude_team_id=None,
        admin_team_ids=["team-1"],
        apply_user_id_filter=False,
    )
    assert {"user_id": "admin-user"} not in _all_and_clauses(where)


def test_substring_matching_uses_contains_for_global_filters():
    where = _build_key_filter_conditions(
        user_id="ali",
        team_id=None,
        organization_id=None,
        key_alias="ali-key",
        key_hash=None,
        exclude_team_id=None,
        admin_team_ids=["team-1"],
        apply_user_id_filter=True,
        use_substring_matching=True,
    )
    clauses = _all_and_clauses(where)
    assert {"key_alias": {"contains": "ali-key", "mode": "insensitive"}} in clauses
    assert {"user_id": {"contains": "ali", "mode": "insensitive"}} in clauses
