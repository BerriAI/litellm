"""What each auth object lookup answers when the row it points at is missing, orphaned or only joined in.

Contract, read from the branch:

* A key row (`LiteLLM_VerificationToken`) is read through one `combined_view` query (litellm/proxy/utils.py:6829-6869)
  that LEFT JOINs the team, the membership and its budget, the key's own budget row (`b`), the project, and the
  key's own organization with its budget (`o`, `b2`). Columns from a joined row that is missing come back NULL.
* The key's `budget_id` row fills `max_budget` (and the other limits) only where the key's own column is NULL
  (`LiteLLM_VerificationTokenView.__init__`, litellm/types/proxy/auth/user_api_key_auth.py:113-125).
* A key's `organization_id` becomes `org_id` on the token and wins over the team's organization
  (user_api_key_auth.py:126-127 there, and `organization_max_budget_check`, litellm/proxy/auth/auth_checks.py:7996-8000).
* The user lookup raises `UserNotFoundError` (a `ValueError`) for a missing row; both callers swallow it and
  carry on with no user object (litellm/proxy/auth/user_api_key_auth.py:3243-3258 and 4075-4085 via `_safe_fetch`
  at 3737-3763), so a key whose user row is gone is served without the user's budget.
* A missing project row reads as `None` (auth_checks.py:7937-7938), a missing organization raises
  `OrganizationNotFoundError` which the org budget check treats as "no organization" (auth_checks.py:8006-8016),
  and a missing membership row is cached as `NO_TEAM_MEMBERSHIP_SENTINEL` (auth_checks.py:2777-2784).
* Every foreign key from the key, team and end user rows to organizations, projects and budgets is
  `ON DELETE SET NULL` (litellm/proxy-extras/litellm_proxy_extras/migrations/*), so deleting the parent row with
  SQL orphans nothing; the child simply loses the link. `team_id` and `user_id` on the key have no foreign key.

A Rust store that turns any of these "no row" answers into an error, or that drops one of the joins, changes who
gets served. Rows that a test removes with SQL are created without the usual API cleanup (see
`dec1_dbstores2_support.py`), because that cleanup asserts the row is still there.
"""

from __future__ import annotations

from typing import Final

from tests.integration._support.client import Gateway, held, string_value
from tests.integration.authorization.dec1_dbstores2_support import (
    CACHED_HOLD_SECONDS,
    bare_budget,
    bare_organization,
    bare_project,
    bare_user,
    forget_budget,
    forget_organization,
    name,
    sql,
)
from tests.integration.authorization.dec1_dbstores_support import (
    PEER_SECONDS,
    WRITER_SECONDS,
    assert_budget_refused,
    assert_ok,
    chat_as,
    settled,
)
from tests.integration.authorization.dec1_keys_support import spend_rows


def test_a_personal_key_whose_user_row_was_deleted_by_sql_is_served_without_the_user_budget(gateway: Gateway) -> None:
    """A user with `max_budget: 0` has a personal key that was never used; the user row is then deleted with SQL.

    The first request is served with 200: the user lookup raises `UserNotFoundError`, both callers swallow it
    (user_api_key_auth.py:3243-3258, 4075-4085), and the zero user budget is never checked. A user store that
    reports "no row" as a failure would start refusing (or 503ing) every key whose owner was removed out of band.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = bare_user(gateway, max_budget=0)
        key: Final = scenario.key(user_id=user)
        sql(gateway, 'DELETE FROM "LiteLLM_UserTable" WHERE user_id = %s', (user,))
        assert_ok(chat_as(gateway, key, model))


def test_a_warmed_user_row_deleted_by_sql_keeps_refusing_while_the_user_is_cached(gateway: Gateway) -> None:
    """A user with `max_budget: 0` has a personal key that is refused with 422; the user row is then deleted by SQL.

    For the next few seconds every request must still be refused with 422: the cached user object
    (`user_api_key_cache` under the user id, TTL 60 s, auth_checks.py:3240-3247) stands in for the row, and nothing
    evicts it because no management route ran. A store that re-reads the user on every request would serve it.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = bare_user(gateway, max_budget=0)
        key: Final = scenario.key(user_id=user)
        assert_budget_refused(chat_as(gateway, key, model))
        sql(gateway, 'DELETE FROM "LiteLLM_UserTable" WHERE user_id = %s', (user,))
        held(
            lambda: chat_as(gateway, key, model).status_code, lambda status: status == 422, holding=CACHED_HOLD_SECONDS
        )


def test_a_never_used_key_whose_blocked_project_was_deleted_by_sql_is_served(gateway: Gateway) -> None:
    """A team holds a project created with `blocked: true` and a key bound to it; the project row is deleted by SQL.

    The key's `project_id` is set to NULL by `LiteLLM_VerificationToken_project_id_fkey ON DELETE SET NULL`, so its
    first request is served with 200 and the old block no longer applies. Even if the key still named the project,
    `get_project_object` returns None for a missing row (auth_checks.py:7937-7938) and `_run_project_checks` skips.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        project: Final = bare_project(gateway, team, blocked=True)
        key: Final = scenario.key(team_id=team, project_id=project)
        sql(gateway, 'DELETE FROM "LiteLLM_ProjectTable" WHERE project_id = %s', (project,))
        assert_ok(chat_as(gateway, key, model))


def test_a_warmed_team_row_deleted_by_sql_keeps_serving_while_the_team_is_cached(gateway: Gateway) -> None:
    """A team key is served once (team cached under `team_id:<id>`), then the team row is deleted with SQL.

    Requests keep answering 200 for the next few seconds: the cached team object is returned before any database
    read (auth_checks.py:3862-3870). Round 1 pins that a cold key in the same situation gets 404; together they fix
    when the team store may notice an out of band delete (at the TTL, not before).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = string_value(gateway.post("/team/new", {"team_alias": name("team")})["team_id"])
        key: Final = scenario.key(team_id=team)
        assert_ok(chat_as(gateway, key, model))
        sql(gateway, 'DELETE FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,))
        held(
            lambda: chat_as(gateway, key, model).status_code, lambda status: status == 200, holding=CACHED_HOLD_SECONDS
        )


def test_a_team_key_whose_organization_row_was_deleted_by_sql_is_served_without_the_org_budget(
    gateway: Gateway,
) -> None:
    """An organization with `max_budget: 0` holds a team and a never used team key; the organization row is deleted.

    `LiteLLM_TeamTable_organization_id_fkey ON DELETE SET NULL` clears the team's link, so the key's first request
    is served with 200. Had the team kept the id, `get_org_object` would raise `OrganizationNotFoundError` and
    `organization_max_budget_check` returns without a check (auth_checks.py:8006-8016): a gone organization is
    "no organization", never an error.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        organization, budget = bare_organization(gateway, max_budget=0)
        forget_budget(scenario, budget)
        team: Final = scenario.team(organization_id=organization)
        key: Final = scenario.key(team_id=team)
        sql(gateway, 'DELETE FROM "LiteLLM_OrganizationTable" WHERE organization_id = %s', (organization,))
        assert_ok(chat_as(gateway, key, model))


def test_a_key_bound_directly_to_an_organization_with_a_zero_budget_is_refused(gateway: Gateway) -> None:
    """A key is generated with `organization_id` of an organization whose budget row has `max_budget: 0`, no team.

    The first request is refused with 422 `budget_exceeded`. The key's own organization reaches the token through
    the `combined_view` join `LEFT JOIN "LiteLLM_OrganizationTable" AS o ON v.organization_id = ...` and the
    `organization_id -> org_id` rename (user_api_key_auth.py:126-127), and `organization_max_budget_check` reads
    `valid_token.org_id` first. Round 1 only pinned the organization reached through a team.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        organization, budget = bare_organization(gateway, max_budget=0)
        forget_organization(scenario, organization, budget)
        key: Final = scenario.key(organization_id=organization)
        assert_budget_refused(chat_as(gateway, key, model))


def test_the_spend_row_of_a_key_bound_directly_to_an_organization_names_it(gateway: Gateway) -> None:
    """A key without a team carries `organization_id` of an organization without a budget.

    It is served with 200 and its spend row has `organization_id` equal to that organization. The id comes from the
    key row itself, not from a team, so a key store must keep `organization_id` on the key projection.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        organization, budget = bare_organization(gateway)
        forget_organization(scenario, organization, budget)
        key: Final = scenario.key(organization_id=organization)
        assert_ok(chat_as(gateway, key, model))
        assert spend_rows(gateway, key)[0]["organization_id"] == organization


def test_a_never_used_key_whose_organization_row_was_deleted_by_sql_is_served(gateway: Gateway) -> None:
    """A key bound directly to an organization with `max_budget: 0` was never used; the organization row is deleted.

    `LiteLLM_VerificationToken_organization_id_fkey ON DELETE SET NULL` clears the key's link and the `o`/`b2`
    joins find nothing, so the first request is served with 200.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        organization, budget = bare_organization(gateway, max_budget=0)
        forget_budget(scenario, budget)
        key: Final = scenario.key(organization_id=organization)
        sql(gateway, 'DELETE FROM "LiteLLM_OrganizationTable" WHERE organization_id = %s', (organization,))
        assert_ok(chat_as(gateway, key, model))


def test_a_key_linked_to_a_zero_budget_row_is_refused(gateway: Gateway) -> None:
    """A budget row with `max_budget: 0` is created with `/budget/new` and a key is generated with its `budget_id`.

    The key sets no `max_budget` of its own, so the view copies `litellm_budget_table_max_budget` into `max_budget`
    (litellm/types/proxy/auth/user_api_key_auth.py:113-125) and the first request is refused with 422
    `budget_exceeded`. A key store that drops the `b` join loses every budget attached by id.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        budget: Final = scenario.budget(max_budget=0)
        key: Final = scenario.key(budget_id=budget)
        assert_budget_refused(chat_as(gateway, key, model))


def test_a_keys_own_max_budget_wins_over_its_budget_rows_max_budget(gateway: Gateway) -> None:
    """A key has `max_budget: 100` of its own and `budget_id` naming a budget row with `max_budget: 0`.

    The request is served with 200: the budget row fills a limit only where the key's own column is NULL
    (`current is None` in `LiteLLM_VerificationTokenView.__init__`). A merge that lets the joined row win would
    refuse it.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        budget: Final = scenario.budget(max_budget=0)
        key: Final = scenario.key(budget_id=budget, max_budget=100)
        assert_ok(chat_as(gateway, key, model))


def test_a_never_used_key_whose_budget_row_was_deleted_by_sql_is_served(gateway: Gateway) -> None:
    """A key linked to a budget row with `max_budget: 0` was never used; the budget row is deleted with SQL.

    `LiteLLM_VerificationToken_budget_id_fkey ON DELETE SET NULL` clears the link and the first request is served
    with 200: no joined budget, no limit.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        budget: Final = bare_budget(gateway, max_budget=0)
        key: Final = scenario.key(budget_id=budget)
        sql(gateway, 'DELETE FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (budget,))
        assert_ok(chat_as(gateway, key, model))


def test_a_member_whose_membership_row_was_deleted_by_sql_is_served_without_the_member_budget(
    gateway: Gateway,
) -> None:
    """A member has `max_budget_in_team: 0` and a never used team key; their `LiteLLM_TeamMembership` row is deleted.

    The first request is served with 200: the membership lookup finds no row, caches
    `NO_TEAM_MEMBERSHIP_SENTINEL` (auth_checks.py:2777-2784) and no member budget applies. The team's
    `members_with_roles` still lists the user; the membership store must not fall back to it or treat the
    missing row as an error.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        member: Final = scenario.member(team)
        gateway.post("/team/member_update", {"team_id": team, "user_id": member, "max_budget_in_team": 0})
        key: Final = scenario.key(team_id=team, user_id=member)
        sql(
            gateway,
            'DELETE FROM "LiteLLM_TeamMembership" WHERE user_id = %s AND team_id = %s',
            (member, team),
        )
        assert_ok(chat_as(gateway, key, model))


def test_a_member_added_after_their_key_was_used_gets_the_member_budget_on_both_proxies(
    gateway: Gateway, peer: Gateway
) -> None:
    """A proxy admin creates a team key for a user who is not a member; it is served on both proxies, so both cache
    the "no membership" sentinel. The user is then added with `/team/member_add` and given `max_budget_in_team: 0`.

    The key must be refused with 422 on the writer within 5 seconds and on the peer within 10 seconds. The negative
    answer is cached for the management TTL (auth_checks.py:2779-2784) and must be evicted and broadcast like a
    positive one, or a newly limited member keeps spending for 60 seconds.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        user: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(team_id=team, user_id=user)
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        gateway.post("/team/member_add", {"team_id": team, "member": {"role": "user", "user_id": user}})
        gateway.post("/team/member_update", {"team_id": team, "user_id": user, "max_budget_in_team": 0})
        assert_budget_refused(settled(lambda: chat_as(gateway, key, model), 422, WRITER_SECONDS))
        assert_budget_refused(settled(lambda: chat_as(peer, key, model), 422, PEER_SECONDS))


def test_a_customer_whose_budget_row_was_deleted_by_sql_is_served(gateway: Gateway) -> None:
    """A customer is linked to a budget row with `max_budget: 0`; the budget row is deleted with SQL before the
    customer's first request.

    `LiteLLM_EndUserTable_budget_id_fkey ON DELETE SET NULL` clears the link, so the request naming the customer as
    body `user` is served with 200. The end user lookup reads the row with `include={"litellm_budget_table": True}`
    and a NULL relation means no budget.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        budget: Final = bare_budget(gateway, max_budget=0)
        customer_id: Final = name("customer")
        gateway.post("/customer/new", {"user_id": customer_id, "budget_id": budget})
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [customer_id]})
        sql(gateway, 'DELETE FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (budget,))
        assert_ok(chat_as(gateway, key, model, user=customer_id))
