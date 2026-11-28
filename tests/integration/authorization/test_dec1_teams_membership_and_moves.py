"""How a key's team and its owner's team membership shape what the key may do, and what moving a key changes.

Contract, read from litellm/proxy/management_endpoints/key_management_endpoints.py (`key_generation_check`,
`_team_key_generation_check`, `_team_key_operation_team_member_check`, `validate_key_team_change`) and
litellm/proxy/auth/user_api_key_auth.py (team object resync, `team_metadata` from the fresh team lookup):

* A proxy admin may create a team key for a user who is not a member of the team: `_team_key_generation_check`
  returns before the membership check for `proxy_admin`. That key is served and its spend is attributed to both the
  user and the team. A Rust port of the membership store that starts requiring a membership row at request time
  would refuse these keys.
* A non admin caller who is not on the team is refused with 400 `User=<id> not assigned to team=<team>`.
* `/key/update` with a new `team_id` refuses with 403 when the key's `user_id` is not a member of the new team.
* After a key moves teams, every team derived effect follows the new team on the next request: the spend row's
  `team_id`, team spend, team tags, team metadata gates and the team block. The key row's `team_id` is the only
  link, so a store that keeps a stale team copy keyed by the key hash breaks all of these at once.

Uses the suite `gateway`. Models come from `scenario.model()`. Not repeated here: model list checks on a move
(`test_dec1_modelaccess_hierarchy.py`), rate limit windows on a move (`test_dec1_ratelim_owner_updates.py`),
the admin and guard matrix of `/key/update` team changes (`management/test_key_team_change.py`), member budgets
(`test_dec1_dbstores_team_org_and_key_lookups.py`).
"""

from __future__ import annotations

import uuid
from typing import Final

from tests.integration._support.client import Gateway, eventually, list_value
from tests.integration.authorization.dec1_teams_support import (
    UPSTREAM_CONTENT,
    WRITER_SECONDS,
    answer_text,
    chat_as,
    error_of,
    marker,
    settled,
    spend_rows_of,
    team_blocked_message,
    team_row,
)


def _team_spend(gateway: Gateway, team_id: str) -> float:
    spend: Final = team_row(gateway, team_id)["spend"]
    assert isinstance(spend, (int, float)), spend
    return float(spend)


def test_a_proxy_admin_may_create_a_team_key_for_a_user_outside_the_team(gateway: Gateway) -> None:
    """The master key calls `/key/generate` with `team_id: <team>` and `user_id: <user>` where the user exists but
    was never added to the team.

    The answer is 200 and the key row stores both ids; no membership row is created as a side effect.
    `_team_key_generation_check` returns True for a proxy admin before `_team_key_operation_team_member_check`.
    """
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        user: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(team_id=team, user_id=user)
        info = gateway.get("/key/info", {"key": key})["info"]
        assert isinstance(info, dict)
        assert (info["team_id"], info["user_id"]) == (team, user), info
        assert (
            gateway.rows('SELECT user_id FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s', (team, user))
            == []
        )


def test_a_team_key_owned_by_a_non_member_is_served_and_billed_to_both(gateway: Gateway) -> None:
    """The key from the previous test (team key, owner not a member) calls chat.

    The request is served with 200 and the spend row carries `"user": <user>` and `team_id: <team>`. The membership
    lookup answers "no membership" and nothing on the request path requires one, so a membership store that treats a
    missing row as an error would refuse this key.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        user: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(team_id=team, user_id=user)
        answer_text(chat_as(gateway, key, model))
        row: Final = spend_rows_of(gateway, key)[0]
        assert (row["user"], row["team_id"]) == (user, team), row


def test_a_team_key_without_a_user_is_served_and_billed_to_the_team_only(gateway: Gateway) -> None:
    """A team key is created with no `user_id` and calls chat.

    The request is served; the spend row's `team_id` is the team and its `"user"` column is empty (no member, no
    membership lookup: `_set_team_member_model_max_budget` returns early when `user_id` is not a string).
    UNVERIFIED: the empty user column is written as `""`; adjust to NULL if the spend writer stores NULL.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        answer_text(chat_as(gateway, key, model))
        row: Final = spend_rows_of(gateway, key)[0]
        assert row["team_id"] == team, row
        assert row["user"] in ("", None), row


def test_an_internal_user_outside_a_team_cannot_create_a_key_in_it(gateway: Gateway) -> None:
    """An internal user who is not on the team uses their own personal key to call `/key/generate` with
    `team_id: <team>`.

    The answer is 400 with `error.message == "User=<user> not assigned to team=<team>"` and no key row for the team
    is created (`_team_key_operation_team_member_check`).
    """
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        user: Final = scenario.user(user_role="internal_user")
        own_key: Final = scenario.key(user_id=user)
        refused: Final = gateway.request("POST", "/key/generate", {"team_id": team, "user_id": user}, key=own_key)
        assert refused.status_code == 400, refused.text
        assert error_of(refused)["message"] == f"User={user} not assigned to team={team}", refused.text
        assert gateway.rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE team_id=%s', (team,)) == []


def test_a_team_admin_creates_a_team_key_for_themselves(gateway: Gateway) -> None:
    """A user with team role `admin` uses their personal key to call `/key/generate` with `team_id` and their own
    `user_id`.

    The answer is 200 and the new key is served on chat: the default `team_key_generation` allows roles `admin` and
    `user`, and a team admin passes the member permission check for `/key/generate`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        admin: Final = scenario.member(team, role="admin")
        own_key: Final = scenario.key(user_id=admin)
        created: Final = gateway.request("POST", "/key/generate", {"team_id": team, "user_id": admin}, key=own_key)
        assert created.status_code == 200, created.text
        team_key: Final = created.json()["key"]
        scenario.cleanups.callback(scenario.delete_key, team_key)
        answer_text(chat_as(gateway, team_key, model))


def test_moving_a_key_to_a_team_its_owner_is_not_in_is_refused_with_403(gateway: Gateway) -> None:
    """A key owned by a member of team one is moved with `/key/update {"team_id": <team two>}` by the master key;
    the owner is not on team two.

    The answer is 403 with message `User=<user> is not a member of the team=<team two>. Check team members via
    `/team/info`.` and the key row still names team one (`validate_key_team_change`).
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.team()
        second: Final = scenario.team()
        user: Final = scenario.member(first)
        key: Final = scenario.key(team_id=first, user_id=user)
        refused: Final = gateway.request("POST", "/key/update", {"key": key, "team_id": second})
        assert refused.status_code == 403, refused.text
        assert error_of(refused)["message"] == (
            f"User={user} is not a member of the team={second}. Check team members via `/team/info`."
        ), refused.text
        info = gateway.get("/key/info", {"key": key})["info"]
        assert isinstance(info, dict) and info["team_id"] == first, info


def test_after_a_move_the_spend_row_names_the_new_team(gateway: Gateway) -> None:
    """A key without a user is served once in team one, then moved to team two, then served again.

    The first spend row names team one and the second names team two. The key row's `team_id` is the only link, so
    a cached key object that keeps the old team id after `/key/update` would bill the wrong team.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        first: Final = scenario.team()
        second: Final = scenario.team()
        key: Final = scenario.key(team_id=first)
        answer_text(chat_as(gateway, key, model))
        gateway.post("/key/update", {"key": key, "team_id": second})
        answer_text(chat_as(gateway, key, model))
        rows: Final = spend_rows_of(gateway, key, at_least=2)
        assert [row["team_id"] for row in rows] == [first, second], rows


def test_after_a_move_spend_accrues_to_the_new_team_only(gateway: Gateway) -> None:
    """A key is moved from team one to team two before it is ever used, then serves one chat completion.

    Team two's `spend` column must rise above 0 and team one's must stay exactly 0.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        first: Final = scenario.team()
        second: Final = scenario.team()
        key: Final = scenario.key(team_id=first)
        gateway.post("/key/update", {"key": key, "team_id": second})
        answer_text(chat_as(gateway, key, model))
        eventually(lambda: _team_spend(gateway, second), lambda spend: spend > 0, seconds=70)
        assert _team_spend(gateway, first) == 0


def test_after_a_move_the_new_teams_tags_replace_the_old_ones(gateway: Gateway) -> None:
    """Team one has `metadata.tags: [<one>]` and team two `[<two>]`; a key is served in team one, moved to team two
    and served again.

    The second spend row's `request_tags` contains `<two>` and not `<one>`.
    """
    one: Final = f"one-{marker()}"
    two: Final = f"two-{marker()}"
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        first: Final = scenario.team(metadata={"tags": [one]})
        second: Final = scenario.team(metadata={"tags": [two]})
        key: Final = scenario.key(team_id=first)
        answer_text(chat_as(gateway, key, model))
        gateway.post("/key/update", {"key": key, "team_id": second})
        answer_text(chat_as(gateway, key, model))
        tags: Final = list_value(spend_rows_of(gateway, key, at_least=2)[-1]["request_tags"])
        assert two in tags and one not in tags, tags


def test_moving_a_key_out_of_a_blocked_team_serves_it_again(gateway: Gateway) -> None:
    """Team one is blocked with `/team/block` and its key is refused with the team blocked 401; the master key then
    moves the key to unblocked team two.

    The key is served with 200 within 5 seconds: the block belongs to the team row, not the key.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        first: Final = scenario.team()
        second: Final = scenario.team()
        key: Final = scenario.key(team_id=first)
        gateway.post("/team/block", {"team_id": first})
        refused: Final = settled(lambda: chat_as(gateway, key, model), 401)
        assert error_of(refused)["message"] == team_blocked_message(first), refused.text
        gateway.post("/key/update", {"key": key, "team_id": second})
        settled(lambda: chat_as(gateway, key, model), 200, WRITER_SECONDS)


def test_moving_a_warmed_key_into_a_blocked_team_refuses_it_naming_that_team(gateway: Gateway) -> None:
    """A key is served in team one; team two is blocked; the master key moves the key to team two.

    Within 5 seconds the key's chat is refused with 401 and `Authentication Error, Team=<team two> is blocked. ...`.
    UNVERIFIED: `validate_key_team_change` does not check the target team's `blocked` flag, so the move itself is
    expected to succeed; if a port refuses the move instead, assert that refusal here.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        first: Final = scenario.team()
        second: Final = scenario.team()
        key: Final = scenario.key(team_id=first)
        answer_text(chat_as(gateway, key, model))
        gateway.post("/team/block", {"team_id": second})
        gateway.post("/key/update", {"key": key, "team_id": second})
        refused: Final = settled(lambda: chat_as(gateway, key, model), 401)
        assert error_of(refused)["message"] == team_blocked_message(second), refused.text


def test_moving_a_key_into_a_team_that_requires_a_trace_id_applies_the_gate(gateway: Gateway) -> None:
    """A key is served in team one; team two has `metadata.require_trace_id: true`; the key is moved to team two.

    Within 5 seconds a request without a trace id answers 400 with `param: "trace_id"`, and a request with an
    `x-litellm-trace-id` header is served: the new team's metadata is read through the team lookup.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        first: Final = scenario.team()
        second: Final = scenario.team(metadata={"require_trace_id": True})
        key: Final = scenario.key(team_id=first)
        answer_text(chat_as(gateway, key, model))
        gateway.post("/key/update", {"key": key, "team_id": second})
        refused: Final = settled(lambda: chat_as(gateway, key, model), 400)
        assert error_of(refused)["param"] == "trace_id", refused.text
        answer_text(chat_as(gateway, key, model, headers={"x-litellm-trace-id": str(uuid.uuid4())}))


def test_a_member_key_keeps_working_after_the_member_is_promoted_to_team_admin(gateway: Gateway) -> None:
    """A member with team role `user` has a team key that is served; `/team/member_update` sets `role: "admin"`.

    The same key keeps being served on chat: the team role changes management permissions only, never whether an
    LLM route is admitted. A membership store that changes the role column type would show up here as a 4xx.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        user: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=user)
        answer_text(chat_as(gateway, key, model))
        gateway.post("/team/member_update", {"team_id": team, "user_id": user, "role": "admin"})
        assert answer_text(chat_as(gateway, key, model)) == UPSTREAM_CONTENT


def test_a_team_admin_key_and_a_member_key_get_the_same_llm_answer(gateway: Gateway) -> None:
    """A team has one `admin` member and one `user` member, each with a team key; both call the same model.

    Both answers are 200 with the upstream's canned content and both spend rows name the team; team role does not
    change routing or attribution on LLM routes.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        admin_key: Final = scenario.key(team_id=team, user_id=scenario.member(team, role="admin"))
        member_key: Final = scenario.key(team_id=team, user_id=scenario.member(team))
        assert answer_text(chat_as(gateway, admin_key, model)) == answer_text(chat_as(gateway, member_key, model))
        for key in (admin_key, member_key):
            assert spend_rows_of(gateway, key)[0]["team_id"] == team
