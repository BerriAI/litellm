"""Which fields of the team, team membership, organization and key rows take effect on a request, and how fast.

Contract, read from litellm/proxy/auth/auth_checks.py (`common_checks`, `get_team_object`,
`_get_team_object_from_user_api_key_cache`, `get_org_object`, `organization_max_budget_check`) and
litellm/proxy/auth/user_api_key_auth.py (`run_centralized_common_checks`, `_token_can_vouch_for_team`):

* The key row (`combined_view`) carries a copy of its team's fields, but every request also reads the team row
  itself through `get_team_object`, cache first. Team `metadata.require_trace_id`, `model_aliases` (the joined
  `LiteLLM_ModelTable` row) and the membership row (`LiteLLM_TeamMembership` joined to its budget) all come from
  those lookups, which draft PR #52176 moves to `RustTeamStore`, `RustTeamMembershipStore` and
  `RustOrganizationStore`.
* A team row that the database answers is gone raises `TeamNotFoundError` (404) and nothing may stand in for it.
* The organization is found through the key's own `organization_id` or the team's, and its budget row
  (`include_budget_table=True`) is cached under `org_id:<id>:with_budget`.
* Management writes evict the cached objects on the proxy that handled the write and broadcast the eviction on
  `litellm_proxy.auth_cache_invalidation`, so a warmed peer drops its copy too. The tests require the writer within
  5 seconds and the peer within 10, against a 60 second cache TTL.
* Keys can be addressed by the raw `sk-` value or by its SHA-256 on every management route, and both forms
  evict the same cache entry.

Every test uses `gateway` and, where two instances matter, `peer` (shares Postgres and Redis). Models come from
`scenario.model()`, served by the scripted upstream. Not duplicated here: key, team, member and user model list
narrowing on warmed keys and team deletion (`test_dec1_keys_cache_staleness.py`), team block precedence
(`test_dec1_keys_check_order.py`), team and key budget arithmetic (`test_dec1_budgets_*`).
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway, string_value
from tests.integration._support.database import write_rows
from tests.integration.authorization.dec1_dbstores_support import (
    DEFAULT_AUTH_CACHE_TTL_SECONDS,
    PEER_SECONDS,
    WRITER_SECONDS,
    assert_budget_refused,
    assert_ok,
    chat_as,
    settled,
)
from tests.integration.authorization.dec1_keys_support import (
    UNKNOWN_KEY_MESSAGE,
    assert_denied,
    digest,
    error_of,
    spend_rows,
)

_TRACE_ID_HEADER: Final = "x-litellm-trace-id"


def _trace_id_message(team_id: str) -> str:
    return (
        f"Team '{team_id}' requires a trace ID on every LLM, MCP and agent request. "
        "Send an x-litellm-trace-id header. "
        "LLM requests can also send a W3C traceparent header or set metadata.trace_id in the body."
    )


def _assert_trace_id_refused(response_status: int, error: dict[str, object], team_id: str) -> None:
    assert response_status == 400, error
    assert error["type"] == "bad_request_error", error
    assert error["param"] == "trace_id", error
    assert error["message"] == _trace_id_message(team_id), error


def test_team_metadata_require_trace_id_refuses_a_request_without_a_trace_id(gateway: Gateway) -> None:
    """A team is created with `metadata: {"require_trace_id": true}` and a key in it calls chat without any trace id.

    The answer must be 400 `bad_request_error` with `param: "trace_id"` and the exact message naming the team,
    and nothing reaches the upstream. The flag is read from the team row's `metadata` on every request, so a
    store that drops or mistypes the JSON column silently turns the gate off.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"require_trace_id": True})
        key: Final = scenario.key(team_id=team)
        refused: Final = chat_as(gateway, key, model)
        _assert_trace_id_refused(refused.status_code, error_of(refused), team)


def test_team_metadata_require_trace_id_is_satisfied_by_the_trace_id_header(gateway: Gateway) -> None:
    """The same team with `require_trace_id: true`; the key sends `x-litellm-trace-id: <uuid>`.

    The request is served with 200. Only the exact boolean `true` turns the gate on (`is True`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"require_trace_id": True})
        key: Final = scenario.key(team_id=team)
        assert_ok(chat_as(gateway, key, model, headers={_TRACE_ID_HEADER: str(uuid.uuid4())}))


def test_a_string_true_require_trace_id_does_not_turn_the_gate_on(gateway: Gateway) -> None:
    """A team stores `metadata: {"require_trace_id": "true"}` (a string, not a boolean).

    A key in it is served without a trace id, because `_team_metadata_requires_trace_id` compares with `is True`.
    A store that coerces JSON strings to booleans while decoding the metadata column would start refusing.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"require_trace_id": "true"})
        key: Final = scenario.key(team_id=team)
        assert_ok(chat_as(gateway, key, model))


def test_turning_on_require_trace_id_reaches_a_warmed_team_key_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A team key is served on both proxies, then `/team/update` sets `metadata: {"require_trace_id": true}`.

    A request without a trace id must get the 400 trace id refusal on the writer within 5 seconds and on the peer
    within 10 seconds. The team object is evicted and the eviction broadcast; the key's own cached copy of the
    team fields must not shadow the new team metadata (`valid_token.team_metadata = _team_obj.metadata`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        gateway.post("/team/update", {"team_id": team, "metadata": {"require_trace_id": True}})
        for proxy, seconds in ((gateway, WRITER_SECONDS), (peer, PEER_SECONDS)):
            refused = settled(lambda proxy=proxy: chat_as(proxy, key, model), 400, seconds)
            _assert_trace_id_refused(refused.status_code, error_of(refused), team)


def test_a_team_model_alias_added_after_warming_is_served_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A team allowed one model has a key that is served on both proxies; an alias name is not served yet.

    `/team/update` then sets `model_aliases: {<alias>: <model>}`. The alias must be served with 200 on the writer
    within 5 seconds and on the peer within 10 seconds. The aliases live in a separate `LiteLLM_ModelTable` row
    that the team lookup joins (`find_team.sql` LEFT JOIN on `model_id`); a store that loses the join loses
    every team alias.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        alias: Final = f"dec1-dbstores-alias-{uuid.uuid4().hex[:8]}"
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team)
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
            assert chat_as(proxy, key, alias).status_code != 200
        gateway.post("/team/update", {"team_id": team, "model_aliases": {alias: model}})
        assert_ok(settled(lambda: chat_as(gateway, key, alias), 200, WRITER_SECONDS))
        assert_ok(settled(lambda: chat_as(peer, key, alias), 200, PEER_SECONDS))


def test_a_team_created_with_model_aliases_serves_the_alias_on_the_first_request(gateway: Gateway) -> None:
    """A team is created with `models: [<model>]` and `model_aliases: {<alias>: <model>}`; its key was never used.

    The very first request for the alias is served with 200: the cold team lookup must return the aliases row.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        alias: Final = f"dec1-dbstores-alias-{uuid.uuid4().hex[:8]}"
        team: Final = scenario.team(models=[model], model_aliases={alias: model})
        key: Final = scenario.key(team_id=team)
        assert_ok(chat_as(gateway, key, alias))


def test_a_key_whose_team_row_is_gone_is_refused_with_404_naming_the_team(gateway: Gateway) -> None:
    """A team and a key in it are created; the key is never used, then the team row is deleted with SQL.

    The first request with the key must not be served: `get_team_object` gets an answer of "no row" from the
    database, raises `TeamNotFoundError`, and `_token_can_vouch_for_team` refuses to stand the key's copy of the
    team in for it. Expected 404 whose body contains
    "Team doesn't exist in db. Team=<team_id>. Create team via `/team/new` call.".

    UNVERIFIED: the error envelope. The detail is a dict `{"error": ...}` passed through
    `_as_proxy_exception`, so only the status and the sentence are asserted.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = string_value(
            gateway.post("/team/new", {"team_alias": f"dec1-dbstores-{uuid.uuid4().hex}"})["team_id"]
        )
        key: Final = scenario.key(team_id=team)
        write_rows('DELETE FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,))
        refused: Final = chat_as(gateway, key, model)
        assert refused.status_code == 404, refused.text
        assert f"Team doesn't exist in db. Team={team}. Create team via `/team/new` call." in refused.text


def test_a_members_budget_in_team_of_zero_refuses_their_team_key(gateway: Gateway) -> None:
    """A team has a member whose `/team/member_update` set `max_budget_in_team: 0`; the member's team key calls chat.

    The answer is a 422 `budget_exceeded`. The member budget lives in a `LiteLLM_BudgetTable` row linked from the
    membership row (`budget_id`), so this proves the membership lookup joins its budget.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        member: Final = scenario.member(team)
        gateway.post("/team/member_update", {"team_id": team, "user_id": member, "max_budget_in_team": 0})
        key: Final = scenario.key(team_id=team, user_id=member)
        assert_budget_refused(chat_as(gateway, key, model))


def test_a_members_budget_does_not_apply_to_another_member_of_the_same_team(gateway: Gateway) -> None:
    """Two members of one team; only the first has `max_budget_in_team: 0`.

    The second member's team key is served with 200 while the first member's key is refused with 422. The
    membership lookup is keyed by `(user_id, team_id)`; a store keyed by team alone would mix the two.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        limited: Final = scenario.member(team)
        free: Final = scenario.member(team)
        gateway.post("/team/member_update", {"team_id": team, "user_id": limited, "max_budget_in_team": 0})
        limited_key: Final = scenario.key(team_id=team, user_id=limited)
        free_key: Final = scenario.key(team_id=team, user_id=free)
        assert_budget_refused(chat_as(gateway, limited_key, model))
        assert_ok(chat_as(gateway, free_key, model))


def test_a_members_budget_does_not_follow_them_to_another_team(gateway: Gateway) -> None:
    """A user is a member of two teams and has `max_budget_in_team: 0` only in the first.

    Their key in the second team is served with 200; their key in the first is refused with 422.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        first: Final = scenario.team()
        second: Final = scenario.team()
        user: Final = scenario.member(first)
        gateway.post("/team/member_add", {"team_id": second, "member": {"role": "user", "user_id": user}})
        gateway.post("/team/member_update", {"team_id": first, "user_id": user, "max_budget_in_team": 0})
        assert_budget_refused(chat_as(gateway, scenario.key(team_id=first, user_id=user), model))
        assert_ok(chat_as(gateway, scenario.key(team_id=second, user_id=user), model))


def test_raising_a_members_budget_in_team_readmits_their_warmed_key_on_both_proxies(
    gateway: Gateway, peer: Gateway
) -> None:
    """A member with `max_budget_in_team: 0` is refused with 422 on both proxies, then `/team/member_update` sets
    `max_budget_in_team: 100`.

    The key must be served on the writer within 5 seconds and on the peer within 10 seconds. The membership is
    cached under two keys (`<team>_<user>` for admission and `team_membership:<user>:<team>` for the reservation,
    see `team_membership_reservation_cache_key`); both must be invalidated.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        member: Final = scenario.member(team)
        gateway.post("/team/member_update", {"team_id": team, "user_id": member, "max_budget_in_team": 0})
        key: Final = scenario.key(team_id=team, user_id=member)
        for proxy in (gateway, peer):
            assert_budget_refused(chat_as(proxy, key, model))
        gateway.post("/team/member_update", {"team_id": team, "user_id": member, "max_budget_in_team": 100})
        assert_ok(settled(lambda: chat_as(gateway, key, model), 200, WRITER_SECONDS))
        assert_ok(settled(lambda: chat_as(peer, key, model), 200, PEER_SECONDS))


def test_setting_a_members_budget_to_zero_refuses_their_warmed_key_on_both_proxies(
    gateway: Gateway, peer: Gateway
) -> None:
    """A member's team key is served on both proxies, then `/team/member_update` sets `max_budget_in_team: 0`.

    Both proxies must answer 422 `budget_exceeded` within 5 and 10 seconds. A stale grant here lets a member
    spend past a budget an admin just set.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        member: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=member)
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        gateway.post("/team/member_update", {"team_id": team, "user_id": member, "max_budget_in_team": 0})
        assert_budget_refused(settled(lambda: chat_as(gateway, key, model), 422, WRITER_SECONDS))
        assert_budget_refused(settled(lambda: chat_as(peer, key, model), 422, PEER_SECONDS))


def test_removing_a_member_stops_their_warmed_team_key_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A member's team key is served on both proxies, then `/team/member_delete` removes the member.

    `team_member_delete` deletes the member's keys in that team in the same transaction, so both proxies must
    answer 401 within 5 and 10 seconds, and the key row is gone from `LiteLLM_VerificationToken`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        member: Final = scenario.member(team)
        key: Final = string_value(gateway.post("/key/generate", {"team_id": team, "user_id": member})["key"])
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        gateway.post("/team/member_delete", {"team_id": team, "user_id": member})
        settled(lambda: chat_as(gateway, key, model), 401, WRITER_SECONDS)
        settled(lambda: chat_as(peer, key, model), 401, PEER_SECONDS)
        assert gateway.rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest(key),)) == []


def test_a_team_in_an_organization_with_a_zero_budget_refuses_its_keys(gateway: Gateway) -> None:
    """An organization is created with `max_budget: 0`, a team inside it, and a key in the team that names no
    organization itself.

    The key's request is refused with 422 `budget_exceeded`: `organization_max_budget_check` falls back to the
    team's `organization_id` and reads the organization joined to its budget row.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        organization: Final = scenario.organization(max_budget=0)
        team: Final = scenario.team(organization_id=organization)
        key: Final = scenario.key(team_id=team)
        assert_budget_refused(chat_as(gateway, key, model))


def test_a_team_in_an_organization_without_a_budget_is_served(gateway: Gateway) -> None:
    """An organization created without `max_budget` holds a team and a key.

    The key is served with 200: an organization budget row whose `max_budget` is null restricts nothing.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        organization: Final = scenario.organization()
        team: Final = scenario.team(organization_id=organization)
        assert_ok(chat_as(gateway, scenario.key(team_id=team), model))


def test_the_spend_row_carries_the_organization_reached_through_the_team(gateway: Gateway) -> None:
    """A key with only `team_id` belongs to a team inside an organization.

    The key's spend row carries the organization in `organization_id`. `_inherit_org_identity` copies
    `team_object.organization_id` onto the auth object when the key has none.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        organization: Final = scenario.organization()
        team: Final = scenario.team(organization_id=organization)
        key: Final = scenario.key(team_id=team)
        assert_ok(chat_as(gateway, key, model))
        assert [row["organization_id"] for row in spend_rows(gateway, key)] == [organization]


def test_lowering_an_organizations_budget_reaches_a_warmed_team_key_on_both_proxies(
    gateway: Gateway, peer: Gateway
) -> None:
    """A team key inside an organization with `max_budget: 100` is served on both proxies, then
    `/organization/update` sets `max_budget: 0`.

    Intended: both proxies refuse with 422 within 5 and 10 seconds, like every other owner object.
    """
    pytest.skip(
        "BUG: /organization/update writes the budget row but evicts neither `org_id:<id>:with_budget` nor the "
        "last known organization, so a warmed proxy keeps the old organization budget until the 60 s TTL"
    )
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        organization: Final = scenario.organization(max_budget=100)
        team: Final = scenario.team(organization_id=organization)
        key: Final = scenario.key(team_id=team)
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        gateway.post("/organization/update", {"organization_id": organization, "max_budget": 0})
        assert_budget_refused(settled(lambda: chat_as(gateway, key, model), 422, WRITER_SECONDS))
        assert_budget_refused(settled(lambda: chat_as(peer, key, model), 422, PEER_SECONDS))


@pytest.mark.timeout(240)
def test_a_lowered_organization_budget_is_enforced_at_the_latest_after_the_auth_cache_ttl(gateway: Gateway) -> None:
    """The same organization budget change without an eviction: the organization row may be served from cache.

    The key must be refused with 422 at the latest within the 60 second auth cache TTL plus 15 seconds of slack.
    This is the floor every port must keep even if it never adds the missing eviction.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        organization: Final = scenario.organization(max_budget=100)
        team: Final = scenario.team(organization_id=organization)
        key: Final = scenario.key(team_id=team)
        assert_ok(chat_as(gateway, key, model))
        gateway.post("/organization/update", {"organization_id": organization, "max_budget": 0})
        assert_budget_refused(settled(lambda: chat_as(gateway, key, model), 422, DEFAULT_AUTH_CACHE_TTL_SECONDS + 15))


def test_updating_a_warmed_key_by_its_hash_is_enforced_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A key allowed two models is served the second model on both proxies, then `/key/update` is called with
    `key: <sha256 of the key>` and `models: [<first>]`.

    The second model must be refused with 403 on the writer within 5 seconds and on the peer within 10. Both the
    raw and the hashed spelling must evict the same cache entry (the cache is keyed by the hash).
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        key: Final = scenario.key(models=[first, second])
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, second))
        gateway.post("/key/update", {"key": digest(key), "models": [first]})
        for proxy, seconds in ((gateway, WRITER_SECONDS), (peer, PEER_SECONDS)):
            refused = settled(lambda proxy=proxy: chat_as(proxy, key, second), 403, seconds)
            assert error_of(refused)["type"] == "key_model_access_denied", refused.text


def test_blocking_a_warmed_key_by_its_hash_stops_it_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A key served on both proxies is blocked with `/key/block {"key": <sha256>}`.

    Both proxies must answer 401 within 5 and 10 seconds. `test_warmed_policy.py` pins the same with the raw key.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        gateway.post("/key/block", {"key": digest(key)})
        settled(lambda: chat_as(gateway, key, model), 401, WRITER_SECONDS)
        settled(lambda: chat_as(peer, key, model), 401, PEER_SECONDS)
        gateway.post("/key/unblock", {"key": digest(key)})


def test_deleting_a_warmed_key_by_its_hash_stops_it_on_both_proxies_with_the_generic_401(
    gateway: Gateway, peer: Gateway
) -> None:
    """A key served on both proxies is deleted with `/key/delete {"keys": [<sha256>]}`.

    Both proxies must answer 401 with exactly the unknown key message within 5 and 10 seconds.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = string_value(gateway.post("/key/generate", {})["key"])
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        gateway.post("/key/delete", {"keys": [digest(key)]})
        for proxy, seconds in ((gateway, WRITER_SECONDS), (peer, PEER_SECONDS)):
            refused = settled(lambda proxy=proxy: chat_as(proxy, key, model), 401, seconds)
            assert_denied(refused, status=401, error_type="auth_error", message=UNKNOWN_KEY_MESSAGE)


def test_key_info_answers_the_same_row_for_the_raw_key_and_its_hash(gateway: Gateway) -> None:
    """`/key/info` is asked once with `key=<raw sk- key>` and once with `key=<sha256>`.

    Both answers carry the same `info.models` and `info.team_id`, and the `key` field is the hash in both. The
    key store has one lookup by hash; the raw form is hashed first.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team, models=[model])
        by_raw: Final = gateway.get("/key/info", {"key": key})
        by_hash: Final = gateway.get("/key/info", {"key": digest(key)})
        assert by_raw["key"] == by_hash["key"] == digest(key)
        for answer in (by_raw, by_hash):
            info = answer["info"]
            assert isinstance(info, dict)
            assert (info["models"], info["team_id"]) == ([model], team)
