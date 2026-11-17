"""Cache invalidation after management mutations that round 1 and the keys files did not pin, on one and two proxies.

Contract, read from the branch:

* Every management write evicts the auth objects it changed on the proxy that handled it and broadcasts the
  eviction on `litellm_proxy.auth_cache_invalidation` (`evict_and_broadcast`,
  litellm/proxy/common_utils/auth_cache_invalidation_pubsub.py), so a warmed peer drops its copy too. The suite
  requires the writer within 5 seconds and the peer within 10, against the 60 second management object TTL.
* The broadcast is symmetric: a write handled by the second proxy must reach the first.
* The key's cached `combined_view` row carries copies of joined rows: the key's budget row (`b`), the team's
  `organization_id` as `org_id` (litellm/proxy/utils.py:6829-6869). `/team/update` evicts the team object but not
  its keys, and `/budget/update` evicts only customer hierarchy entries and `team_member_default_budget:<id>`
  (litellm/proxy/management_endpoints/budget_management_endpoints.py:238-277), so changes that only reach the key
  through those copies wait for the key's TTL. Those are written for the intended behaviour and skipped as BUG.
* `organization_max_budget_check` uses `valid_token.org_id` first and only then the fresh team object
  (litellm/proxy/auth/auth_checks.py:7996-8000).

Not repeated: key, team, member and user model narrowing, key moves between teams, team deletion, writer only key
unblock (`test_dec1_keys_cache_staleness.py`), require_trace_id, aliases, member budgets, org and customer budget
updates, project block and model widening, tag create/delete/update, user budget lowering (round 1).
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration.authorization.dec1_dbstores2_support import AFTER_TTL_SECONDS
from tests.integration.authorization.dec1_dbstores_support import (
    PEER_SECONDS,
    WRITER_SECONDS,
    assert_budget_refused,
    assert_customer_model_denied,
    assert_ok,
    chat_as,
    customer,
    settled,
    tag,
)
from tests.integration.authorization.dec1_keys_support import (
    BLOCKED_KEY_MESSAGE,
    assert_denied,
    digest,
    error_of,
    spend_rows,
)


def _team_blocked_message(team: str) -> str:
    return f"Authentication Error, Team={team} is blocked. Update via `/team/unblock` if you're an admin."


def _project_blocked_message(project: str) -> str:
    return f"Authentication Error, Project={project} is blocked. Update via `/project/update` if you're an admin."


def test_unblocking_a_warmed_team_serves_its_key_again_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A team key is refused on both proxies after `/team/block`, then `/team/unblock` is called on the first.

    The key must be served with 200 on the writer within 5 seconds and on the peer within 10 seconds. Before the
    unblock both refusals carry the exact team blocked message.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        gateway.post("/team/block", {"team_id": team})
        for proxy, seconds in ((gateway, WRITER_SECONDS), (peer, PEER_SECONDS)):
            refused = settled(lambda proxy=proxy: chat_as(proxy, key, model), 401, seconds)
            assert error_of(refused)["message"] == _team_blocked_message(team), refused.text
        gateway.post("/team/unblock", {"team_id": team})
        assert_ok(settled(lambda: chat_as(gateway, key, model), 200, WRITER_SECONDS))
        assert_ok(settled(lambda: chat_as(peer, key, model), 200, PEER_SECONDS))


def test_a_team_blocked_through_the_second_proxy_is_refused_on_the_first(gateway: Gateway, peer: Gateway) -> None:
    """A team key is served on both proxies, then `/team/block` is sent to the peer, not to the first proxy.

    The first proxy must refuse the key with the team blocked 401 within 10 seconds. The eviction broadcast has no
    preferred direction; a port that only publishes from one instance, or only subscribes on another, fails here.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        peer.post("/team/block", {"team_id": team})
        refused: Final = settled(lambda: chat_as(gateway, key, model), 401, PEER_SECONDS)
        assert error_of(refused)["message"] == _team_blocked_message(team), refused.text
        gateway.post("/team/unblock", {"team_id": team})


def test_a_key_blocked_through_the_second_proxy_by_its_hash_is_refused_on_the_first(
    gateway: Gateway, peer: Gateway
) -> None:
    """A key is served on both proxies, then the peer receives `/key/block {"key": <sha256 of the key>}`.

    The first proxy must answer 401 `auth_error` with the exact blocked key message within 10 seconds.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        peer.post("/key/block", {"key": digest(key)})
        refused: Final = settled(lambda: chat_as(gateway, key, model), 401, PEER_SECONDS)
        assert_denied(refused, status=401, error_type="auth_error", message=BLOCKED_KEY_MESSAGE)


def test_unblocking_a_key_by_its_hash_restores_it_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A key is blocked and refused on both proxies, then `/key/unblock {"key": <sha256>}` is sent to the first.

    The key must be served on the writer within 5 seconds and on the peer within 10 seconds. The writer only, raw
    key form is pinned in `test_dec1_keys_cache_staleness.py`; this adds the hash form and the peer.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        gateway.post("/key/block", {"key": key})
        for proxy, seconds in ((gateway, WRITER_SECONDS), (peer, PEER_SECONDS)):
            settled(lambda proxy=proxy: chat_as(proxy, key, model), 401, seconds)
        gateway.post("/key/unblock", {"key": digest(key)})
        assert_ok(settled(lambda: chat_as(gateway, key, model), 200, WRITER_SECONDS))
        assert_ok(settled(lambda: chat_as(peer, key, model), 200, PEER_SECONDS))


def test_unblocking_a_warmed_project_serves_its_key_again_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A project created with `blocked: true` refuses its key on both proxies with the exact project message; then
    `/project/update` sets `blocked: false`.

    The key must be served on the writer within 5 seconds and on the peer within 10 seconds
    (`delete_cached_project_object`, auth_checks.py:7955-7968, evicts `project_id:<id>` and broadcasts).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        project: Final = scenario.project(team, blocked=True)
        key: Final = scenario.key(team_id=team, project_id=project)
        for proxy in (gateway, peer):
            refused = chat_as(proxy, key, model)
            assert refused.status_code == 401, refused.text
            assert error_of(refused)["message"] == _project_blocked_message(project), refused.text
        gateway.post("/project/update", {"project_id": project, "blocked": False})
        assert_ok(settled(lambda: chat_as(gateway, key, model), 200, WRITER_SECONDS))
        assert_ok(settled(lambda: chat_as(peer, key, model), 200, PEER_SECONDS))


def test_removing_a_model_from_a_warmed_project_refuses_it_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A project allows models A and B and its key is served B on both proxies; `/project/update` sets `models: [A]`.

    Model B must be refused with 403 on the writer within 5 seconds and on the peer within 10 seconds, while A
    stays served, with `error.type == "project_model_access_denied"` as round 1 pinned for a cold key. Round 1
    pinned widening; narrowing is the direction that matters for access control.
    """
    with gateway.scenario() as scenario:
        kept: Final = scenario.model()
        removed: Final = scenario.model()
        team: Final = scenario.team()
        project: Final = scenario.project(team, models=[kept, removed])
        key: Final = scenario.key(team_id=team, project_id=project)
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, removed))
        gateway.post("/project/update", {"project_id": project, "models": [kept]})
        for proxy, seconds in ((gateway, WRITER_SECONDS), (peer, PEER_SECONDS)):
            refused = settled(lambda proxy=proxy: chat_as(proxy, key, removed), 403, seconds)
            assert error_of(refused)["type"] == "project_model_access_denied", refused.text
        assert_ok(chat_as(gateway, key, kept))


def test_narrowing_a_warmed_customers_models_refuses_the_removed_model_on_both_proxies(
    gateway: Gateway, peer: Gateway
) -> None:
    """A customer allowed models A and B is served B on both proxies; `/customer/update` sets `models: [A]`.

    B must be refused with 403 `customer_model_access_denied` on the writer within 5 seconds and on the peer within
    10 seconds. The `end_user_id:<id>` entry and the restricted registry both hold the old list.
    """
    with gateway.scenario() as scenario:
        kept: Final = scenario.model()
        removed: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = customer(scenario, models=[kept, removed])
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, removed, user=end_user))
        gateway.post("/customer/update", {"user_id": end_user, "models": [kept]})
        assert_customer_model_denied(
            settled(lambda: chat_as(gateway, key, removed, user=end_user), 403, WRITER_SECONDS)
        )
        assert_customer_model_denied(settled(lambda: chat_as(peer, key, removed, user=end_user), 403, PEER_SECONDS))


def test_adding_a_zero_budget_tag_to_a_warmed_keys_metadata_refuses_it_on_both_proxies(
    gateway: Gateway, peer: Gateway
) -> None:
    """A tag with `max_budget: 0` exists; a key without tags is served on both proxies, then `/key/update` sets
    `metadata: {"tags": [<tag>]}`.

    The key must be refused with 422 `budget_exceeded` on the writer within 5 seconds and on the peer within 10
    seconds: the key cache entry is evicted and broadcast, and the tag object is read on the next request.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        zero: Final = tag(scenario, max_budget=0)
        key: Final = scenario.key()
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        gateway.post("/key/update", {"key": key, "metadata": {"tags": [zero]}})
        assert_budget_refused(settled(lambda: chat_as(gateway, key, model), 422, WRITER_SECONDS))
        assert_budget_refused(settled(lambda: chat_as(peer, key, model), 422, PEER_SECONDS))


def test_raising_a_users_budget_readmits_their_warmed_personal_key_on_both_proxies(
    gateway: Gateway, peer: Gateway
) -> None:
    """A user with `max_budget: 0` has a personal key refused with 422 on both proxies; `/user/update` then sets
    `max_budget: 100`.

    The key must be served on the writer within 5 seconds and on the peer within 10 seconds. Round 1 pinned the
    lowering direction; readmission proves the user object, not only a refusal flag, is evicted.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(max_budget=0)
        key: Final = scenario.key(user_id=user)
        for proxy in (gateway, peer):
            assert_budget_refused(chat_as(proxy, key, model))
        gateway.post("/user/update", {"user_id": user, "max_budget": 100})
        assert_ok(settled(lambda: chat_as(gateway, key, model), 200, WRITER_SECONDS))
        assert_ok(settled(lambda: chat_as(peer, key, model), 200, PEER_SECONDS))


def test_pointing_a_warmed_key_at_a_zero_budget_row_refuses_it_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A key without a budget is served on both proxies; `/key/update` sets `budget_id` to a row with `max_budget: 0`.

    The key must be refused with 422 on the writer within 5 seconds and on the peer within 10 seconds: the key
    entry is evicted by hash and broadcast, and the next `combined_view` read joins the new budget row.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        budget: Final = scenario.budget(max_budget=0)
        key: Final = scenario.key()
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        gateway.post("/key/update", {"key": key, "budget_id": budget})
        assert_budget_refused(settled(lambda: chat_as(gateway, key, model), 422, WRITER_SECONDS))
        assert_budget_refused(settled(lambda: chat_as(peer, key, model), 422, PEER_SECONDS))


def test_lowering_a_warmed_keys_budget_row_refuses_it_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A key linked to a budget row with `max_budget: 100` is served on both proxies; `/budget/update` sets
    `max_budget: 0` on that row.

    Intended: refused with 422 on the writer within 5 seconds and on the peer within 10 seconds. Today the row's
    limit lives in the cached key view (`litellm_budget_table_max_budget`) and `/budget/update` evicts no key
    (budget_management_endpoints.py:238-277), so the change waits for the key's 60 second TTL.
    """
    pytest.skip("BUG: /budget/update does not evict keys linked by budget_id; their cached view keeps the old limit")
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        budget: Final = scenario.budget(max_budget=100)
        key: Final = scenario.key(budget_id=budget)
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        gateway.post("/budget/update", {"budget_id": budget, "max_budget": 0})
        assert_budget_refused(settled(lambda: chat_as(gateway, key, model), 422, WRITER_SECONDS))
        assert_budget_refused(settled(lambda: chat_as(peer, key, model), 422, PEER_SECONDS))


def test_a_lowered_key_budget_row_is_enforced_at_the_latest_after_the_auth_cache_ttl(gateway: Gateway) -> None:
    """The same `/budget/update` to `max_budget: 0` on a warmed key's budget row, without any eviction.

    The key must be refused with 422 within 75 seconds: the key view is re-read from the database once the 60 second
    management object TTL lapses. This bound holds today and must keep holding after the port.

    BUG (run 44176682961): the key was still served 75 seconds after /budget/update set max_budget 0, while a key
    generated with a zero budget row is refused (test_a_key_linked_to_a_zero_budget_row_is_refused). The cached key
    view outlived the auth cache TTL. Medium confidence."""
    pytest.skip("BUG: a lowered budget row is not enforced on a warmed key even after the 60 second auth cache TTL")
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        budget: Final = scenario.budget(max_budget=100)
        key: Final = scenario.key(budget_id=budget)
        assert_ok(chat_as(gateway, key, model))
        gateway.post("/budget/update", {"budget_id": budget, "max_budget": 0})
        assert_budget_refused(settled(lambda: chat_as(gateway, key, model), 422, AFTER_TTL_SECONDS))


def test_moving_a_warmed_team_into_a_zero_budget_organization_refuses_its_key_on_both_proxies(
    gateway: Gateway, peer: Gateway
) -> None:
    """A team outside any organization has a key served on both proxies; `/team/update` then sets `organization_id`
    to an organization with `max_budget: 0`.

    The key must be refused with 422 on the writer within 5 seconds and on the peer within 10 seconds. The cached
    key view has `org_id = NULL`, so the organization comes from the freshly evicted team object
    (`_inherit_org_identity`, user_api_key_auth.py:3826-3827).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        organization: Final = scenario.organization(max_budget=0)
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        gateway.post("/team/update", {"team_id": team, "organization_id": organization})
        assert_budget_refused(settled(lambda: chat_as(gateway, key, model), 422, WRITER_SECONDS))
        assert_budget_refused(settled(lambda: chat_as(peer, key, model), 422, PEER_SECONDS))


def test_moving_a_warmed_team_between_organizations_applies_the_new_organizations_budget(
    gateway: Gateway, peer: Gateway
) -> None:
    """A team inside organization A (no budget) has a key served on both proxies; `/team/update` moves the team to
    organization B with `max_budget: 0`.

    Intended: refused with 422 on the writer within 5 seconds and on the peer within 10 seconds, and the next spend
    row names B. Today the cached key view keeps `org_id = A` (the `t.organization_id AS org_id` copy), which
    `organization_max_budget_check` prefers over the team object, and `/team/update` does not evict the team's keys.
    """
    pytest.skip("BUG: a warmed team key keeps its old org_id after /team/update moves the team to another org")
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        first: Final = scenario.organization()
        second: Final = scenario.organization(max_budget=0)
        team: Final = scenario.team(organization_id=first)
        key: Final = scenario.key(team_id=team)
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        gateway.post("/team/update", {"team_id": team, "organization_id": second})
        assert_budget_refused(settled(lambda: chat_as(gateway, key, model), 422, WRITER_SECONDS))
        assert_budget_refused(settled(lambda: chat_as(peer, key, model), 422, PEER_SECONDS))


def test_after_a_team_moves_organization_a_never_used_key_bills_the_new_organization(gateway: Gateway) -> None:
    """A team is moved from organization A to organization B before its key is ever used.

    The key's first request is served and its spend row has `organization_id` B: the cold `combined_view` read
    takes the team's current `organization_id`. This is the half of the move that works today.

    BUG (run 44176682961): the first spend row named organization A. The key view is cached when the key is
    generated, with org_id copied from the team, and /team/update does not evict the team's keys. Same root cause as
    the skipped warmed key test above."""
    pytest.skip("BUG: a team key generated before the team moved organization bills the old organization")
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        first: Final = scenario.organization()
        second: Final = scenario.organization()
        team: Final = scenario.team(organization_id=first)
        key: Final = scenario.key(team_id=team)
        gateway.post("/team/update", {"team_id": team, "organization_id": second})
        assert_ok(chat_as(gateway, key, model))
        assert spend_rows(gateway, key)[0]["organization_id"] == second
