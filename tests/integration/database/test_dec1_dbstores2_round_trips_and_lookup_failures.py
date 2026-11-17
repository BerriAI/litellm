"""Database round trips per request, and lookup failures of the objects round 1 did not cover (tags, memberships).

Contract, read from the branch:

* Auth reads go to the reader pool and are cached: the key view (`combined_view`) under its hash, the user under
  its id, the team under `team_id:<id>` (litellm/proxy/auth/auth_checks.py:3133-3290, 3841-3901, 4485-4567). A
  user id with no row is remembered as missing for `DEFAULT_IN_MEMORY_TTL` (5 s) by `last_db_access_time`
  (`_should_check_db`, auth_checks.py:2914-2925).
* Unknown end user ids and unknown tag names never reach the database one by one: a cached registry of restricted
  end user ids (auth_checks.py:2198-2253) and of existing tag names (auth_checks.py:2467-2486, 2597-2632) answers
  them. A Rust store that loses these registries turns every request with a fresh `user` or tag into a query.
* A failed tag read is returned as a value and the tag budget check fails open (`_fetch_uncached_tags`,
  auth_checks.py:2627-2628, "budget checks fail open").
* A failed membership read is not caught in `common_checks` (auth_checks.py:1220-1233): it reaches
  `handle_authentication_error` (litellm/proxy/auth/auth_exception_handler.py:199-223), which serves the restricted
  fallback identity only with `allow_requests_on_db_unavailable` and otherwise raises.

The round trip tests count statements on a replica relay (`counted_proxy` in `dec1_dbstores2_support.py`). The
proxy runs background readers too, so the assertions bound the growth well below one statement per request rather
than expecting zero. The failure tests reuse round 1's `lookup_outage` rig, which drops only connections whose
outbound bytes carry a marker embedded in one object id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from tests.integration._support.client import Gateway
from tests.integration.authorization.dec1_dbstores2_support import counted_proxy, name
from tests.integration.authorization.dec1_dbstores_support import assert_ok, chat_as, lookup_outage
from tests.integration.authorization.dec1_keys_support import DB_UNAVAILABLE_MESSAGE, assert_denied

#: Requests per round trip test; a store that misses its cache adds at least one statement per request.
_REQUESTS: Final = 25


def test_a_warmed_key_answers_repeated_requests_without_a_read_per_request(tmp_path: Path) -> None:
    """A team key with a user is served once on a private proxy, then 25 more chat requests are sent.

    The replica relay must count fewer than 25 new statements over those 25 requests: key, team, user and
    membership are all served from the auth cache. A store that reads through on every request adds at least four.
    """
    with counted_proxy(tmp_path) as rig:
        model: Final = rig.scenario.model()
        team: Final = rig.scenario.team()
        key: Final = rig.scenario.key(team_id=team, user_id=rig.scenario.member(team))
        assert_ok(chat_as(rig.proxy, key, model))
        before: Final = rig.reads()
        for _ in range(_REQUESTS):
            assert_ok(chat_as(rig.proxy, key, model))
        assert rig.reads() - before < _REQUESTS, f"{rig.reads() - before} reader statements for {_REQUESTS} requests"


def test_a_key_whose_user_has_no_row_does_not_read_the_user_table_on_every_request(tmp_path: Path) -> None:
    """A key names a `user_id` with no `LiteLLM_UserTable` row and is served 25 times in a row after one warm up.

    The replica relay must count fewer than 25 new statements: the missing user is remembered for 5 seconds
    (`last_db_access_time`, auth_checks.py:3170-3176, 3238-3244) and the key itself is cached. The user store must
    keep a negative cache, or a key for an unprovisioned user costs a read per request.
    """
    with counted_proxy(tmp_path) as rig:
        model: Final = rig.scenario.model()
        key: Final = rig.scenario.key(user_id=name("ghost"))
        assert_ok(chat_as(rig.proxy, key, model))
        before: Final = rig.reads()
        for _ in range(_REQUESTS):
            assert_ok(chat_as(rig.proxy, key, model))
        assert rig.reads() - before < _REQUESTS, f"{rig.reads() - before} reader statements for {_REQUESTS} requests"


def test_requests_naming_many_unknown_end_users_share_one_registry_read(tmp_path: Path) -> None:
    """One key sends 25 chat requests, each with a different never seen body `user`, after one warm up request.

    The replica relay must count fewer than 25 new statements: ids outside the cached restricted registry are known
    to restrict nothing and their row is never read (`_end_user_is_known_unrestricted`, auth_checks.py:2221-2253).
    """
    with counted_proxy(tmp_path) as rig:
        model: Final = rig.scenario.model()
        key: Final = rig.scenario.key()
        assert_ok(chat_as(rig.proxy, key, model, user=name("end-user")))
        before: Final = rig.reads()
        for _ in range(_REQUESTS):
            assert_ok(chat_as(rig.proxy, key, model, user=name("end-user")))
        assert rig.reads() - before < _REQUESTS, f"{rig.reads() - before} reader statements for {_REQUESTS} requests"


def test_requests_naming_many_unknown_tags_share_one_registry_read(tmp_path: Path) -> None:
    """One key sends 25 chat requests, each carrying `metadata.tags` with a different tag name that has no row.

    The replica relay must count fewer than 25 new statements: names absent from the cached tag registry never
    reach the database (`_fetch_uncached_tags`, auth_checks.py:2606-2612).
    """
    with counted_proxy(tmp_path) as rig:
        model: Final = rig.scenario.model()
        key: Final = rig.scenario.key()
        assert_ok(chat_as(rig.proxy, key, model, metadata={"tags": [name("tag")]}))
        before: Final = rig.reads()
        for _ in range(_REQUESTS):
            assert_ok(chat_as(rig.proxy, key, model, metadata={"tags": [name("tag")]}))
        assert rig.reads() - before < _REQUESTS, f"{rig.reads() - before} reader statements for {_REQUESTS} requests"


def test_a_cold_key_reads_the_database_again_after_it_is_updated(tmp_path: Path) -> None:
    """A key is warmed, then `/key/update` changes its `metadata`; the next request is counted.

    The replica relay must count at least one new statement for that request: the update evicted the key and the
    next request re-reads `combined_view`. Paired with the warm test this pins that the cache is per write, not a
    process lifetime memo.
    UNVERIFIED: whether the post-update read goes to the reader or, after a write, to the writer
    (`check_db_only`/`use_writer` paths); the reader is the default for auth reads.
    """
    with counted_proxy(tmp_path) as rig:
        model: Final = rig.scenario.model()
        key: Final = rig.scenario.key()
        assert_ok(chat_as(rig.proxy, key, model))
        rig.proxy.post("/key/update", {"key": key, "metadata": {"dec1": name("marker")}})
        before: Final = rig.reads()
        assert_ok(chat_as(rig.proxy, key, model))
        assert rig.reads() > before, "the updated key was not read again"


def test_a_failed_tag_read_serves_the_request_without_the_tag_budget(gateway: Gateway, tmp_path: Path) -> None:
    """On the lookup outage rig a tag whose name carries the marker has `max_budget: 0`; it was never used. With the
    relay armed, a key sends a request with `metadata.tags: [<tag>]`.

    The request is served with 200: the tag fetch query names the tag, its connection is dropped, and the error is
    returned as a value so the tag budget check fails open (auth_checks.py:2627-2628). The registry query names no
    tag, so it still succeeds. A tag store that raises instead would turn a tag outage into failed requests.
    """
    with lookup_outage(gateway, tmp_path) as outage:
        model: Final = outage.scenario.model()
        tag_name: Final = f"{outage.trigger}-tag"
        outage.proxy.post("/tag/new", {"name": tag_name, "max_budget": 0})
        key: Final = outage.scenario.key()
        outage.arm()
        assert_ok(chat_as(outage.proxy, key, model, metadata={"tags": [tag_name]}))
        assert outage.dropped(), "the tag lookup never reached the relay"


def test_a_failed_membership_read_answers_503(gateway: Gateway, tmp_path: Path) -> None:
    """On the lookup outage rig a member whose user id carries the marker has `max_budget_in_team: 0` and a never
    used team key. With the relay armed, the key calls chat.

    The answer is 503 with the database unavailable message: the user read fails and is swallowed, but the
    membership read in `common_checks` raises and `handle_authentication_error` re-raises it. Never 200, since the
    member budget could not be checked. UNVERIFIED: that a dropped connection is classified as
    `is_database_service_unavailable_error` (503) rather than a generic 401.
    """
    with lookup_outage(gateway, tmp_path) as outage:
        model: Final = outage.scenario.model()
        team: Final = outage.scenario.team()
        member: Final = f"{outage.trigger}-member"
        outage.proxy.post("/user/new", {"user_id": member, "auto_create_key": False})
        outage.proxy.post("/team/member_add", {"team_id": team, "member": {"role": "user", "user_id": member}})
        outage.proxy.post("/team/member_update", {"team_id": team, "user_id": member, "max_budget_in_team": 0})
        key: Final = outage.scenario.key(team_id=team, user_id=member)
        outage.arm()
        refused: Final = chat_as(outage.proxy, key, model)
        assert_denied(refused, status=503, error_type="no_db_connection", message=DB_UNAVAILABLE_MESSAGE)


def test_with_the_availability_flag_a_failed_membership_read_is_served(gateway: Gateway, tmp_path: Path) -> None:
    """The same never used member key during a failed membership read, on a rig started with
    `allow_requests_on_db_unavailable: true`.

    The request is served with 200: the connection error reaches `handle_authentication_error`, which returns the
    restricted `INTERNAL_USER` fallback identity (auth_exception_handler.py:199-223). UNVERIFIED: whether the
    fallback identity may call this model with the suite config (it carries no model list, so every model).
    """
    with lookup_outage(gateway, tmp_path, general_settings={"allow_requests_on_db_unavailable": True}) as outage:
        model: Final = outage.scenario.model()
        team: Final = outage.scenario.team()
        member: Final = f"{outage.trigger}-member"
        outage.proxy.post("/user/new", {"user_id": member, "auto_create_key": False})
        outage.proxy.post("/team/member_add", {"team_id": team, "member": {"role": "user", "user_id": member}})
        outage.proxy.post("/team/member_update", {"team_id": team, "user_id": member, "max_budget_in_team": 0})
        key: Final = outage.scenario.key(team_id=team, user_id=member)
        outage.arm()
        assert_ok(chat_as(outage.proxy, key, model))
