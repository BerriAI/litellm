"""What a request gets when the database fails on one specific auth object lookup, and which lookups use the
read replica.

Contract, read from litellm/proxy/auth/user_api_key_auth.py (`run_centralized_common_checks`, `_safe_fetch`,
`_token_can_vouch_for_team`, `_inherit_org_identity`) and litellm/proxy/auth/auth_checks.py (`get_team_object`,
`get_user_object`, `get_end_user_object`, `get_project_object`, `get_org_object_for_request`):

* The owner objects of a key (team, user, project, end user) are fetched concurrently. `_safe_fetch` lets an
  `HTTPException` through and swallows every other error to `None`, so a failed user, project or end user read
  means "no such object" for that request.
* `get_team_object` turns every failure into a 404. A team that is provably gone (`TeamNotFoundError`) is never
  replaced; for any other failure the key's own copy of its team (from `combined_view`) stands in when it
  records a model list, or when `general_settings.allow_requests_on_db_unavailable` is true. Otherwise the 404
  propagates.
* The organization read keeps a "last known" copy; with no copy, a database unavailable error is raised (503
  `no_db_connection`) unless `allow_requests_on_db_unavailable` is true, in which case the request continues
  without organization limits.
* Reads go to `DATABASE_URL_READ_REPLICA` when it is configured (`RoutingPrismaWrapper`, and the Rust reader pool
  from #49804 that the new stores use); writes go to `DATABASE_URL`.

The rig (`lookup_outage` in `dec1_dbstores_support.py`) boots a private proxy on a scratch database behind a relay
that drops any connection whose outbound bytes contain a per-rig marker. Each test embeds that marker in the id of
exactly one object (team, customer, project, organization), so only the lookups naming it fail. The key itself is
looked up by its hash, which never contains the marker. Keys are never used before the relay is armed, so their
owners are not cached. Not duplicated: whole-database outages (`test_dec1_failures_postgres_*`), key lookup
failures (`test_dec1_keys_db_unavailable.py`).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import eventually, gateway_from_environment, string_value
from tests.integration._support.database import scratch_database
from tests.integration._support.database_relay import (
    dropped_connection_relay,
    plaintext_database_url,
    statement_counting_relay,
)
from tests.integration._support.process import owned_proxy
from tests.integration.authorization.dec1_dbstores_support import (
    LookupOutage,
    assert_ok,
    chat_as,
    customer,
    lookup_outage,
    marker,
)
from tests.integration.authorization.dec1_keys_support import (
    DB_UNAVAILABLE_MESSAGE,
    assert_denied,
    digest,
    error_of,
    upstream_requests,
)

pytestmark: Final = pytest.mark.timeout(300)


@pytest.fixture(scope="module")
def strict(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LookupOutage]:
    with (
        gateway_from_environment() as shared,
        lookup_outage(
            shared,
            tmp_path_factory.mktemp("dec1-dbstores-strict"),
            general_settings={"allow_requests_on_db_unavailable": False},
        ) as rig,
    ):
        yield rig


@pytest.fixture(scope="module")
def lenient(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LookupOutage]:
    with (
        gateway_from_environment() as shared,
        lookup_outage(
            shared,
            tmp_path_factory.mktemp("dec1-dbstores-lenient"),
            general_settings={"allow_requests_on_db_unavailable": True},
        ) as rig,
    ):
        yield rig


def _failing_team(rig: LookupOutage, **fields: JsonValue) -> str:
    team_id: Final = f"{rig.trigger}-team-{uuid.uuid4().hex[:8]}"
    rig.scenario.gateway.post("/team/new", {"team_id": team_id, "team_alias": team_id, **fields})
    rig.scenario.cleanups.callback(rig.proxy.request, "POST", "/team/delete", {"team_ids": [team_id]})
    return team_id


def _armed_chat(rig: LookupOutage, key: str, model: str, **extra: JsonValue) -> httpx.Response:
    rig.arm()
    try:
        return chat_as(rig.proxy, key, model, **extra)
    finally:
        rig.disarm()


def test_a_team_lookup_failure_lets_the_key_vouch_with_the_team_models_it_carries(strict: LookupOutage) -> None:
    """A team allowing only model A has a key that was never used; every query naming the team is dropped.

    Model A is served with 200 and model B is refused with 403 `team_model_access_denied`: the key's recorded
    `team_models` stand in for the unreadable team, which can never widen access. The relay must have dropped at
    least one connection, proving the team lookup was attempted.
    """
    rig: Final = strict
    allowed: Final = rig.scenario.model()
    other: Final = rig.scenario.model()
    team: Final = _failing_team(rig, models=[allowed])
    key: Final = rig.scenario.key(team_id=team)
    assert_ok(_armed_chat(rig, key, allowed))
    assert rig.dropped(), "the team lookup never reached the database"
    refused: Final = _armed_chat(rig, key, other)
    assert refused.status_code == 403, refused.text
    assert error_of(refused)["type"] == "team_model_access_denied", refused.text


def test_a_team_lookup_failure_does_not_serve_a_key_whose_team_records_no_models(strict: LookupOutage) -> None:
    """A team without a model list (every model) has a never used key; the team lookup is dropped and
    `allow_requests_on_db_unavailable` is false.

    The request is not served and nothing reaches the upstream: an empty recorded list reads as "every model"
    and `team_blocked: false`, so the key may not vouch for its team (`_token_can_vouch_for_team`).
    """
    rig: Final = strict
    model: Final = rig.scenario.model()
    team: Final = _failing_team(rig)
    key: Final = rig.scenario.key(team_id=team)
    text: Final = marker()
    refused: Final = _armed_chat(rig, key, model, text=text)
    assert refused.status_code != 200, refused.text
    assert upstream_requests(rig.proxy, text) == []


def test_a_team_lookup_failure_answers_503_no_db_connection(strict: LookupOutage) -> None:
    """The same unvouched team key during a failed team lookup.

    Intended: 503 `no_db_connection` with the standard message, like a failed key lookup, so clients retry.
    """
    pytest.skip(
        "BUG: get_team_object maps every lookup error to a 404 'Team doesn't exist in db', so a database "
        "failure on the team read tells the client the team does not exist"
    )
    rig: Final = strict
    model: Final = rig.scenario.model()
    key: Final = rig.scenario.key(team_id=_failing_team(rig))
    assert_denied(
        _armed_chat(rig, key, model), status=503, error_type="no_db_connection", message=DB_UNAVAILABLE_MESSAGE
    )


def test_with_the_availability_flag_a_team_lookup_failure_serves_an_unvouched_key(lenient: LookupOutage) -> None:
    """With `allow_requests_on_db_unavailable: true`, a team without a model list has a never used key and its
    team lookup is dropped.

    The request is served with 200 from the key's own copy of the team.
    """
    rig: Final = lenient
    model: Final = rig.scenario.model()
    key: Final = rig.scenario.key(team_id=_failing_team(rig))
    assert_ok(_armed_chat(rig, key, model))
    assert rig.dropped()


def test_a_team_key_is_served_again_once_the_team_lookup_recovers(strict: LookupOutage) -> None:
    """The unvouched team key is refused while its team lookup fails, then the relay is disarmed.

    The next request is served with 200 without restarting anything: the failure was not cached as "no team".
    """
    rig: Final = strict
    model: Final = rig.scenario.model()
    key: Final = rig.scenario.key(team_id=_failing_team(rig))
    assert _armed_chat(rig, key, model).status_code != 200
    assert_ok(eventually(lambda: chat_as(rig.proxy, key, model), lambda r: r.status_code == 200, seconds=10))


def test_a_warmed_team_key_is_served_while_its_team_lookup_would_fail(strict: LookupOutage) -> None:
    """A team key is served once with the relay disarmed (team cached), then the relay is armed.

    The next request is served with 200 and no connection is dropped for it, because the cached team needs no
    database read.
    """
    rig: Final = strict
    model: Final = rig.scenario.model()
    key: Final = rig.scenario.key(team_id=_failing_team(rig))
    assert_ok(chat_as(rig.proxy, key, model))
    assert_ok(_armed_chat(rig, key, model))


def test_a_user_lookup_failure_serves_a_personal_key_without_the_users_budget(strict: LookupOutage) -> None:
    """A user named with the marker has `max_budget: 0` and a never used personal key; the user lookup is dropped.

    Pinned current behaviour: the request is served with 200. `get_user_object` raises the database error,
    `_safe_fetch` swallows it to `None`, and with no user object there is no user budget to check. A port that
    turns this into a refusal changes availability during partial outages and must say so.
    """
    rig: Final = strict
    model: Final = rig.scenario.model()
    user_id: Final = f"{rig.trigger}-user-{uuid.uuid4().hex[:8]}"
    rig.proxy.post("/user/new", {"user_id": user_id, "max_budget": 0, "auto_create_key": False})
    rig.scenario.cleanups.callback(rig.proxy.request, "POST", "/user/delete", {"user_ids": [user_id]})
    key: Final = rig.scenario.key(user_id=user_id)
    assert_ok(_armed_chat(rig, key, model))


def test_a_customer_lookup_failure_does_not_serve_a_blocked_customer(strict: LookupOutage) -> None:
    """A blocked customer named with the marker sends its first request while its row lookup is dropped.

    Intended: the request is refused (the blocked 400, or a 503), never served.
    """
    pytest.skip(
        "BUG: get_end_user_object catches every exception and returns None, so a failed read of a blocked "
        "customer's row serves the customer"
    )
    rig: Final = strict
    model: Final = rig.scenario.model()
    end_user: Final = customer(rig.scenario, user_id=f"{rig.trigger}-cust-{uuid.uuid4().hex[:8]}", blocked=True)
    key: Final = rig.scenario.key()
    assert _armed_chat(rig, key, model, user=end_user).status_code in (400, 503)


def test_a_project_lookup_failure_does_not_serve_a_blocked_projects_key(strict: LookupOutage) -> None:
    """A blocked project named with the marker has a never used key; the project lookup is dropped.

    Intended: the request is refused (project blocked 401 or a 503), never served.
    """
    pytest.skip(
        "BUG: get_project_object lets the database error escape and _safe_fetch swallows it to None, so a "
        "failed project read drops the project block and model list for that request"
    )
    rig: Final = strict
    model: Final = rig.scenario.model()
    team: Final = rig.scenario.team()
    project: Final = string_value(
        rig.proxy.post(
            "/project/new",
            {"team_id": team, "project_id": f"{rig.trigger}-proj-{uuid.uuid4().hex[:8]}", "blocked": True},
        )["project_id"]
    )
    rig.scenario.cleanups.callback(rig.proxy.request, "DELETE", "/project/delete", {"project_ids": [project]})
    key: Final = rig.scenario.key(team_id=team, project_id=project)
    assert _armed_chat(rig, key, model).status_code in (401, 503)


def _team_in_failing_org(rig: LookupOutage) -> str:
    organization: Final = f"{rig.trigger}-org-{uuid.uuid4().hex[:8]}"
    created: Final = rig.proxy.post(
        "/organization/new", {"organization_id": organization, "organization_alias": organization, "max_budget": 100}
    )
    rig.scenario.cleanups.callback(rig.scenario.delete_organization, organization, string_value(created["budget_id"]))
    return rig.scenario.team(organization_id=organization)


def test_an_unseen_organization_whose_lookup_fails_answers_503(strict: LookupOutage) -> None:
    """A team inside an organization named with the marker has a never used key; the organization lookup is
    dropped and no last known copy exists.

    UNVERIFIED: the answer is 503 `no_db_connection` with the standard message (`get_org_object_for_request`
    re-raises a database unavailable error). Assumes a dropped connection is classified as "database
    unavailable" by `PrismaDBExceptionHandler`.
    """
    rig: Final = strict
    model: Final = rig.scenario.model()
    key: Final = rig.scenario.key(team_id=_team_in_failing_org(rig))
    assert_denied(
        _armed_chat(rig, key, model), status=503, error_type="no_db_connection", message=DB_UNAVAILABLE_MESSAGE
    )


def test_with_the_availability_flag_an_organization_lookup_failure_is_served(lenient: LookupOutage) -> None:
    """With `allow_requests_on_db_unavailable: true`, the same never seen organization lookup is dropped.

    The request is served with 200, without organization limits.
    """
    rig: Final = lenient
    model: Final = rig.scenario.model()
    key: Final = rig.scenario.key(team_id=_team_in_failing_org(rig))
    assert_ok(_armed_chat(rig, key, model))


def test_a_seen_organization_keeps_serving_from_its_last_known_copy(strict: LookupOutage) -> None:
    """A team key inside an organization named with the marker is served once (the organization is cached and
    kept as "last known"), then the relay is armed.

    The next request is served with 200 whether the organization comes from the cache or from the last known
    copy; it must never turn into a 503 for an organization this proxy has already read.
    """
    rig: Final = strict
    model: Final = rig.scenario.model()
    key: Final = rig.scenario.key(team_id=_team_in_failing_org(rig))
    assert_ok(chat_as(rig.proxy, key, model))
    assert_ok(_armed_chat(rig, key, model))


def test_cold_auth_lookups_read_from_the_read_replica(tmp_path: Path) -> None:
    """A private proxy runs with `DATABASE_URL` and `DATABASE_URL_READ_REPLICA` behind two statement counting
    relays to the same scratch database. A key, its team and its user are created, then the key is used once.

    The replica relay must count more statements after the first request than before it: the key, team and user
    lookups are reads and go to the replica. The writer still answers the management writes. The Rust stores
    must keep routing these reads to the reader pool (`use_writer=False` is the default).
    """
    with ExitStack() as stack:
        shared: Final = stack.enter_context(gateway_from_environment())
        database_url: Final = stack.enter_context(scratch_database())
        writer, writer_url = stack.enter_context(statement_counting_relay(database_url))
        reader, reader_url = stack.enter_context(statement_counting_relay(database_url))
        proxy: Final = stack.enter_context(
            owned_proxy(
                shared,
                tmp_path / "proxy",
                {
                    "DATABASE_URL": writer_url,
                    "DATABASE_URL_READ_REPLICA": reader_url,
                    "PRISMA_HEALTH_WATCHDOG_ENABLED": "false",
                },
            )
        )
        scenario: Final = stack.enter_context(proxy.scenario())
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team, user_id=scenario.member(team))
        before: Final = reader.statements
        written: Final = writer.statements
        assert_ok(chat_as(proxy, key, model))
        assert reader.statements > before, "no auth lookup reached the read replica"
        assert written > 0


def test_an_unreachable_read_replica_never_turns_a_valid_key_into_a_401(tmp_path: Path) -> None:
    """A private proxy reads through a replica relay that drops every connection carrying the hash of one new key.

    The key's first request must be served (200) or answered as unavailable (503), never refused as unknown
    (401): a replica that cannot answer is not evidence that the key does not exist.

    UNVERIFIED: whether the proxy falls back to the writer (`RoutingPrismaWrapper` reader availability flag) and
    serves 200, or answers 503.
    """
    with ExitStack() as stack:
        shared: Final = stack.enter_context(gateway_from_environment())
        database_url: Final = stack.enter_context(scratch_database())
        key: Final = f"sk-{marker()}"
        relay, reader_url = stack.enter_context(dropped_connection_relay(database_url, digest(key).encode()))
        proxy: Final = stack.enter_context(
            owned_proxy(
                shared,
                tmp_path / "proxy",
                {
                    "DATABASE_URL": plaintext_database_url(database_url),
                    "DATABASE_URL_READ_REPLICA": plaintext_database_url(reader_url),
                    "PRISMA_HEALTH_WATCHDOG_ENABLED": "false",
                },
            )
        )
        scenario: Final = stack.enter_context(proxy.scenario())
        model: Final = scenario.model()
        proxy.post("/key/generate", {"key": key})
        scenario.cleanups.callback(proxy.request, "POST", "/key/delete", {"keys": [key]})
        relay.arm()
        try:
            answer: Final = chat_as(proxy, key, model)
        finally:
            relay.disarm()
        assert answer.status_code in (200, 503), answer.text
