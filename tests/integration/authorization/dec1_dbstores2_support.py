"""Shared helpers for the round 2 `test_dec1_dbstores2_*` auth object store contract files.

Round 1 (`dec1_dbstores_support.py`) is imported, never edited. The helpers added here create rows that a test
later removes or rewrites with SQL (so they carry no API cleanup that would fail on a row that is gone), and boot
a private proxy whose read replica connection runs through a statement counting relay so a test can observe how
many database round trips a sequence of requests costs. Nothing here imports or patches proxy internals.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, LiteralString

import httpx
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, string_value
from tests.integration._support.database import write_rows
from tests.integration.authorization.dec1_keys_support import assert_denied, upstream_requests

#: The 401 every unexpected auth failure becomes (`_as_proxy_exception`, auth_exception_handler.py:79-84).
GENERIC_AUTH_ERROR_PREFIX: Final = "Authentication Error, "
#: How long a test watches that a cached object keeps standing in for a row SQL changed or removed.
CACHED_HOLD_SECONDS: Final = 3
#: An upper bound for "enforced at the latest after the 60 second management object TTL" with poll slack.
AFTER_TTL_SECONDS: Final = 75


def name(kind: str) -> str:
    return f"dec1-dbstores2-{kind}-{uuid.uuid4().hex[:12]}"


def sql(gateway: Gateway, query: LiteralString, parameters: tuple[str, ...]) -> None:
    """Write straight to the database the given proxy reads, the way an out of band migration or operator would."""
    write_rows(query, parameters, database_url=gateway.database_url)


def bare_user(gateway: Gateway, **fields: JsonValue) -> str:
    """A `LiteLLM_UserTable` row created through `/user/new` without a cleanup, for tests that delete it with SQL."""
    identity: Final = name("user")
    gateway.post("/user/new", {"user_id": identity, "auto_create_key": False, **fields})
    return identity


def bare_organization(gateway: Gateway, **fields: JsonValue) -> tuple[str, str]:
    """An organization created without a cleanup; returns `(organization_id, budget_id)`."""
    created: Final = gateway.post("/organization/new", {"organization_alias": name("org"), **fields})
    return string_value(created["organization_id"]), string_value(created["budget_id"])


def bare_project(gateway: Gateway, team_id: str, **fields: JsonValue) -> str:
    created: Final = gateway.post("/project/new", {"team_id": team_id, "project_alias": name("project"), **fields})
    return string_value(created["project_id"])


def bare_budget(gateway: Gateway, **fields: JsonValue) -> str:
    return string_value(gateway.post("/budget/new", fields)["budget_id"])


def forget_budget(scenario: Scenario, budget_id: str) -> None:
    """Remove a budget row left behind by an organization or by a SQL test, after the keys using it are gone."""
    scenario.cleanups.callback(
        sql, scenario.gateway, 'DELETE FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (budget_id,)
    )


def forget_organization(scenario: Scenario, organization_id: str, budget_id: str) -> None:
    """Remove an organization row and its budget row by SQL once the scenario ends (teams and keys SET NULL)."""
    forget_budget(scenario, budget_id)
    scenario.cleanups.callback(
        sql, scenario.gateway, 'DELETE FROM "LiteLLM_OrganizationTable" WHERE organization_id = %s', (organization_id,)
    )


def assert_generic_auth_refusal(response: httpx.Response) -> None:
    """A 401 `auth_error` whose message is the generic `Authentication Error, <exception text>` wrapping."""
    error: Final = assert_denied(response, status=401, error_type="auth_error")
    assert string_value(error["message"]).startswith(GENERIC_AUTH_ERROR_PREFIX), response.text


def nothing_reached_upstream(gateway: Gateway, text: str) -> bool:
    return upstream_requests(gateway, text) == []


@dataclass(frozen=True, slots=True)
class CountedProxy:
    """A private proxy whose reads go to a replica URL relayed through a statement counter."""

    proxy: Gateway
    scenario: Scenario
    reads: Callable[[], int]


@contextmanager
def counted_proxy(directory: Path) -> Iterator[CountedProxy]:
    """Boot a proxy on a scratch database with `DATABASE_URL_READ_REPLICA` behind `statement_counting_relay`.

    Auth lookups read through the reader pool (`use_writer=False` is the default, see round 1's
    `test_cold_auth_lookups_read_from_the_read_replica`), so the replica counter moves with auth round trips while
    spend and management writes go to the writer.
    """
    from tests.integration._support.client import gateway_from_environment
    from tests.integration._support.database import scratch_database
    from tests.integration._support.database_relay import plaintext_database_url, statement_counting_relay
    from tests.integration._support.process import owned_proxy

    with ExitStack() as stack:
        shared: Final = stack.enter_context(gateway_from_environment())
        database_url: Final = stack.enter_context(scratch_database())
        reader, reader_url = stack.enter_context(statement_counting_relay(database_url))
        proxy: Final = stack.enter_context(
            owned_proxy(
                shared,
                directory / "proxy",
                {
                    "DATABASE_URL": plaintext_database_url(database_url),
                    "DATABASE_URL_READ_REPLICA": reader_url,
                    "PRISMA_HEALTH_WATCHDOG_ENABLED": "false",
                },
            )
        )
        scenario: Final = stack.enter_context(proxy.scenario())
        yield CountedProxy(proxy, scenario, lambda: reader.statements)
