"""Shared helpers for the `test_dec1_mgmt_*` management API contract files.

These helpers only build callers and read state. They never assert a contract themselves, so a test that
uses them still says in its own body what must be observed.

Error bodies come in two shapes on the management routes, and a rewrite must keep the shape per route:
an `HTTPException` raised straight out of a handler is FastAPI's `{"detail": ...}`, while anything that goes
through `handle_exception_on_proxy` or the auth layer is the OpenAI shaped
`{"error": {"message", "type", "param", "code"}}` (`litellm/proxy/proxy_server.py` `openai_exception_handler`).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, object_value, string_value
from tests.integration._support.database import read_rows

PROXY_ADMIN_ONLY_PREFIX: Final = (
    "Authentication Error, Only proxy admin can be used to generate, delete, update info for new keys/users/teams. "
)
NOT_ALLOWED_ACCESS: Final = "Admin-only endpoint. Not allowed to access this."


@dataclass(frozen=True, slots=True)
class Caller:
    """A user row plus a virtual key bound to it."""

    user_id: str
    key: str


def unique(prefix: str) -> str:
    return f"dec1-mgmt-{prefix}-{uuid.uuid4().hex[:12]}"


def digest(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def body_of(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(response.json())


def proxy_error(response: httpx.Response) -> dict[str, JsonValue]:
    """The `error` object of an OpenAI shaped error body."""
    return object_value(body_of(response)["error"])


def detail_error(response: httpx.Response) -> str:
    """The `detail.error` string of a FastAPI `HTTPException(detail={"error": ...})` body."""
    return string_value(object_value(body_of(response)["detail"])["error"])


def error_text(response: httpx.Response) -> str:
    """Every message string in the body, whichever envelope carried it, for substring checks."""
    return json.dumps(response.json())


def caller(scenario: Scenario, role: str, **user_fields: JsonValue) -> Caller:
    """A user with `role` and a personal key, both cleaned up with the scenario."""
    user_id: Final = scenario.user(user_role=role, **user_fields)
    return Caller(user_id, scenario.key(user_id=user_id))


def team_admin(scenario: Scenario, team_id: str) -> Caller:
    """An internal user who is an `admin` member of `team_id`, with a key on that team."""
    user_id: Final = scenario.member(team_id, role="admin")
    return Caller(user_id, scenario.key(user_id=user_id, team_id=team_id))


def org_admin(scenario: Scenario, organization_id: str) -> Caller:
    """An internal user with an `org_admin` membership in `organization_id`, with a personal key."""
    user_id: Final = scenario.org_member(organization_id, "org_admin")
    return Caller(user_id, scenario.key(user_id=user_id))


def customer(scenario: Scenario, **fields: JsonValue) -> str:
    """Create a customer through `/customer/new`; deleted with the scenario unless the test already did."""
    identity: Final = unique("customer")
    scenario.gateway.post("/customer/new", {"user_id": identity, **fields})
    scenario.cleanups.callback(_delete_customer_if_present, scenario.gateway, identity)
    return identity


def _delete_customer_if_present(gateway: Gateway, identity: str) -> None:
    if customer_rows(identity):
        gateway.post("/customer/delete", {"user_ids": [identity]})


def tag(scenario: Scenario, **fields: JsonValue) -> str:
    """Create a tag through `/tag/new`; deleted with the scenario unless the test already did."""
    name: Final = unique("tag")
    scenario.gateway.post("/tag/new", {"name": name, **fields})
    scenario.cleanups.callback(_delete_tag_if_present, scenario.gateway, name)
    return name


def _delete_tag_if_present(gateway: Gateway, name: str) -> None:
    if read_rows('SELECT tag_name FROM "LiteLLM_TagTable" WHERE tag_name = %s', (name,)):
        gateway.post("/tag/delete", {"name": name})


def delete_budget_if_present(gateway: Gateway, budget_id: str) -> None:
    if budget_rows(budget_id):
        gateway.post("/budget/delete", {"id": budget_id})


def customer_rows(identity: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT user_id, alias, blocked, spend, budget_id, models FROM "LiteLLM_EndUserTable" WHERE user_id = %s',
        (identity,),
    )


def budget_rows(budget_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        "SELECT budget_id, max_budget, soft_budget, tpm_limit, rpm_limit, budget_duration, "
        "budget_reset_at::text AS budget_reset_at, created_by, updated_by "
        'FROM "LiteLLM_BudgetTable" WHERE budget_id = %s',
        (budget_id,),
    )


def key_rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        "SELECT token, key_name, key_alias, user_id, team_id, blocked, budget_id "
        'FROM "LiteLLM_VerificationToken" WHERE token = %s',
        (digest(key),),
    )


def deleted_key_rows(hashed: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT token, key_alias, deleted_by FROM "LiteLLM_DeletedVerificationToken" WHERE token = %s',
        (hashed,),
    )


def deprecated_key_rows(hashed: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT token, active_token_id, revoke_at::text AS revoke_at FROM "LiteLLM_DeprecatedVerificationToken" '
        "WHERE token = %s",
        (hashed,),
    )


def user_rows(user_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT user_id, user_email, user_role, teams, max_budget FROM "LiteLLM_UserTable" WHERE user_id = %s',
        (user_id,),
    )


def organization_rows(organization_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT organization_id, organization_alias, budget_id FROM "LiteLLM_OrganizationTable" '
        "WHERE organization_id = %s",
        (organization_id,),
    )


def team_rows(team_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT team_id, organization_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))


def org_membership_rows(organization_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT user_id, user_role FROM "LiteLLM_OrganizationMembership" WHERE organization_id = %s',
        (organization_id,),
    )


def project_rows(project_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT project_id, team_id, budget_id FROM "LiteLLM_ProjectTable" WHERE project_id = %s', (project_id,)
    )


def tag_rows(name: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT tag_name, description, models, budget_id, created_by FROM "LiteLLM_TagTable" WHERE tag_name = %s',
        (name,),
    )


def get(
    gateway: Gateway, path: str, params: Mapping[str, str] | None = None, *, key: str | None = None
) -> httpx.Response:
    return gateway.request("GET", path, params=params, key=key)


def post(gateway: Gateway, path: str, body: Mapping[str, JsonValue], *, key: str | None = None) -> httpx.Response:
    return gateway.request("POST", path, body, key=key)
