"""Shared building blocks for the `test_dec1_budgets3_*` contract files (budgets, round 3).

Round 3 pins what rounds 1 and 2 (`test_dec1_budgets_*`, `test_dec1_budgets2_*`) listed as not covered: budgets on
every entity level of one request at once and which level refuses first, hold sizing details, refunds on paths that
never reach a normal settle (provider errors on every level, cache hits, limiter refusals after the hold, retries
and fallbacks), budget changes through the management API taking effect, Redis outages, and the routes that run
budget checks without taking a hold.

Everything reuses the round 1 price (`dec1_budgets_support.hold_model`: input free, output 0.004 a token, the
scripted provider answers 5 completion tokens, so one served chat call costs 0.02 and a hold is exactly
`max_tokens * 0.004`). Counter names are the ones built in `litellm/redis/stores/spend_counters/types.py:51-89`:

    spend:key:<sha256>                spend:key:<sha256>:window:<duration>
    spend:team:<team_id>              spend:team:<team_id>:window:<duration>
    spend:team_member:<user>:<team>   spend:user:<user_id>
    spend:org:<org_id>                spend:project:<project_id>
    spend:tag:<tag>                   spend:end_user:<end_user_id>
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.spend.dec1_budgets_support import call, hashed, key_counter
from tests.integration.spend.dec1_spend_support import new_customer, new_tag, respond, served

TOKENS: Final = 10
HOLD: Final = 0.04
COST: Final = 0.02

__all__ = [
    "COST",
    "HOLD",
    "TOKENS",
    "FlakyProvider",
    "Levels",
    "end_user_counter",
    "flaky_provider",
    "key_counter",
    "key_window_counter",
    "levels",
    "org_counter",
    "project_counter",
    "spend_once",
    "tag_counter",
    "team_counter",
    "team_member_counter",
    "team_window_counter",
    "user_counter",
]


def team_counter(team_id: str) -> str:
    return f"spend:team:{team_id}"


def team_window_counter(team_id: str, duration: str) -> str:
    return f"spend:team:{team_id}:window:{duration}"


def key_window_counter(key: str, duration: str) -> str:
    return f"spend:key:{hashed(key)}:window:{duration}"


def team_member_counter(user_id: str, team_id: str) -> str:
    return f"spend:team_member:{user_id}:{team_id}"


def user_counter(user_id: str) -> str:
    return f"spend:user:{user_id}"


def org_counter(org_id: str) -> str:
    return f"spend:org:{org_id}"


def project_counter(project_id: str) -> str:
    return f"spend:project:{project_id}"


def tag_counter(tag: str) -> str:
    return f"spend:tag:{tag}"


def end_user_counter(end_user_id: str) -> str:
    return f"spend:end_user:{end_user_id}"


def spend_once(gateway: Gateway, model: str, key: str, **extra: JsonValue) -> None:
    """One served call with `max_tokens: 10` (hold 0.04, real cost 0.02)."""
    served(call(gateway, model, key, max_tokens=TOKENS, **extra))


@dataclass(frozen=True, slots=True)
class Levels:
    """One key that sits under every budgeted entity a request can carry: organization, team, team member,
    project, tag (sent in `metadata.tags`) and end user (sent as the body `user`).

    A budget of `None` leaves that level without a budget. The key's own budget is `key_budget`."""

    gateway: Gateway
    model: str
    key: str
    org_id: str
    team_id: str
    user_id: str
    project_id: str
    tag: str
    customer: str

    def extra(self) -> dict[str, JsonValue]:
        return {"metadata": {"tags": [self.tag]}, "user": self.customer}

    def call(self, *, hold: bool = False, max_tokens: int | None = TOKENS) -> httpx.Response:
        return call(self.gateway, self.model, self.key, hold=hold, max_tokens=max_tokens, **self.extra())

    def counters(self) -> dict[str, str]:
        """Every counter this key's call may hold on, by level name."""
        return {
            "key": key_counter(self.key),
            "team": team_counter(self.team_id),
            "team_member": team_member_counter(self.user_id, self.team_id),
            "org": org_counter(self.org_id),
            "project": project_counter(self.project_id),
            "tag": tag_counter(self.tag),
            "end_user": end_user_counter(self.customer),
        }


def _budget(value: float | None) -> dict[str, JsonValue]:
    return {} if value is None else {"max_budget": value}


def levels(
    gateway: Gateway,
    scenario: Scenario,
    model: str,
    *,
    key_budget: float | None = None,
    team_budget: float | None = None,
    member_budget: float | None = None,
    org_budget: float | None = None,
    project_budget: float | None = None,
    tag_budget: float | None = None,
    customer_budget: float | None = None,
) -> Levels:
    """Build the entity tree of `Levels`. The member budget is the team's default `team_member_budget`, the path
    round 1 already exercised (`test_a_default_team_member_budget_refuses_a_member_who_used_it_up`)."""
    org_id: Final = scenario.organization(models=[model], **_budget(org_budget))
    team_fields: Final[dict[str, JsonValue]] = {"organization_id": org_id, "models": [model], **_budget(team_budget)}
    if member_budget is not None:
        team_fields["team_member_budget"] = member_budget
    team_id: Final = scenario.team(**team_fields)
    user_id: Final = scenario.member(team_id)
    project_id: Final = scenario.project(team_id, models=[model], **_budget(project_budget))
    tag: Final = new_tag(gateway, scenario, **_budget(tag_budget))
    customer: Final = new_customer(gateway, scenario, **_budget(customer_budget))
    key: Final = scenario.key(
        user_id=user_id, team_id=team_id, project_id=project_id, models=[model], **_budget(key_budget)
    )
    return Levels(gateway, model, key, org_id, team_id, user_id, project_id, tag, customer)


@dataclass
class FlakyProvider:
    """A scripted provider that fails the first `failures` inference calls with `status`, then answers like
    `dec1_spend_support.respond` (5 completion tokens). Used for retry and fallback settlement."""

    failures: int
    status: int = 500
    wire: Wire | None = None
    seen: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def answer(self, request: Request) -> Reply:
        if request.method != "POST":
            return respond(request)
        with self.lock:
            self.seen += 1
            failing = self.seen <= self.failures
        if failing:
            return Reply(
                status=self.status,
                body=b'{"error":{"message":"dec1 budgets3 scripted failure","type":"server_error","code":"500"}}',
            )
        return respond(request)


@contextmanager
def flaky_provider(failures: int, status: int = 500) -> Iterator[FlakyProvider]:
    provider: Final = FlakyProvider(failures=failures, status=status)
    with wire_server(provider.answer) as wire:
        provider.wire = wire
        yield provider
