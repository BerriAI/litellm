"""Shared building blocks for the `test_dec1_budgets2_*` contract files (budgets, round 2).

Round 2 covers what `test_dec1_budgets_*` listed as not covered: customer hierarchy window counters, model access
group holds, billing agent fail closed, hold lease renewal, cancellation settling at the input cost, background
responses, committed twin counters behind the Prometheus remaining budget gauges, soft budget emails and
`projected_limit_exceeded`, user windows on team keys, tiered pricing estimates, several workers of one proxy, and
the reseed of a lost Redis counter from Postgres.

Everything reuses the price of `dec1_budgets_support` (input free, output 0.004 a token, 5 completion tokens per
served chat call, so one call costs 0.02 and a hold is `max_tokens * 0.004`) unless a file says otherwise. Where
input must cost something (cancellation, tiered pricing) the file states its own literal price.

Counter names are the ones built in `litellm/redis/stores/spend_counters/types.py` and
`litellm/rust/crates/redis-store/src/spend_counters/keys.rs`:

    spend:key:<sha256>                          spend:committed:key:<sha256>   (charges only twin)
    spend:team:<team_id>                        spend:user:<user_id>
    spend:user_window:<user_id>:<duration>      spend:model_access_group:<group>
    spend:customer_org:<quoted id>:window:<1mo|12mo>@<window start, UTC isoformat>
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final
from urllib.parse import quote

import httpx
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire
from tests.integration.spend.dec1_budgets_support import counter
from tests.integration.spend.dec1_spend_support import LANDING_SECONDS, hashed, proxy_config, respond

CUSTOMERS: Final = "/management/v1"
MEMBER_HEADER: Final = "x-litellm-customer-member-id"
ORG_ALIAS_HEADER: Final = "x-litellm-customer-org-alias"
TEAM_HEADER: Final = "x-litellm-customer-team-id"
INCREASE_HINT: Final = " Request a temporary increase through POST /management/v1/customer_budget_increase_requests."
RESET_HINT: Final = " Spend resumes when the budget window resets."
FAST_RESET: Final = {"PROXY_BUDGET_RESCHEDULER_MIN_TIME": "2", "PROXY_BUDGET_RESCHEDULER_MAX_TIME": "3"}

__all__ = [
    "CUSTOMERS",
    "FAST_RESET",
    "INCREASE_HINT",
    "MEMBER_HEADER",
    "ORG_ALIAS_HEADER",
    "RESET_HINT",
    "TEAM_HEADER",
    "Customer",
    "committed_key_counter",
    "customer",
    "customer_window_keys",
    "gauge",
    "metrics_text",
    "owned",
    "responses_provider",
    "ttl_of",
    "user_window_counter",
]


def committed_key_counter(key: str) -> str:
    """The charges-only twin of `spend:key:<hash>` (`committed_spend.py::committed_spend_key`)."""
    return f"spend:committed:key:{hashed(key)}"


def user_window_counter(user_id: str, duration: str) -> str:
    """`types.py::user_window_counter`, its own prefix since #51763 so a user id cannot spill into another's window."""
    return f"spend:user_window:{user_id}:{duration}"


def customer_window_keys(cache: Redis, scope: str, entity_id: str, duration: str = "1mo") -> dict[str, float]:
    """Every window counter of one customer level and its value, found by pattern because the key ends with the
    window start (`types.py::customer_window_counter`)."""
    pattern: Final = f"spend:{scope}:{quote(entity_id, safe='')}:window:{duration}@*"
    found: Final[dict[str, float]] = {}
    for raw in cache.scan_iter(match=pattern):
        name = raw.decode() if isinstance(raw, bytes) else str(raw)
        value = counter(cache, name)
        found[name] = 0.0 if value is None else value
    return found


def ttl_of(cache: Redis, name: str) -> float:
    """Seconds left on a Redis key (-2 when missing, -1 when it has no TTL)."""
    milliseconds: Final = int(cache.pttl(name))
    return float(milliseconds) if milliseconds < 0 else milliseconds / 1000


class Customer:
    """One customer organization with one team and one member, created through the customer management API.

    `org_budget`, `team_budget` and `member_template` are `Budget` objects
    (`types/proxy/management_endpoints/customer_hierarchy.py::Budget`); the member gets the organization's
    default budget template named `member`.
    """

    def __init__(self, gateway: Gateway, scenario: Scenario, suffix: str) -> None:
        self.gateway: Final = gateway
        self.scenario: Final = scenario
        self.org_id: Final = f"dec1-b2-org-{suffix}"
        self.team_id: Final = f"dec1-b2-team-{suffix}"
        self.member_id: Final = f"dec1-b2-member-{suffix}"

    def create(
        self,
        *,
        org_budget: Mapping[str, JsonValue] | None,
        team_budget: Mapping[str, JsonValue] | None,
        member_template: Mapping[str, JsonValue],
        alias: str | None = None,
    ) -> Customer:
        org_body: Final[dict[str, JsonValue]] = {
            "customer_org_id": self.org_id,
            "default_budgets": [{"name": "member", **member_template}],
            "default_budget_template": "member",
        }
        if org_budget is not None:
            org_body["budget"] = dict(org_budget)
        if alias is not None:
            org_body["alias"] = alias
        self._created("customer_organizations", org_body)
        self.scenario.cleanups.callback(
            self.gateway.request, "DELETE", f"{CUSTOMERS}/customer_organizations/{self.org_id}?cascade=true"
        )
        team_body: Final[dict[str, JsonValue]] = {"customer_team_id": self.team_id, "parent_org_id": self.org_id}
        if team_budget is not None:
            team_body["budget"] = dict(team_budget)
        self._created("customer_teams", team_body)
        self._created("customer_members", {"customer_member_id": self.member_id, "parent_team_ids": [self.team_id]})
        return self

    def _created(self, collection: str, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        response: Final = self.gateway.request("POST", f"{CUSTOMERS}/{collection}", dict(body))
        assert response.status_code == 201, f"{collection}: {response.status_code} {response.text}"
        return JSON_OBJECT.validate_json(response.content)

    def read(self, collection: str, identity: str) -> dict[str, JsonValue]:
        response: Final = self.gateway.request("GET", f"{CUSTOMERS}/{collection}/{identity}")
        assert response.status_code == 200, response.text
        return JSON_OBJECT.validate_json(response.content)

    def chat(self, model: str, key: str, *, max_tokens: int | None = 5, **headers: str) -> httpx.Response:
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "messages": [{"role": "user", "content": f"dec1-budgets2 {uuid.uuid4().hex}"}],
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        return self.gateway.request(
            "POST", "/v1/chat/completions", body, key=key, headers={MEMBER_HEADER: self.member_id, **headers}
        )


def customer(gateway: Gateway, scenario: Scenario) -> Customer:
    return Customer(gateway, scenario, uuid.uuid4().hex[:12])


@contextmanager
def owned(
    gateway: Gateway,
    directory: Path,
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    environment: Mapping[str, str] | None = None,
    workers: int = 1,
) -> Iterator[Gateway]:
    """An owned proxy on the suite's database and Redis with settings merged into the shared config."""
    config: Final = proxy_config(directory, general_settings=general_settings, litellm_settings=litellm_settings)
    with owned_proxy(gateway, directory, dict(environment or {}), config=config, workers=workers) as booted:
        yield booted


def metrics_text(gateway: Gateway) -> str:
    response: Final = gateway.client.get("/metrics", headers={"Authorization": f"Bearer {gateway.key}"})
    assert response.status_code == 200, response.text
    return response.text


_SAMPLE: Final = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)\{(?P<labels>[^}]*)\} (?P<value>\S+)$")


def gauge(gateway: Gateway, name: str, **labels: str) -> float | None:
    """The value of the one `/metrics` sample of `name` whose labels include `labels`, None when absent."""
    for line in metrics_text(gateway).splitlines():
        matched = _SAMPLE.match(line)
        if matched is None or matched["name"] != name:
            continue
        pairs = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', matched["labels"]))
        if all(pairs.get(label) == value for label, value in labels.items()):
            return float(matched["value"])
    return None


def gauge_reads(gateway: Gateway, name: str, expected: float, **labels: str) -> float | None:
    return eventually(
        lambda: gauge(gateway, name, **labels),
        lambda value: value is not None and abs(value - expected) < 1e-9,
        seconds=LANDING_SECONDS,
    )


class ResponsesProvider:
    """A scripted OpenAI provider whose `POST /v1/responses` with `background: true` answers `queued` without usage
    and whose `GET /v1/responses/<id>` answers `completed` with 9 input and 4 output tokens. Everything else is
    answered by `dec1_spend_support.respond`."""

    def __init__(self) -> None:
        self.response_id: Final = f"resp_dec1b2{uuid.uuid4().hex}"
        self.reads: int = 0

    def _body(self, status: str, model: JsonValue) -> bytes:
        done: Final = status == "completed"
        return json.dumps(
            {
                "id": self.response_id,
                "object": "response",
                "created_at": int(time.time()),
                "status": status,
                "background": True,
                "model": model,
                "output": [
                    {
                        "type": "message",
                        "id": "msg_dec1b2",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done", "annotations": []}],
                    }
                ]
                if done
                else [],
                "usage": {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13} if done else None,
            }
        ).encode()

    def __call__(self, request: Request) -> Reply:
        path: Final = request.target.split("?", 1)[0]
        if request.method == "GET" and path.endswith(f"/responses/{self.response_id}"):
            self.reads += 1
            return Reply(body=self._body("completed", "gpt-4o-mini"))
        if request.method == "POST" and path.endswith("/responses"):
            body: Final = JSON_OBJECT.validate_json(request.body)
            if body.get("background") is True:
                return Reply(body=self._body("queued", body.get("model")))
        return respond(request)


def responses_provider() -> ResponsesProvider:
    return ResponsesProvider()


def team_key(scenario: Scenario, model: str, **team: JsonValue) -> tuple[str, str]:
    team_id: Final = scenario.team(models=[model], **team)
    return team_id, scenario.key(team_id=team_id, models=[model])


def json_of(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def error_message(response: httpx.Response) -> str:
    return string_value(object_value(json_of(response)["error"])["message"])


def wire_posts(wire: Wire) -> Sequence[Request]:
    return tuple(request for request in wire.drain() if request.method == "POST")
