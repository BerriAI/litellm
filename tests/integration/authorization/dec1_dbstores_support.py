"""Shared helpers for the `test_dec1_dbstores_*` auth object lookup contract files.

The auth object lookups (virtual key, team, team membership, organization, user, end user, budget, project,
tag, object permission) are being moved behind Rust stores in draft PRs #52173, #52176, #52170 and #52178.
These helpers drive the proxy over HTTP like a customer would and read Postgres, Redis or the scripted upstream
only to observe an effect. Nothing here imports or patches proxy internals.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import yaml
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually, string_value
from tests.integration.authorization.dec1_keys_support import (
    BASE_CONFIG,
    CHAT_PATH,
    assert_denied,
    digest,
    error_of,
)

#: The writer proxy must show a management change within this many seconds, the peer within `PEER_SECONDS`.
#: Both are far below the 60 second auth cache TTL, so a port that drops an eviction or its broadcast fails.
WRITER_SECONDS: Final = 5
PEER_SECONDS: Final = 10
#: `DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL` and `UserAPIKeyCacheTTLEnum.in_memory_cache_ttl`.
DEFAULT_AUTH_CACHE_TTL_SECONDS: Final = 60
#: `litellm.budget_exceeded_status_code` (litellm/__init__.py:393).
BUDGET_EXCEEDED_STATUS: Final = 422


def marker() -> str:
    return f"dec1-dbstores-{uuid.uuid4().hex}"


def chat_as(
    gateway: Gateway,
    key: str,
    model: str,
    *,
    user: str | None = None,
    headers: Mapping[str, str] | None = None,
    text: str | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    """A chat completion with an optional OpenAI `user` field and extra headers next to the bearer key."""
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": text or marker()}],
        **({"user": user} if user is not None else {}),
        **extra,
    }
    return gateway.request("POST", CHAT_PATH, body, key=key, headers=headers)


def settled(call: Callable[[], httpx.Response], status: int, seconds: float = WRITER_SECONDS) -> httpx.Response:
    """Repeat `call` until it answers `status`; `eventually` fails the test with the last answer otherwise."""
    return eventually(call, lambda response: response.status_code == status, seconds=seconds)


def assert_ok(response: httpx.Response) -> None:
    assert response.status_code == 200, f"{response.status_code} {response.text}"


def assert_budget_refused(response: httpx.Response) -> None:
    """A budget refusal from auth or from the admission hold: 422 with `error.type == "budget_exceeded"`.

    The exact wording is pinned by `test_dec1_budgets_boundaries.py`; here only the fact that the budget row
    linked to the object was found and applied matters.
    """
    assert response.status_code == BUDGET_EXCEEDED_STATUS, f"{response.status_code} {response.text}"
    assert error_of(response)["type"] == "budget_exceeded", response.text


def assert_blocked_customer(response: httpx.Response, end_user: str) -> None:
    """The blocked end user refusal pinned in `test_end_user_block.py`: 400 naming the end user."""
    assert response.status_code == 400, f"{response.status_code} {response.text}"
    assert error_of(response)["message"] == f"User blocked from making LLM API Calls. User={end_user}", response.text


def assert_customer_model_denied(response: httpx.Response) -> None:
    assert_denied(response, status=403, error_type="customer_model_access_denied", param="model")


def customer(scenario: Scenario, **fields: JsonValue) -> str:
    """An end user created through `POST /customer/new`, deleted through `/customer/delete` afterwards."""
    identity: Final = string_value(fields.pop("user_id", None) or f"dec1-dbstores-customer-{uuid.uuid4().hex}")
    scenario.gateway.post("/customer/new", {"user_id": identity, **fields})
    scenario.cleanups.callback(scenario.gateway.request, "POST", "/customer/delete", {"user_ids": [identity]})
    return identity


def forget_end_user(scenario: Scenario, identity: str) -> None:
    """Remove an end user row the proxy auto created on spend, once the scenario ends."""
    scenario.cleanups.callback(scenario.gateway.request, "POST", "/customer/delete", {"user_ids": [identity]})


def tag(scenario: Scenario, **fields: JsonValue) -> str:
    """A tag created through `POST /tag/new`, deleted through `/tag/delete` afterwards."""
    name: Final = f"dec1-dbstores-tag-{uuid.uuid4().hex[:12]}"
    scenario.gateway.post("/tag/new", {"name": name, **fields})
    scenario.cleanups.callback(scenario.gateway.request, "POST", "/tag/delete", {"name": name})
    return name


def end_user_rows(gateway: Gateway, end_user: str) -> list[dict[str, JsonValue]]:
    return gateway.rows(
        'SELECT user_id, spend, blocked, budget_id, models FROM "LiteLLM_EndUserTable" WHERE user_id=%s', (end_user,)
    )


def spend_end_users(gateway: Gateway, key: str, *, at_least: int = 1, seconds: float = 70) -> list[JsonValue]:
    """The `end_user` column of every spend row the key produced, oldest first, once `at_least` landed."""
    rows: Final = eventually(
        lambda: gateway.rows(
            'SELECT end_user FROM "LiteLLM_SpendLogs" WHERE api_key=%s ORDER BY "startTime"', (digest(key),)
        ),
        lambda found: len(found) >= at_least,
        seconds=seconds,
    )
    return [row["end_user"] for row in rows]


def config_with_settings(
    directory: Path,
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    name: str = "dec1-dbstores.yaml",
) -> Path:
    """The suite config with extra `general_settings` and `litellm_settings` merged over it."""
    config: Final = yaml.safe_load(BASE_CONFIG.read_text())
    config["general_settings"] = {**(config.get("general_settings") or {}), **(general_settings or {})}
    config["litellm_settings"] = {**(config.get("litellm_settings") or {}), **(litellm_settings or {})}
    directory.mkdir(parents=True, exist_ok=True)
    path: Final = directory / name
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class LookupOutage:
    """A private proxy on a scratch database behind a relay that drops any connection whose outbound bytes
    contain `trigger` once armed. Embedding `trigger` in one object id (a team id, a customer id, ...) makes
    exactly the lookups that name that object fail, while every other query keeps working."""

    proxy: Gateway
    scenario: Scenario
    trigger: str
    arm: Callable[[], None]
    disarm: Callable[[], None]
    dropped: Callable[[], bool]


@contextmanager
def lookup_outage(
    gateway: Gateway,
    directory: Path,
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
) -> Iterator[LookupOutage]:
    """Boot the rig. The relay is disarmed during setup and teardown so management writes never fail."""
    from tests.integration._support.database import scratch_database
    from tests.integration._support.database_relay import dropped_connection_relay, plaintext_database_url
    from tests.integration._support.process import owned_proxy

    trigger: Final = f"dec1fail{uuid.uuid4().hex[:16]}"
    with ExitStack() as stack:
        database_url: Final = stack.enter_context(scratch_database())
        relay, relayed_url = stack.enter_context(dropped_connection_relay(database_url, trigger.encode()))
        config: Final = config_with_settings(
            directory, general_settings=general_settings, litellm_settings=litellm_settings
        )
        proxy: Final = stack.enter_context(
            owned_proxy(
                gateway,
                directory / "proxy",
                {"DATABASE_URL": plaintext_database_url(relayed_url), "PRISMA_HEALTH_WATCHDOG_ENABLED": "false"},
                config=config,
                remove_environment=("DATABASE_URL_READ_REPLICA",),
            )
        )
        scenario: Final = stack.enter_context(proxy.scenario())
        stack.callback(relay.disarm)
        yield LookupOutage(proxy, scenario, trigger, relay.arm, relay.disarm, relay.dropped.is_set)
