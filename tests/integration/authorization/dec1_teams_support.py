"""Shared helpers for the `test_dec1_teams_*` files: teams as request time context.

Team, team membership and organization lookups are moving behind Rust stores (#52176). These helpers drive the
proxy over HTTP like a customer would and read Postgres or the scripted upstream only to observe an effect.
Nothing here imports or patches proxy internals.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value
from tests.integration.authorization.dec1_keys_support import CHAT_PATH, digest, error_of

#: A management write must reach the writer proxy within this many seconds (the auth cache TTL is 60).
WRITER_SECONDS: Final = 5
#: How long a spend row may take to land (batch writer interval plus margin), as the other spend tests use.
SPEND_ROW_SECONDS: Final = 70
#: `_PROMPT_TOKENS` and `_COMPLETION_TOKENS` of the scripted upstream (tests/_fake_openai_endpoint_server.py:36).
UPSTREAM_PROMPT_TOKENS: Final = 20
UPSTREAM_COMPLETION_TOKENS: Final = 20
#: `_CANNED_CONTENT` of the scripted upstream (tests/_fake_openai_endpoint_server.py:32).
UPSTREAM_CONTENT: Final = "Hello! This is a mock response from the fake OpenAI endpoint."
ATTEMPTED_FALLBACKS_HEADER: Final = "x-litellm-attempted-fallbacks"

__all__ = (
    "ATTEMPTED_FALLBACKS_HEADER",
    "CHAT_PATH",
    "SPEND_ROW_SECONDS",
    "UPSTREAM_COMPLETION_TOKENS",
    "UPSTREAM_CONTENT",
    "UPSTREAM_PROMPT_TOKENS",
    "WRITER_SECONDS",
    "answer_text",
    "chat_as",
    "digest",
    "error_of",
    "marker",
    "row_metadata",
    "scripted_model",
    "settled",
    "spend_rows_of",
    "team_blocked_message",
    "team_row",
    "upstream_bodies",
)


def marker() -> str:
    return f"dec1-teams-{uuid.uuid4().hex}"


def team_blocked_message(team_id: str) -> str:
    """`common_checks` team block wording (litellm/proxy/auth/auth_checks.py:1251) behind the auth prefix."""
    return f"Authentication Error, Team={team_id} is blocked. Update via `/team/unblock` if you're an admin."


def chat_as(
    gateway: Gateway,
    key: str,
    model: str,
    *,
    text: str | None = None,
    headers: Mapping[str, str] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": text or marker()}],
        **extra,
    }
    return gateway.request("POST", CHAT_PATH, body, key=key, headers=headers)


def settled(call: object, status: int, seconds: float = WRITER_SECONDS) -> httpx.Response:
    """Repeat `call` until it answers `status`; `eventually` fails the test with the last answer otherwise."""
    assert callable(call)
    return eventually(call, lambda response: response.status_code == status, seconds=seconds)


def answer_text(response: httpx.Response) -> str:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    choices: Final = JSON_OBJECT.validate_json(response.content)["choices"]
    assert isinstance(choices, list) and len(choices) == 1, response.text
    content: Final = object_value(object_value(choices[0])["message"])["content"]
    assert isinstance(content, str), response.text
    return content


def spend_rows_of(gateway: Gateway, key: str, *, at_least: int = 1) -> list[dict[str, JsonValue]]:
    """Every spend row the key produced, oldest first, once `at_least` of them landed."""
    return eventually(
        lambda: gateway.rows(
            'SELECT request_id, api_key, "user", team_id, organization_id, request_tags, metadata, status, spend, '
            'model_group, model FROM "LiteLLM_SpendLogs" WHERE api_key=%s ORDER BY "startTime"',
            (digest(key),),
        ),
        lambda rows: len(rows) >= at_least,
        seconds=SPEND_ROW_SECONDS,
    )


def row_metadata(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """The spend row `metadata` column as an object, whether the driver returns JSON text or a decoded value."""
    raw: Final = row["metadata"]
    return object_value(json.loads(raw) if isinstance(raw, str) else raw)


def team_row(gateway: Gateway, team_id: str) -> dict[str, JsonValue]:
    rows: Final = gateway.rows(
        'SELECT team_id, team_alias, spend, blocked, metadata FROM "LiteLLM_TeamTable" WHERE team_id=%s', (team_id,)
    )
    assert len(rows) == 1, rows
    return rows[0]


def upstream_bodies(gateway: Gateway, text: str) -> list[dict[str, JsonValue]]:
    """Bodies of the requests the scripted upstream received whose body mentions `text`."""
    observed: Final = httpx.get(f"{gateway.upstream_url}/__observations", timeout=15, trust_env=False)
    observed.raise_for_status()
    requests: Final = JSON_OBJECT.validate_json(observed.content)["requests"]
    assert isinstance(requests, list)
    return [
        object_value(object_value(request)["body"])
        for request in requests
        if text in json.dumps(object_value(request)["body"])
    ]


def scripted_model(scenario: Scenario, statuses: list[int], **parameters: JsonValue) -> str:
    """A deployment whose upstream model name is unique and answers the scripted statuses in order, then 200s.

    Uses the scripted upstream's `/__scripts/<model>` control (tests/integration/_support/upstream.py `script`);
    once the script is exhausted the upstream answers 500 "Script exhausted", so a test scripts every attempt.
    """
    upstream_model: Final = f"dec1-teams-{uuid.uuid4().hex}"
    script_url: Final = f"{scenario.gateway.upstream_url}/__scripts/{upstream_model}"
    configured: Final = httpx.post(script_url, json={"statuses": statuses}, timeout=15, trust_env=False)
    assert configured.status_code == 200, configured.text
    scenario.cleanups.callback(httpx.delete, script_url)
    return scenario.model(model=f"openai/{upstream_model}", **parameters)
