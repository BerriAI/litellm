import json
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import yaml
from pydantic import JsonValue

from integration._support.client import Gateway, Scenario, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request

MANAGEMENT_ROUTES: Final = ["/key/*", "/team/new", "/team/update", "/v1/chat/completions"]


def denying_guardrail(request: Request) -> Reply:
    assert request.target == "/beta/litellm_basic_guardrail_api"
    return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": "synthetic policy denial"}).encode())


def guardrail_config(policy_url: str, path: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [
        {
            "guardrail_name": "guardrail" + uuid.uuid4().hex,
            "litellm_params": {
                "guardrail": "generic_guardrail_api",
                "mode": "pre_call",
                "default_on": True,
                "api_base": policy_url,
                "api_key": "synthetic-guardrail-key",
            },
        }
    ]
    path.write_text(yaml.safe_dump(config))
    return path


def stored_metadata(token: str) -> dict[str, object]:
    rows: Final = read_rows(
        'SELECT metadata FROM "LiteLLM_VerificationToken" WHERE token = %s', (sha256(token.encode()).hexdigest(),)
    )
    assert len(rows) == 1, rows
    return rows[0]["metadata"]


def non_admin_caller(scenario: Scenario, member: str, team: str, model: str) -> str:
    return scenario.key(user_id=member, team_id=team, models=[model], allowed_routes=MANAGEMENT_ROUTES)


def chat(candidate: Gateway, model: str, key: str, marker: str, *, stream: bool = False) -> httpx.Response:
    return candidate.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": marker}], "stream": stream},
        key=key,
    )


def upstream_observations(gateway: Gateway) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(timeout=5, trust_env=False) as client:
        drained: Final = object_value(client.get(f"{gateway.upstream_url}/__observations").json())
    requests: Final = drained["requests"]
    assert isinstance(requests, list)
    return tuple(object_value(entry) for entry in requests)


def upstream_hits(gateway: Gateway, marker: str) -> int:
    return sum(1 for entry in upstream_observations(gateway) if marker in json.dumps(entry.get("body")))
