import json
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_KEY_ROUTES: Final = ["/key/generate", "/key/update", "/key/regenerate", "/v1/chat/completions"]


def _denying_guardrail(request: Request) -> Reply:
    assert request.target == "/beta/litellm_basic_guardrail_api"
    return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": "synthetic policy denial"}).encode())


def _default_on_guardrail_config(policy_url: str, path: Path) -> Path:
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


def _stored_metadata(token: str) -> dict[str, object]:
    rows: Final = read_rows(
        'SELECT metadata FROM "LiteLLM_VerificationToken" WHERE token = %s', (sha256(token.encode()).hexdigest(),)
    )
    assert len(rows) == 1, rows
    return rows[0]["metadata"]


@pytest.mark.covers("mgmt.key.disable_global_guardrails.non_admin_denied_and_default_on_guardrail_still_runs")
def test_non_admin_cannot_opt_key_out_of_default_on_guardrail(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_denying_guardrail) as policy:
        config: Final = _default_on_guardrail_config(policy.url, tmp_path / "default_on.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            member: Final = scenario.user(user_role="internal_user")
            team: Final = scenario.team(models=[model], members_with_roles=[{"role": "admin", "user_id": member}])
            caller: Final = scenario.key(user_id=member, models=[model], allowed_routes=_KEY_ROUTES)
            own: Final = scenario.key(team_id=team, models=[model])

            plain: Final = candidate.request("POST", "/key/generate", {"team_id": team, "models": [model]}, key=caller)
            assert plain.status_code == 200, plain.text
            scenario.cleanups.callback(scenario.delete_key, string_value(plain.json()["key"]))

            generated: Final = candidate.request(
                "POST",
                "/key/generate",
                {"team_id": team, "models": [model], "disable_global_guardrails": True},
                key=caller,
            )
            if generated.status_code == 200:
                scenario.cleanups.callback(scenario.delete_key, string_value(generated.json()["key"]))
            assert generated.status_code == 403, generated.text
            assert "disable_global_guardrails" in generated.text

            smuggled: Final = candidate.request(
                "POST",
                "/key/generate",
                {"team_id": team, "models": [model], "metadata": {"disable_global_guardrails": True}},
                key=caller,
            )
            if smuggled.status_code == 200:
                scenario.cleanups.callback(scenario.delete_key, string_value(smuggled.json()["key"]))
            assert smuggled.status_code == 403, smuggled.text

            updated: Final = candidate.request(
                "POST", "/key/update", {"key": own, "disable_global_guardrails": True}, key=caller
            )
            assert updated.status_code == 403, updated.text
            regenerated: Final = candidate.request(
                "POST", "/key/regenerate", {"key": own, "disable_global_guardrails": True}, key=caller
            )
            assert regenerated.status_code == 403, regenerated.text
            assert "disable_global_guardrails" not in _stored_metadata(own)

            blocked: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "synthetic denied marker"}]},
                key=own,
            )
            assert blocked.status_code == 400 and "synthetic policy denial" in blocked.text, blocked.text

            exempt: Final = scenario.key(team_id=team, models=[model], disable_global_guardrails=True)
            assert _stored_metadata(exempt)["disable_global_guardrails"] is True
            resaved: Final = candidate.request(
                "POST",
                "/key/update",
                {
                    "key": exempt,
                    "key_alias": "renamed" + uuid.uuid4().hex,
                    "metadata": {"disable_global_guardrails": True},
                },
                key=caller,
            )
            assert resaved.status_code == 200, resaved.text
            assert _stored_metadata(exempt)["disable_global_guardrails"] is True
            served: Final = candidate.chat(model, key=exempt, text="synthetic denied marker")
            assert object_value(served["usage"])["total_tokens"] == 40
            assert len(policy.drain()) == 1
