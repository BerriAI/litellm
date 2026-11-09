import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import Gateway, object_value, string_value
from tests.integration._support.process import owned_proxy

WINDOW: Final = {"start_date": "2026-01-01", "end_date": "2026-01-07"}


def listed_guardrail(gateway: Gateway, guardrail_name: str) -> dict[str, JsonValue]:
    listed: Final = gateway.get("/v2/guardrails/list")["guardrails"]
    assert isinstance(listed, list), listed
    matches: Final = tuple(object_value(row) for row in listed if object_value(row)["guardrail_name"] == guardrail_name)
    assert len(matches) == 1, f"{guardrail_name} appears {len(matches)} times in {listed}"
    return matches[0]


@pytest.mark.covers("mgmt.guardrails.usage.config_yaml_guardrail_has_detail_and_overview_row")
def test_config_yaml_guardrail_is_served_by_usage_detail_and_overview(gateway: Gateway, tmp_path: Path) -> None:
    guardrail_name: Final = "tool-permission-" + uuid.uuid4().hex
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [
        {
            "guardrail_name": guardrail_name,
            "litellm_params": {
                "guardrail": "tool_permission",
                "mode": "post_call",
                "default_on": False,
                "rules": [{"id": "deny_delete", "tool_name": "(?i)^.*(delete|drop).*", "decision": "deny"}],
                "default_action": "allow",
                "on_disallowed_action": "block",
            },
            "guardrail_info": {"type": "Tool Permission", "description": "declared in config.yaml"},
        }
    ]
    path: Final = tmp_path / "guardrail.yaml"
    path.write_text(yaml.safe_dump(config))
    with owned_proxy(gateway, tmp_path, {}, config=path) as candidate:
        guardrail_id: Final = string_value(listed_guardrail(candidate, guardrail_name)["guardrail_id"])
        detail: Final = candidate.request("GET", f"/guardrails/usage/detail/{guardrail_id}", params=WINDOW)
        assert detail.status_code == 200, detail.text
        body: Final = object_value(detail.json())
        assert {
            "guardrail_id": body["guardrail_id"],
            "guardrail_name": body["guardrail_name"],
            "provider": body["provider"],
            "type": body["type"],
            "description": body["description"],
            "requestsEvaluated": body["requestsEvaluated"],
            "failRate": body["failRate"],
        } == {
            "guardrail_id": guardrail_id,
            "guardrail_name": guardrail_name,
            "provider": "tool_permission",
            "type": "Tool Permission",
            "description": "declared in config.yaml",
            "requestsEvaluated": 0,
            "failRate": 0.0,
        }, detail.text
        overview: Final = candidate.request("GET", "/guardrails/usage/overview", params=WINDOW)
        assert overview.status_code == 200, overview.text
        rows: Final = object_value(overview.json())["rows"]
        assert isinstance(rows, list), overview.text
        config_rows: Final = tuple(object_value(row) for row in rows if object_value(row)["id"] == guardrail_id)
        assert len(config_rows) == 1, overview.text
        assert (config_rows[0]["name"], config_rows[0]["provider"], config_rows[0]["requestsEvaluated"]) == (
            guardrail_name,
            "tool_permission",
            0,
        ), overview.text
        missing: Final = candidate.request("GET", f"/guardrails/usage/detail/{uuid.uuid4()}", params=WINDOW)
        assert missing.status_code == 404, missing.text
