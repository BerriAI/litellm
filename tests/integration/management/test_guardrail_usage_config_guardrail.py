import json
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, object_value, string_value
from tests.integration._support.database import scratch_database
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

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


def _config_file(directory: Path, guardrail: dict[str, JsonValue] | None) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    if guardrail is not None:
        policy: Final = f"policy-{guardrail['guardrail_name']}"
        config["guardrails"] = [guardrail]
        config["policies"] = {policy: {"guardrails": {"add": [guardrail["guardrail_name"]]}}}
        config["policy_attachments"] = [{"policy": policy, "scope": "*"}]
    directory.mkdir()
    path: Final = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _content_filter(guardrail_name: str) -> dict[str, JsonValue]:
    return {
        "guardrail_name": guardrail_name,
        "litellm_params": {
            "guardrail": "litellm_content_filter",
            "mode": "pre_call",
            "patterns": [
                {"pattern_type": "regex", "pattern": "tok_[a-z0-9]{12}", "name": "api_token", "action": "MASK"}
            ],
        },
    }


def _generic_api(guardrail_name: str, api_base: str) -> dict[str, JsonValue]:
    return {
        "guardrail_name": guardrail_name,
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "api_base": api_base,
            "api_key": "synthetic-guardrail-key",
        },
    }


def _allows(request: Request) -> Reply:
    return Reply(body=json.dumps({"action": "NONE"}).encode())


def _rows_with_runs(
    gateway: Gateway, window: dict[str, str], ids: tuple[str, ...]
) -> list[tuple[JsonValue, JsonValue]]:
    rows: Final = gateway.get("/guardrails/usage/overview", window)["rows"]
    assert isinstance(rows, list), rows
    return [
        (object_value(row)["id"], object_value(row)["requestsEvaluated"])
        for row in rows
        if object_value(row)["id"] in ids
    ]


@pytest.mark.parametrize(
    ("declared_guardrail", "runs_stored_under"),
    [(lambda name, _vendor: _content_filter(name), "guardrail_id"), (_generic_api, "guardrail_name")],
    ids=["content_filter_stores_runs_under_its_id", "generic_api_stores_runs_under_its_name"],
)
def test_usage_detail_serves_the_stored_runs_of_a_guardrail_removed_from_config(
    gateway: Gateway,
    tmp_path: Path,
    declared_guardrail: Callable[[str, str], dict[str, JsonValue]],
    runs_stored_under: str,
) -> None:
    guardrail_name: Final = "api-token-redact-" + uuid.uuid4().hex
    today: Final = datetime.now(timezone.utc).date()
    window: Final = {
        "start_date": (today - timedelta(days=1)).isoformat(),
        "end_date": (today + timedelta(days=1)).isoformat(),
    }
    runs: Final = 3
    with wire_server(_allows) as vendor, scratch_database("removed_guardrail") as database_url:
        one_database: Final = {"DATABASE_URL": database_url}
        declared: Final = _config_file(tmp_path / "declared", declared_guardrail(guardrail_name, vendor.url))
        with owned_proxy(gateway, declared.parent, one_database, config=declared) as before:
            guardrail_id: Final = string_value(listed_guardrail(before, guardrail_name)["guardrail_id"])
            with before.scenario() as scenario:
                model: Final = scenario.model()
                for _ in range(runs):
                    before.chat(model, text="rotate tok_abcdef123456 today")
                stored: Final = eventually(
                    lambda: before.get(f"/guardrails/usage/detail/{guardrail_id}", window),
                    lambda body: body["requestsEvaluated"] == runs,
                    seconds=70,
                )
                assert before.get(f"/guardrails/usage/detail/{guardrail_name}", window) == stored
                assert before.get("/guardrails/usage/logs", {"guardrail_id": guardrail_name})["total"] == runs
        removed: Final = _config_file(tmp_path / "removed", None)
        with owned_proxy(gateway, removed.parent, one_database, config=removed) as after:
            listed: Final = after.get("/v2/guardrails/list")["guardrails"]
            assert isinstance(listed, list), listed
            assert guardrail_name not in [object_value(row)["guardrail_name"] for row in listed], listed
            stored_under: Final = {"guardrail_id": guardrail_id, "guardrail_name": guardrail_name}[runs_stored_under]
            assert _rows_with_runs(after, window, (guardrail_id, guardrail_name)) == [(stored_under, runs)]
            logs: Final = after.get("/guardrails/usage/logs", {"guardrail_id": stored_under})
            assert logs["total"] == runs, logs
            detail: Final = after.request("GET", f"/guardrails/usage/detail/{stored_under}", params=window)
            assert detail.status_code == 200, detail.text
            assert object_value(detail.json()) == {
                **stored,
                "guardrail_id": stored_under,
                "guardrail_name": stored_under,
                "provider": "Unknown",
                "type": "Guardrail",
                "description": None,
            }, detail.text
            outside_window: Final = after.get(f"/guardrails/usage/detail/{stored_under}", WINDOW)
            assert (outside_window["requestsEvaluated"], outside_window["time_series"]) == (0, []), outside_window
            missing: Final = after.request("GET", f"/guardrails/usage/detail/{uuid.uuid4()}", params=window)
            assert missing.status_code == 404, missing.text
