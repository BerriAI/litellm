import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import execute, read_rows
from integration._support.process import owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

ANTHROPIC_MODEL: Final = "claude-sonnet-4-5-20250929"
MESSAGES_TARGET: Final = "/v1/messages"


def respond(request: Request) -> Reply:
    assert request.method == "POST", request.target
    assert request.target == MESSAGES_TARGET, request.target
    return Reply(
        body=json.dumps(
            {
                "id": "msg_0",
                "type": "message",
                "role": "assistant",
                "model": ANTHROPIC_MODEL,
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 4},
            }
        ).encode()
    )


def interception_config(
    wire_url: str,
    tmp_path: Path,
    litellm_settings: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config: Final = {
        **base,
        "search_tools": [
            {
                "search_tool_name": "integration-search",
                "litellm_params": {
                    "search_provider": "tavily",
                    "api_key": "synthetic-tavily-key",
                    "api_base": wire_url + "/tavily",
                },
            }
        ],
        "general_settings": {**base["general_settings"], "proxy_config_reload_interval_seconds": 1},
        "litellm_settings": {**base["litellm_settings"], **litellm_settings},
        **(extra or {}),
    }
    path: Final = tmp_path / "websearch.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def observed_tool_names(wire: Wire) -> tuple[str, ...]:
    return tuple(
        tool["name"]
        for request in wire.drain()
        if request.target == MESSAGES_TARGET
        for tool in json.loads(request.body)["tools"]
    )


def restore_litellm_settings_row() -> None:
    execute(
        'UPDATE "LiteLLM_Config" SET param_value = param_value - %s WHERE param_name = %s',
        ("websearch_interception_params", "litellm_settings"),
    )
    remaining: Final = read_rows(
        "SELECT param_value ? 'websearch_interception_params' AS present FROM \"LiteLLM_Config\" WHERE param_name = %s",
        ("litellm_settings",),
    )
    assert remaining == [] or remaining == [{"present": False}], remaining


def probe(candidate: Gateway, model: str) -> httpx.Response:
    return candidate.request(
        "POST",
        MESSAGES_TARGET,
        {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": f"websearch interception control {uuid.uuid4().hex}"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        },
    )


def register_wire_model(scenario: Scenario, wire_url: str) -> str:
    return scenario.model(
        model=f"anthropic/{ANTHROPIC_MODEL}",
        api_key="synthetic-anthropic-key",
        api_base=wire_url,
    )


@pytest.mark.covers("mgmt.websearch_interception.readback.agrees_with_observed_interception")
def test_readback_reports_interception_off_after_eviction_on_a_pod_that_served_traffic(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(respond) as wire:
        path: Final = interception_config(
            wire.url,
            tmp_path,
            {"callbacks": ["websearch_interception"]},
            {
                "callback_settings": {
                    "websearch_interception": {
                        "enabled_providers": ["anthropic"],
                        "search_tool_name": "integration-search",
                        "max_agentic_loops": 1,
                    }
                }
            },
        )
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            scenario.cleanups.callback(restore_litellm_settings_row)
            model: Final = register_wire_model(scenario, wire.url)
            response: Final = probe(candidate, model)
            assert response.status_code == 200, response.text
            assert observed_tool_names(wire) == ("litellm_web_search",)
            assert candidate.get("/get/websearch_interception_settings")["active_on_this_pod"] is True
            patched: Final = candidate.request("PATCH", "/update/websearch_interception_settings", {"enabled": False})
            assert patched.status_code == 200, patched.text
            evicted: Final = probe(candidate, model)
            assert evicted.status_code == 200, evicted.text
            names: Final = observed_tool_names(wire)
            assert names == ("web_search",), names
            rewritten: Final = "litellm_web_search" in names
            assert candidate.get("/get/websearch_interception_settings")["active_on_this_pod"] == rewritten, names


@pytest.mark.covers("mgmt.websearch_interception.reload.config_owned_params_survive_stored_row")
def test_stored_row_does_not_evict_config_declared_interception(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(respond) as wire:
        path: Final = interception_config(
            wire.url,
            tmp_path,
            {
                "callbacks": ["websearch_interception"],
                "websearch_interception_params": {
                    "enabled_providers": ["anthropic"],
                    "search_tool_name": "integration-search",
                    "max_agentic_loops": 1,
                },
            },
        )
        with owned_proxy_process(gateway, tmp_path, {"LITELLM_LOG": "DEBUG"}, config=path) as owned:
            candidate: Final = owned.gateway
            with candidate.scenario() as scenario:
                scenario.cleanups.callback(restore_litellm_settings_row)
                model: Final = register_wire_model(scenario, wire.url)
                response: Final = probe(candidate, model)
                assert response.status_code == 200, response.text
                assert observed_tool_names(wire) == ("litellm_web_search",)
                refused: Final = candidate.request(
                    "PATCH", "/update/websearch_interception_settings", {"enabled": False}
                )
                assert refused.status_code == 400, refused.text
                execute(
                    'INSERT INTO "LiteLLM_Config" (param_name, param_value) VALUES (%s, %s::jsonb) '
                    "ON CONFLICT (param_name) DO UPDATE SET param_value = "
                    "COALESCE(\"LiteLLM_Config\".param_value, '{}'::jsonb) || EXCLUDED.param_value",
                    ("litellm_settings", json.dumps({"websearch_interception_params": {"enabled": False}})),
                )

                def reload_cycles() -> int:
                    return owned.log.read_text().count("add_deployment_job started")

                baseline: Final = reload_cycles()
                eventually(reload_cycles, lambda count: count >= baseline + 2, seconds=30)
                reloaded: Final = probe(candidate, model)
                names: Final = observed_tool_names(wire)
                assert names == ("litellm_web_search",), names
                assert reloaded.status_code == 200, reloaded.text
                assert candidate.get("/get/websearch_interception_settings")["active_on_this_pod"] is True
