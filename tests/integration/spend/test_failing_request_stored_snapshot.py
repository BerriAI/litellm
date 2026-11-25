import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.process import owned_proxy
from pydantic import JsonValue

UNREACHABLE_BASE: Final = "http://127.0.0.1:9/v1"
TRANSCRIPT_TURNS: Final = 40
RETRIES: Final = 2
STORED_REQUEST_BUDGET_KB: Final = 64


def _config_storing_prompts(directory: Path) -> Path:
    config: Final = object_value(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    general_settings: Final = object_value(config["general_settings"])
    path: Final = directory / "store-prompts.yaml"
    path.write_text(
        yaml.safe_dump({**config, "general_settings": {**general_settings, "store_prompts_in_spend_logs": True}})
    )
    return path


@pytest.fixture(scope="module")
def gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with gateway_from_environment() as parent:
        directory: Final = tmp_path_factory.mktemp("failing_request_stored_snapshot")
        with owned_proxy(parent, directory, {}, config=_config_storing_prompts(directory)) as owned:
            yield owned


def _transcript(marker: str) -> list[JsonValue]:
    turns: Final[list[JsonValue]] = [
        {"role": role, "content": f"turn {turn // 2} {role}"}
        for turn, role in enumerate(("user", "assistant") * TRANSCRIPT_TURNS)
    ]
    return [*turns, {"role": "user", "content": marker}]


def _stored_request(gateway: Gateway, call_id: str) -> JsonValue:
    rows: Final = eventually(
        lambda: gateway.rows('SELECT proxy_server_request FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (call_id,)),
        lambda values: len(values) == 1,
        seconds=40,
    )
    stored: Final = rows[0]["proxy_server_request"]
    return json.loads(stored) if isinstance(stored, str) else stored


def test_failing_request_with_retries_and_fallback_stores_one_copy_of_the_request(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        primary: Final = f"stored-snapshot-{uuid.uuid4().hex}"
        fallback: Final = f"stored-snapshot-fb-{uuid.uuid4().hex}"
        for name in (primary, primary, fallback, fallback):
            scenario.model(name=name, api_base=UNREACHABLE_BASE, cooldown_time=0)
        key: Final = scenario.key(models=[primary, fallback])
        marker: Final = f"stored snapshot {uuid.uuid4().hex}"

        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": primary,
                "max_tokens": 16,
                "messages": _transcript(marker),
                "router_settings_override": {"num_retries": RETRIES, "fallbacks": [{primary: [fallback]}]},
            },
            key=key,
        )
        assert response.status_code != 200, response.text
        assert fallback in response.text, response.text

        serialized: Final = json.dumps(_stored_request(gateway, response.headers["x-litellm-call-id"]))
        stored_kb: Final = len(serialized.encode()) / 1024
        assert marker in serialized, serialized[:400]
        assert stored_kb <= STORED_REQUEST_BUDGET_KB, (
            f"one failing request stored a {stored_kb:.0f} KB request snapshot for a "
            f"{2 * TRANSCRIPT_TURNS + 1}-message transcript with {RETRIES} retries and a fallback"
        )
