"""Client for registering scenarios with the integration upstream."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

import httpx
from integration._support.scripted_wires import (
    WIRE_MOUNTS,
    Scenario,
    ScenarioDeleted,
    ScenarioRegistered,
    Wire,
)

CONTROL_URL: Final = os.environ.get("INTEGRATION_UPSTREAM_URL", "http://127.0.0.1:8190").rstrip("/")


@dataclass(frozen=True, slots=True)
class ScenarioHandle:
    scenario_id: str
    wire: Wire
    control_url: str

    def api_base(self) -> str:
        return f"{self.control_url}/{self.scenario_id}/{self._mount()}"

    def _mount(self) -> str:
        return WIRE_MOUNTS[self.wire]


def register_scenario(scenario: Scenario) -> ScenarioHandle:
    response: Final = httpx.post(
        f"{CONTROL_URL}/__scenarios",
        json=scenario.model_dump(mode="json"),
        trust_env=False,
        timeout=15,
    )
    response.raise_for_status()
    result: Final = ScenarioRegistered.model_validate_json(response.content)
    return ScenarioHandle(
        scenario_id=result.scenario_id,
        wire=scenario.wire,
        control_url=CONTROL_URL,
    )


def delete_scenario(handle: ScenarioHandle) -> None:
    response: Final = httpx.delete(
        f"{CONTROL_URL}/__scenarios/{handle.scenario_id}",
        trust_env=False,
        timeout=15,
    )
    response.raise_for_status()
    ScenarioDeleted.model_validate_json(response.content)
