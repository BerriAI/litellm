"""Client side of the scripted-provider sidecar: register scenarios over its
control API through the shared transport helpers and get back a handle whose
``api_base`` is what a /model/new deployment should register for the proxy to
reach the scripted wire."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from e2e_config import SCRIPTED_PROVIDER_CONTROL_URL, SCRIPTED_PROVIDER_PROXY_BASE
from e2e_http import URL, NoBody, unwrap, post
from e2e_http import delete as http_delete
from scripted_provider import (
    WIRE_MOUNTS,
    Scenario,
    ScenarioDeleted,
    ScenarioRegistered,
    Wire,
)


@dataclass(frozen=True, slots=True)
class ScenarioHandle:
    scenario_id: str
    wire: Wire
    proxy_base: str

    def api_base(self) -> str:
        return f"{self.proxy_base}/{self.scenario_id}/{self._mount()}"

    def _mount(self) -> str:
        return WIRE_MOUNTS[self.wire]


def register_scenario(scenario: Scenario) -> ScenarioHandle:
    """POST the scenario to the sidecar's control API and return its handle."""
    result: Final = unwrap(
        post(
            URL(f"{SCRIPTED_PROVIDER_CONTROL_URL}/_scenarios"),
            headers=NoBody(),
            json=scenario,
            response_type=ScenarioRegistered,
        )
    )
    return ScenarioHandle(
        scenario_id=result.scenario_id,
        wire=scenario.wire,
        proxy_base=SCRIPTED_PROVIDER_PROXY_BASE,
    )


def delete_scenario(handle: ScenarioHandle) -> None:
    unwrap(
        http_delete(
            URL(f"{SCRIPTED_PROVIDER_CONTROL_URL}/_scenarios/{handle.scenario_id}"),
            headers=NoBody(),
            json=NoBody(),
            response_type=ScenarioDeleted,
        )
    )


CONTROL_URL: Final = SCRIPTED_PROVIDER_CONTROL_URL
