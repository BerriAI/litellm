from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal
from urllib.parse import urlsplit

from capture_policy import canonical_scenario_id
from e2e_http import NetworkError, forward
from fixture_mode import current_test_key
from provider_edge import ProviderEdge
from provider_edge_control import ControlReply, ControlRequest


@dataclass(frozen=True, slots=True)
class RemoteEdge:
    control_url: str
    data_url: str

    def __post_init__(self) -> None:
        control: Final = urlsplit(self.control_url)
        data: Final = urlsplit(self.data_url)
        if (
            control.scheme != "http"
            or control.hostname != "127.0.0.1"
            or control.path not in ("", "/")
            or control.username
            or control.query
            or control.fragment
        ):
            raise ValueError("edge management URL must be loopback HTTP")
        if (
            data.scheme != "http"
            or not data.hostname
            or data.port is None
            or data.path not in ("", "/")
            or data.username
            or data.query
            or data.fragment
        ):
            raise ValueError("edge data URL must name an HTTP host and port")

    @property
    def edge(self) -> ProviderEdge:
        parsed: Final = urlsplit(self.data_url)
        assert parsed.hostname is not None and parsed.port is not None
        return ProviderEdge(parsed.port, parsed.hostname)

    def command(self, request: ControlRequest) -> ControlReply:
        response: Final = forward(
            "POST",
            self.control_url.rstrip("/") + "/control",
            headers={"content-type": "application/json"},
            body=request.model_dump_json().encode(),
            timeout=10,
        )
        if isinstance(response, NetworkError):
            raise RuntimeError("trusted edge control unavailable")
        reply: Final = ControlReply.model_validate_json(response.body)
        if response.status_code != 200 or not reply.ok:
            raise RuntimeError(reply.error or "trusted edge control rejected operation")
        return reply

    def begin(self, node: str) -> None:
        self.command(ControlRequest(action="begin", node=canonical_scenario_id(node)))

    def phase(self, node: str, phase: Literal["setup", "call", "teardown"], passed: bool) -> None:
        self.command(ControlRequest(action="phase", node=canonical_scenario_id(node), phase=phase, passed=passed))

    def observe(self, marker: str) -> str:
        reply: Final = self.command(
            ControlRequest(action="observe", node=canonical_scenario_id(current_test_key()), marker=marker)
        )
        if reply.observation_id is None:
            raise RuntimeError("trusted edge did not register observation")
        return reply.observation_id

    def count(self, observation_id: str) -> int:
        reply: Final = self.command(
            ControlRequest(
                action="count", node=canonical_scenario_id(current_test_key()), observation_id=observation_id
            )
        )
        if reply.count is None:
            raise RuntimeError("trusted edge did not return observed count")
        return reply.count
