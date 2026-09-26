import os
import uuid
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.conformance import authenticated_endpoint, reference_server, run_scenario
from integration._support.mcp import official_client_outcomes, register_mcp


@pytest.mark.parametrize("name", ("server-initialize", "tools-list", "tools-call-image"))
def test_official_scenario_through_gateway(gateway: Gateway, tmp_path: Path, unused_tcp_port: int, name: str) -> None:
    root: Final = Path(os.environ["MCP_CONFORMANCE_ROOT"])
    output: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(tmp_path))) / f"conformance-{uuid.uuid4().hex}"
    with reference_server(root, output, unused_tcp_port) as reference, gateway.scenario() as scenario:
        alias: Final = "official" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, reference, alias, mcp_info={"protocol_version": "2025-11-25"})
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        direct: Final = run_scenario(root, reference.url, name, output / "direct")
        endpoint: Final = str(gateway.client.base_url).rstrip("/") + f"/{alias}/mcp"
        with authenticated_endpoint(endpoint, key) as authenticated:
            proxied: Final = run_scenario(root, authenticated, name, output / "gateway")
        assert tuple(check.status for check in direct if check.id == name) == ("SUCCESS",)
        assert tuple(check.status for check in proxied if check.id == name) == ("SUCCESS",)
        listed, called = official_client_outcomes(gateway, key, f"/{alias}/mcp", "test_simple_text", {})
        assert f"{alias}-test_simple_text" in listed.tools, listed
        assert called.ok and called.text == "This is a simple text response for testing.", called
