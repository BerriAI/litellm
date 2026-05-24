import json
from typing import Any

from litellm_provisioning_mcp.commands import CommandResult
from litellm_provisioning_mcp.config import Settings


def make_settings(**overrides: Any) -> Settings:
    base = dict(
        oauth_jwks_url="https://idp/jwks",
        oauth_issuer="https://idp/",
        oauth_audience="litellm-provisioning-mcp",
        oauth_required_scope="litellm:provision",
        oauth_algorithms=("RS256",),
        resource_server_url="https://mcp.example.com",
        namespace="litellm",
        chart_path="/app/helm/litellm",
        default_image_registry="ghcr.io/berriai",
        release_prefix="litellm-e2e",
        helm_binary="helm",
        kubectl_binary="kubectl",
        command_timeout=600,
        host="0.0.0.0",
        port=8080,
        log_level="INFO",
    )
    base.update(overrides)
    return Settings(**base)


class FakeRunner:
    """Records invocations and returns canned results keyed by command shape."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []

    async def __call__(self, args, *, input_text=None, timeout=None) -> CommandResult:
        self.calls.append((args, input_text))
        binary = args[0]
        if binary.endswith("kubectl"):
            if "get" in args and "pods" in args:
                return CommandResult(0, json.dumps({"items": []}), "")
            return CommandResult(0, "applied", "")
        # helm
        if "status" in args:
            return CommandResult(0, json.dumps({"info": {"status": "deployed"}}), "")
        if "list" in args:
            return CommandResult(
                0, json.dumps([{"name": "litellm-e2e-abc", "status": "deployed"}]), ""
            )
        return CommandResult(0, "ok", "")

    def find(self, *needles: str) -> tuple[list[str], str | None]:
        for args, input_text in self.calls:
            if all(n in args for n in needles):
                return args, input_text
        raise AssertionError(f"no recorded call matching {needles}; calls={self.calls}")
