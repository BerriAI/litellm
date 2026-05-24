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
        provisioning_service_account="litellm-provisioning-mcp",
        allowed_service_accounts=(),
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
        # Toggles driving the read-only `kubectl get` / `helm list` probes.
        self.master_key_exists = False
        self.owns_release = True
        self.helm_releases = ["litellm-e2e-other"]

    async def __call__(self, args, *, input_text=None, timeout=None) -> CommandResult:
        self.calls.append((args, input_text))
        binary = args[0]
        if binary.endswith("kubectl"):
            if "get" in args:
                if "pods" in args:
                    return CommandResult(0, json.dumps({"items": []}), "")
                if "--selector" in args:  # count_by_label (ownership check)
                    items = (
                        [{"metadata": {"name": "owned"}}] if self.owns_release else []
                    )
                    return CommandResult(0, json.dumps({"items": items}), "")
                # resource_exists: `get <kind> <name> --ignore-not-found -o name`
                return CommandResult(
                    0, "secret/x" if self.master_key_exists else "", ""
                )
            return CommandResult(0, "applied", "")
        # helm
        if "status" in args:
            return CommandResult(0, json.dumps({"info": {"status": "deployed"}}), "")
        if "list" in args:
            payload = [{"name": n, "status": "deployed"} for n in self.helm_releases]
            return CommandResult(0, json.dumps(payload), "")
        return CommandResult(0, "ok", "")

    def find(self, *needles: str) -> tuple[list[str], str | None]:
        for args, input_text in self.calls:
            if all(n in args for n in needles):
                return args, input_text
        raise AssertionError(f"no recorded call matching {needles}; calls={self.calls}")
