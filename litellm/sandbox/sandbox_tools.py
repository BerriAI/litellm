"""
Registry for sandbox tools configured via the proxy's top-level `sandbox_tools`.

A sandbox tool maps a name to a sandbox provider plus its credentials, so the
code interpreter interceptor can resolve a tool by name to provider/key/base.
"""

from typing import Optional

_SANDBOX_TOOL_REGISTRY: dict[str, dict] = {}


def _resolve_secret_value(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    if value.startswith("os.environ/"):
        from litellm.secret_managers.main import get_secret_str

        return get_secret_str(value)
    return value


def register_sandbox_tools(tools: list[dict]) -> None:
    _SANDBOX_TOOL_REGISTRY.clear()
    for tool in tools:
        name = tool["sandbox_tool_name"]
        litellm_params = tool.get("litellm_params", {}) or {}
        _SANDBOX_TOOL_REGISTRY[name] = {
            "sandbox_provider": litellm_params.get("sandbox_provider"),
            "api_key": _resolve_secret_value(litellm_params.get("api_key")),
            "api_base": _resolve_secret_value(litellm_params.get("api_base")),
        }


def resolve_sandbox_tool(name: str) -> Optional[dict]:
    return _SANDBOX_TOOL_REGISTRY.get(name)


def clear_sandbox_tools() -> None:
    _SANDBOX_TOOL_REGISTRY.clear()
