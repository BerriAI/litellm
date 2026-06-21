"""
Registry for sandbox tools configured via the proxy's top-level `sandbox_tools`.

A sandbox tool maps a name to a sandbox provider plus its credentials, so the
code interpreter interceptor can resolve a tool by name to provider/key/base.
"""

_SANDBOX_TOOL_REGISTRY: dict[str, dict] = {}


def _resolve_secret_value(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    if value.startswith("os.environ/"):
        from litellm.secret_managers.main import get_secret_str

        return get_secret_str(value)
    return value


def register_sandbox_tools(tools: list[dict]) -> None:
    global _SANDBOX_TOOL_REGISTRY
    _SANDBOX_TOOL_REGISTRY = {
        tool["sandbox_tool_name"]: {
            "sandbox_provider": (tool.get("litellm_params") or {}).get(
                "sandbox_provider"
            ),
            "api_key": _resolve_secret_value(
                (tool.get("litellm_params") or {}).get("api_key")
            ),
            "api_base": _resolve_secret_value(
                (tool.get("litellm_params") or {}).get("api_base")
            ),
        }
        for tool in tools
    }


def resolve_sandbox_tool(name: str) -> dict | None:
    return _SANDBOX_TOOL_REGISTRY.get(name)


def clear_sandbox_tools() -> None:
    register_sandbox_tools([])
