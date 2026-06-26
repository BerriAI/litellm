"""
Registry for sandbox tools configured via the proxy's top-level `sandbox_tools`.

A sandbox tool maps a name to a sandbox provider plus its credentials, so the
code interpreter interceptor can resolve a tool by name to provider/key/base.
"""

from collections.abc import Iterator

from litellm._logging import verbose_logger

_SANDBOX_TOOL_REGISTRY: dict[str, dict] = {}


def _resolve_secret_value(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    if value.startswith("os.environ/"):
        from litellm.secret_managers.main import get_secret_str

        return get_secret_str(value)
    return value


def _iter_valid_tools(tools: list[dict]) -> Iterator[tuple[str, dict]]:
    for tool in tools:
        if not isinstance(tool, dict):
            verbose_logger.warning("sandbox_tools: skipping non-dict entry %r", tool)
            continue
        name = tool.get("sandbox_tool_name")
        if not name:
            verbose_logger.warning(
                "sandbox_tools: skipping entry missing 'sandbox_tool_name': %r", tool
            )
            continue
        params = tool.get("litellm_params") or {}
        provider = params.get("sandbox_provider")
        if not provider:
            verbose_logger.warning(
                "sandbox_tools: skipping entry missing 'sandbox_provider': %r", tool
            )
            continue
        yield (
            name,
            {
                "sandbox_provider": provider,
                "api_key": _resolve_secret_value(params.get("api_key")),
                "api_base": _resolve_secret_value(params.get("api_base")),
            },
        )


def register_sandbox_tools(tools: list[dict]) -> None:
    global _SANDBOX_TOOL_REGISTRY
    _SANDBOX_TOOL_REGISTRY = dict(_iter_valid_tools(tools))


def resolve_sandbox_tool(name: str) -> dict | None:
    return _SANDBOX_TOOL_REGISTRY.get(name)


def clear_sandbox_tools() -> None:
    register_sandbox_tools([])
