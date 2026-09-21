import ast
import sys
from pathlib import Path
from typing import Final

PACKAGE: Final = Path("litellm/proxy/_experimental/mcp_server")
LEGACY_ADAPTERS: Final = frozenset({"server.py", "legacy_callbacks.py", "mcp_context.py", "mcp_debug.py"})
CONFINED_NAMES: Final = frozenset(
    {
        "auth_context_var",
        "active_mcp_session_var",
        "active_mcp_request_ctx_var",
        "get_active_auth_context",
        "get_active_mcp_session",
        "get_active_mcp_request_ctx",
        "get_or_extract_auth_context",
        "_session_obj_auth_storage",
        "WeakKeyDictionary",
        "_mcp_active_toolset_id",
        "_mcp_gateway_initialize_instructions",
        "_mcp_gateway_server_name",
        "_mcp_proxy_mode",
    }
)


def is_confined(name: str) -> bool:
    return name in CONFINED_NAMES or name.startswith("_stateful_session_")


def violations(path: Path, source: str) -> tuple[str, ...]:
    if path.name in LEGACY_ADAPTERS:
        return ()
    tree: Final = ast.parse(source, filename=str(path))
    return tuple(
        f"{path}:{node.lineno}: MCP request/session state belongs in a legacy adapter"
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.ImportFrom)
            and (
                (node.module or "").endswith(".mcp_context")
                or any(is_confined(alias.name) for alias in node.names)
                or (path.name in {"operations.py", "contracts.py"} and (node.module or "").endswith(".server"))
            )
            or isinstance(node, ast.Name)
            and is_confined(node.id)
            or isinstance(node, ast.Attribute)
            and is_confined(node.attr)
        )
    )


def main() -> int:
    findings: Final = tuple(
        finding for path in sorted(PACKAGE.rglob("*.py")) for finding in violations(path, path.read_text())
    )
    if findings:
        print("\n".join(findings), file=sys.stderr)
        return 1
    print("MCP operation boundary: passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
