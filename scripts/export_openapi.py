"""
Generate TypeScript types for LiteLLM management endpoints.

Usage:
    python scripts/export_openapi.py

Does NOT require a running server — imports the FastAPI app and calls app.openapi().
Filters to management endpoints only (no passthrough/LLM API routes).
Produces ui/litellm-dashboard/src/types/api.generated.ts (gitignored).
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Tags for routes the UI dashboard actually uses.
# Routes without these tags (passthrough, OpenAI-compatible API, etc.) are excluded.
INCLUDED_TAGS = {
    # Management endpoints (UI dashboard)
    "access group management",
    "budget management",
    "credential management",
    "email management",
    "key management",
    "model management",
    "organization management",
    "policy management",
    "project management",
    "tag management",
    "team management",
    "tool management",
    "vector store management",
    "Settings",
    "SSO Settings",
    "UI Settings",
    "UI Theme Settings",
    "Router Settings",
    "[beta] MCP",
    "[beta] Agents",
    "[beta] A2A Agents",
    "health",
    # LLM API endpoints
    "chat/completions",
    "completions",
    "responses",
}


def filter_management_routes(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Remove all routes that aren't management endpoints.
    Also prunes unused schemas from components.
    """
    filtered_paths: dict[str, Any] = {}

    for path, path_obj in schema.get("paths", {}).items():
        filtered_methods: dict[str, Any] = {}
        for method, method_obj in path_obj.items():
            if not isinstance(method_obj, dict):
                filtered_methods[method] = method_obj
                continue
            tags = set(method_obj.get("tags", []))
            if tags & INCLUDED_TAGS:
                filtered_methods[method] = method_obj
        if any(isinstance(v, dict) for v in filtered_methods.values()):
            filtered_paths[path] = filtered_methods

    schema["paths"] = filtered_paths

    # Prune schemas not referenced by the remaining paths
    paths_json = json.dumps(filtered_paths)
    referenced = set(re.findall(r'#/components/schemas/([\w.\-]+)', paths_json))

    # Schemas can reference other schemas, so resolve transitively
    all_schemas = schema.get("components", {}).get("schemas", {})
    resolved: set[str] = set()
    to_resolve = list(referenced)
    while to_resolve:
        name = to_resolve.pop()
        if name in resolved:
            continue
        resolved.add(name)
        if name in all_schemas:
            nested_json = json.dumps(all_schemas[name])
            for ref in re.findall(r'#/components/schemas/([\w.\-]+)', nested_json):
                if ref not in resolved:
                    to_resolve.append(ref)

    schema["components"]["schemas"] = {
        k: v for k, v in all_schemas.items() if k in resolved
    }

    return schema


def deduplicate_operation_ids(schema: dict[str, Any]) -> dict[str, Any]:
    """
    FastAPI pass-through proxy routes generate duplicate operationIds when
    multiple HTTP methods share the same path pattern.
    Fix: append _2, _3, etc. to duplicates.
    """
    seen: dict[str, int] = {}
    for path_obj in schema.get("paths", {}).values():
        for method_obj in path_obj.values():
            if not isinstance(method_obj, dict):
                continue
            op_id = method_obj.get("operationId")
            if op_id is None:
                continue
            count = seen.get(op_id, 0) + 1
            seen[op_id] = count
            if count > 1:
                method_obj["operationId"] = f"{op_id}_{count}"
    return schema


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ui_dir = repo_root / "ui" / "litellm-dashboard"
    output_ts = ui_dir / "src" / "types" / "api.generated.ts"

    # Step 1: Export OpenAPI schema from FastAPI app
    print("Exporting OpenAPI schema from FastAPI app...")
    from litellm.proxy.proxy_server import app

    schema = app.openapi()

    full_paths = len(schema.get("paths", {}))
    full_schemas = len(schema.get("components", {}).get("schemas", {}))

    # Step 2: Filter to management endpoints only
    schema = filter_management_routes(schema)
    schema = deduplicate_operation_ids(schema)

    mgmt_paths = len(schema.get("paths", {}))
    mgmt_schemas = len(schema.get("components", {}).get("schemas", {}))

    print(f"  Full schema: {full_paths} paths, {full_schemas} schemas")
    print(f"  After filtering: {mgmt_paths} paths, {mgmt_schemas} schemas")

    # Step 3: Run openapi-typescript to generate TS types
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(schema, tmp, indent=2)
        tmp_path = tmp.name

    print(f"Generating TypeScript types -> {output_ts.relative_to(repo_root)}")
    output_ts.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["npx", "openapi-typescript", tmp_path, "-o", str(output_ts)],
        cwd=str(ui_dir),
        capture_output=True,
        text=True,
    )

    Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"openapi-typescript failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Report final size
    lines = output_ts.read_text().count("\n")
    size_kb = output_ts.stat().st_size / 1024
    print(f"  Output: {lines} lines, {size_kb:.0f} KB")
    print("Done.")


if __name__ == "__main__":
    main()
