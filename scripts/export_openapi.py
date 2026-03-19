"""
Export the LiteLLM proxy's OpenAPI schema to a JSON file.

Usage:
    python scripts/export_openapi.py [output_path]

Defaults to ui/litellm-dashboard/openapi.json if no path is given.
Does NOT require a running server — imports the FastAPI app and calls app.openapi().
"""

import json
import sys
from pathlib import Path
from typing import Any


def deduplicate_operation_ids(schema: dict[str, Any]) -> dict[str, Any]:
    """
    FastAPI pass-through proxy routes generate duplicate operationIds when
    multiple HTTP methods share the same path pattern. This causes
    openapi-typescript to emit duplicate TS identifiers.

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
    # Determine output path
    repo_root = Path(__file__).resolve().parent.parent
    default_output = repo_root / "ui" / "litellm-dashboard" / "openapi.json"
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_output

    # Import the FastAPI app — this triggers module-level setup including
    # the assignment of app.openapi = get_openapi_schema
    from litellm.proxy.proxy_server import app

    schema = app.openapi()
    schema = deduplicate_operation_ids(schema)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2) + "\n")

    print(f"OpenAPI schema written to {output_path}")
    print(f"  paths: {len(schema.get('paths', {}))}")
    print(f"  schemas: {len(schema.get('components', {}).get('schemas', {}))}")


if __name__ == "__main__":
    main()
