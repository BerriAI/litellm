"""Regenerate the checked-in management MCP tool inventory.

Run ``python -m litellm.proxy._experimental.mcp_server.management.inventory --write``
after any change that alters the proxy's OpenAPI spec or the catalog's
inclusion rules. The unit test in
``tests/test_litellm/proxy/_experimental/mcp_server/management/test_catalog.py``
fails when this file drifts from what ``app.openapi()`` produces.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Final, cast

from litellm.proxy._experimental.mcp_server.management.catalog import (
    ManagementCatalog,
    build_catalog,
)

INVENTORY_PATH: Final = Path(__file__).with_name("inventory.json")


def catalog_inventory(catalog: ManagementCatalog) -> dict[str, dict[str, str]]:
    tool_entries: Final = {  # mutable-ok: JSON inventory tool entries
        f"{tool.method} {tool.path_template}": {"tool": name}  # mutable-ok: inventory entry body
        for name, tool in catalog.tools.items()
    }
    excluded_entries: Final = {  # mutable-ok: JSON inventory excluded entries
        key: {"excluded": reason}  # mutable-ok: inventory entry body
        for key, reason in catalog.exclusions.items()
    }
    return dict(sorted((tool_entries | excluded_entries).items()))  # mutable-ok: sorted JSON inventory mapping


def render_inventory(catalog: ManagementCatalog) -> str:
    return json.dumps(catalog_inventory(catalog), indent=2, sort_keys=True) + "\n"


def _catalog_from_app() -> ManagementCatalog:
    from litellm.proxy.proxy_server import app

    return build_catalog(app.openapi())


def main(argv: list[str] | None = None) -> int:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite inventory.json from the live app spec")
    args: Final = parser.parse_args(argv)
    if not cast(bool, args.write):  # cast-ok: argparse Namespace attribute is typed Any
        parser.print_help()
        return 2
    INVENTORY_PATH.write_text(render_inventory(_catalog_from_app()))
    sys.stdout.write(f"wrote {INVENTORY_PATH}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
