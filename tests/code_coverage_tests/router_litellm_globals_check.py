"""
The router and its cores never read or write ``litellm.*``.

Process settings reach the router through ``RouterDynamicConfig`` (litellm/router_config.py) and
everything else outside it (inference calls, callbacks, the global cache) through ``PythonHost``
(litellm/router_python_host.py). Those two modules are the only router code allowed to touch the
``litellm`` package namespace. This check fails on ``import litellm`` (or ``import litellm.<x>``,
which binds the same name) and on any ``litellm.<attr>`` access in the files below.

Run: python tests/code_coverage_tests/router_litellm_globals_check.py
"""

import ast
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKED = (
    "litellm/router.py",
    "litellm/router_api.py",
    "litellm/router_core.py",
    "litellm/router_deprecated.py",
    "litellm/router_python_core.py",
)


def violations(rel: str) -> list[str]:
    tree = ast.parse(open(os.path.join(ROOT, rel)).read(), rel)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(
                f"{rel}:{node.lineno}: import {alias.name}"
                for alias in node.names
                if alias.asname is None and (alias.name == "litellm" or alias.name.startswith("litellm."))
            )
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "litellm":
            found.append(f"{rel}:{node.lineno}: litellm.{node.attr}")
    return found


def main() -> int:
    found = [line for rel in CHECKED for line in violations(rel)]
    for line in found:
        print(line)
    if found:
        print(f"{len(found)} litellm.* uses in router modules; read settings from RouterDynamicConfig and use PythonHost")
        return 1
    print("router modules use no litellm.* globals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
