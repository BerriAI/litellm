"""
Fence for the Router's public surface.

Code outside the router may use only the Router members listed in
router_surface_allowlist.json, and may import only the litellm.router_utils
modules listed there. The lists hold what is used today and may only shrink:

- a member or module used outside the router that is not listed fails the
  check (use an existing entry point instead, or discuss widening the surface);
- a listed member or module that nothing outside the router uses any more
  also fails, so the entry is removed in the same change.

"Outside" means every non-test file under litellm/ and enterprise/ except
litellm/router.py and the router's own modules. Uses from the router family
(litellm/router_utils, litellm/router_strategy) are listed separately, because
those modules become private parts of the router core in a later step.

Member uses are found the same way the boundary survey found them: attribute
access on a receiver whose name looks like a router (llm_router,
litellm_router_instance, router, ...) or on the Router class, plus
getattr/setattr/hasattr with a constant name on such a receiver. Members
include the container file endpoints Router.__init__ registers by name.

Run: python tests/code_coverage_tests/router_surface_check.py [--write]
--write regenerates the allowlist from the current tree (for shrinking it).
"""

import ast
import json
import os
import re
import sys
from typing import Dict, List, Set

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ALLOWLIST = os.path.join(os.path.dirname(__file__), "router_surface_allowlist.json")

ROUTER_FILES = {
    "litellm/router.py",
}
ROUTER_CLASS_FILES = ["litellm/router.py"]
ROUTER_CLASS_NAMES = {"Router"}

SCAN_ROOTS = ["litellm", "enterprise"]
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "__pycache__",
    "litellm-rust",
    "ui",
    "docs",
    "cookbook",
    "tests",
    "test",
}

RECEIVER = re.compile(
    r"(^|[._])(llm_router|router|litellm_router|litellm_router_instance|router_instance"
    r"|_router|proxy_router|llm_router_instance|router_obj|mock_router|Router)$"
)
# Router.__init__ registers these with setattr(self, name, ...), so they never appear as self.<name>.
CONTAINER_FILE_ENDPOINTS = "litellm/containers/endpoints.json"
DYNAMIC_FUNCS = {"getattr", "setattr", "hasattr"}


def _area(rel: str) -> str:
    if rel.startswith(("litellm/router_utils/", "litellm/router_strategy/")):
        return "router_family"
    return "outside"


def _is_test(rel: str) -> bool:
    name = os.path.basename(rel)
    return "/tests/" in rel or name.startswith("test_") or name.endswith("_test.py")


def router_members() -> Set[str]:
    members: Set[str] = set()
    for rel in ROUTER_CLASS_FILES:
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef) or cls.name not in ROUTER_CLASS_NAMES:
                continue
            for node in cls.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(node.name)
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    members.add(node.target.id)
                elif isinstance(node, ast.Assign):
                    members.update(t.id for t in node.targets if isinstance(t, ast.Name))
            for node in ast.walk(cls):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                    and isinstance(node.ctx, ast.Store)
                ):
                    members.add(node.attr)
    with open(os.path.join(ROOT, CONTAINER_FILE_ENDPOINTS), encoding="utf-8") as f:
        for endpoint in json.load(f)["endpoints"]:
            members.update((endpoint["name"], endpoint["async_name"]))
    return {m for m in members if not (m.startswith("__") and m.endswith("__"))}


def _python_files():
    for top in SCAN_ROOTS:
        base = os.path.join(ROOT, top)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if fn.endswith(".py"):
                    path = os.path.join(dirpath, fn)
                    yield os.path.relpath(path, ROOT).replace(os.sep, "/"), path


def scan() -> Dict[str, Dict[str, Set[str]]]:
    """Return {"members": {area: {member: files}}, "router_utils_imports": {module: files}}."""
    members = router_members()
    uses: Dict[str, Dict[str, Set[str]]] = {"outside": {}, "router_family": {}}
    imports: Dict[str, Set[str]] = {}
    for rel, path in _python_files():
        if rel in ROUTER_FILES or _is_test(rel):
            continue
        with open(path, encoding="utf-8") as f:
            src = f.read()
        has_router = "router" in src.lower()
        if not has_router:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        area = _area(rel)
        for name in member_uses(tree, members):
            uses[area].setdefault(name, set()).add(rel)
        if area == "outside":
            for node in ast.walk(tree):
                for mod in _router_utils_modules(node):
                    imports.setdefault(mod, set()).add(rel)
    return {"members": uses, "router_utils_imports": imports}


def member_uses(tree: ast.AST, members: Set[str]) -> Set[str]:
    """The Router members a module uses, on an instance or on the Router class itself."""
    used: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in members:
            if RECEIVER.search(ast.unparse(node.value)):
                used.add(node.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in DYNAMIC_FUNCS
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in members
            and RECEIVER.search(ast.unparse(node.args[0]))
        ):
            used.add(node.args[1].value)
    return used


def _router_utils_modules(node: ast.AST) -> List[str]:
    if isinstance(node, ast.ImportFrom) and node.module:
        if node.module == "litellm.router_utils":
            return ["litellm.router_utils." + a.name for a in node.names]
        if node.module.startswith("litellm.router_utils."):
            return [node.module]
    if isinstance(node, ast.Import):
        return [a.name for a in node.names if a.name.startswith("litellm.router_utils.")]
    return []


def build_allowlist(found) -> dict:
    return {
        "_doc": (
            "Router members and litellm.router_utils modules used outside the router. "
            "Shrink-only: see tests/code_coverage_tests/router_surface_check.py."
        ),
        "members_outside": sorted(found["members"]["outside"]),
        "members_router_family": sorted(found["members"]["router_family"]),
        "router_utils_imports_outside": sorted(found["router_utils_imports"]),
    }


def compare(allow: dict, found) -> List[str]:
    errors: List[str] = []
    sections = [
        ("members_outside", found["members"]["outside"], "Router member"),
        ("members_router_family", found["members"]["router_family"], "Router member (router family)"),
        ("router_utils_imports_outside", found["router_utils_imports"], "litellm.router_utils import"),
    ]
    for key, used, what in sections:
        listed = set(allow.get(key, []))
        for name in sorted(set(used) - listed):
            files = ", ".join(sorted(used[name]))
            errors.append(
                f"new {what} outside the router: {name!r} in {files}. The router surface may only "
                f"shrink: use an entry point instead."
            )
        for name in sorted(listed - set(used)):
            errors.append(
                f"{what} {name!r} is listed in {os.path.basename(ALLOWLIST)} [{key}] but nothing uses it "
                f"any more: remove it from the list."
            )
    return errors


def main(argv: List[str]) -> int:
    found = scan()
    if "--write" in argv:
        with open(ALLOWLIST, "w", encoding="utf-8") as f:
            json.dump(build_allowlist(found), f, indent=2)
            f.write("\n")
        print(f"wrote {ALLOWLIST}")
        return 0
    with open(ALLOWLIST, encoding="utf-8") as f:
        allow = json.load(f)
    errors = compare(allow, found)
    for e in errors:
        print(e)
    if errors:
        return 1
    print(
        f"router surface ok: {len(allow['members_outside'])} members outside, "
        f"{len(allow['members_router_family'])} from the router family, "
        f"{len(allow['router_utils_imports_outside'])} router_utils modules imported outside"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
