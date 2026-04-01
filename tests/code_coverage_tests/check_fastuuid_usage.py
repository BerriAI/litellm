import ast
import os
from typing import List, Dict, Any


ALLOWED_FILE = os.path.normpath("litellm/_uuid.py")


def _to_module_path(relative_path: str) -> str:
    module = os.path.splitext(relative_path)[0].replace(os.sep, ".")
    if module.endswith(".__init__"):
        return module[: -len(".__init__")]
    return module


def _find_fastuuid_imports_in_file(
    file_path: str, base_dir: str
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=file_path)
    except Exception:
        return results

    relative = os.path.normpath(os.path.relpath(file_path, base_dir))
    if relative == ALLOWED_FILE:
        return results

    module = _to_module_path(relative)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "fastuuid":
                    results.append(
                        {
                            "file": relative,
                            "line": getattr(node, "lineno", 0),
                            "import": f"import {alias.name}",
                            "module": module,
                        }
                    )
        elif isinstance(node, ast.ImportFrom) and node.module == "fastuuid":
            names = ", ".join([a.name for a in node.names])
            results.append(
                {
                    "file": relative,
                    "line": getattr(node, "lineno", 0),
                    "import": f"from fastuuid import {names}",
                    "module": module,
                }
            )

    return results


def scan_directory_for_fastuuid(base_dir: str) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    scan_root = os.path.join(base_dir, "litellm")
    for root, _, files in os.walk(scan_root):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)
                violations.extend(_find_fastuuid_imports_in_file(file_path, base_dir))
    return violations


def main() -> None:
    base_dir = "."  # tests run from repo root in CI
    violations = scan_directory_for_fastuuid(base_dir)
    if violations:
        print(
            "\n🚨 fastuuid must only be imported inside litellm/_uuid.py. Found violations:"
        )
        for v in violations:
            print(f"* {v['module']} ({v['file']}:{v['line']}) -> {v['import']}")
        print("\n")
        raise Exception(
            "Found fastuuid imports outside litellm/_uuid.py. Use litellm._uuid.uuid or litellm._uuid.uuid4 instead."
        )
    else:
        print("✅ No invalid fastuuid imports found.")


if __name__ == "__main__":
    main()
