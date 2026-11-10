import ast
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
PROOFS: Final = {
    "WithinModelBudgets": "litellm/proxy/auth/core/model_budgets.py",
    "BudgetsSkipped": "litellm/proxy/auth/core/model_budgets.py",
}


def _called_name(call: ast.Call) -> str | None:
    match call.func:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
        case _:
            return None


def _violations(path: Path) -> Iterator[str]:
    relative: Final = path.relative_to(ROOT).as_posix()
    tree: Final = ast.parse(path.read_text(), filename=relative)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node)
        if name in PROOFS and PROOFS[name] != relative:
            yield f"{relative}:{node.lineno}: {name} is built only by {PROOFS[name]}"


def main() -> int:
    found: Final = tuple(
        violation for path in sorted((ROOT / "litellm").rglob("*.py")) for violation in _violations(path)
    )
    if found:
        sys.stderr.write("\n".join(found) + "\n")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
