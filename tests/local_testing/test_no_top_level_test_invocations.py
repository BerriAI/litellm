import ast
from pathlib import Path

LOCAL_TESTING_DIR = Path(__file__).parent


def _top_level_test_invocations(tree):
    invocations = []
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name and name.startswith("test_"):
            invocations.append((name, node.lineno))
    return invocations


def test_no_module_level_test_invocations():
    offenders = []
    for path in sorted(LOCAL_TESTING_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for name, lineno in _top_level_test_invocations(tree):
            offenders.append(
                f"{path.relative_to(LOCAL_TESTING_DIR)}:{lineno} calls {name}()"
            )

    assert not offenders, (
        "Test functions are invoked at module scope, so they run during pytest "
        "collection (making network calls and erroring collection for every job "
        "that globs this directory). Remove these calls; pytest collects test "
        "functions automatically:\n" + "\n".join(offenders)
    )
