import ast
import os
import re

# Symbols that only exist in stdlib `typing` on Python 3.11+.
# pyproject sets requires-python >=3.10, so importing these from `typing`
# on a path that runs under 3.10 breaks `import litellm`.
POST_310_TYPING = {
    "NotRequired", "Required", "assert_never", "assert_type", "Self",
    "LiteralString", "Never", "TypeVarTuple", "Unpack",
    "dataclass_transform", "override", "TypeAliasType", "ReadOnly", "TypeIs",
}

# Cheap text prefilter so we only AST-parse the ~400 files that could offend,
# rather than all ~2,250. A real offender must contain both patterns as text,
# so this cannot hide one.
_TYPING_IMPORT_RE = re.compile(r"from\s+typing\s+import")
_SYMBOL_RE = re.compile("|".join(sorted(POST_310_TYPING)))

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
LITELLM_ROOT = os.path.join(REPO_ROOT, "litellm")


def _version_branch(test):
    """For `if sys.version_info <op> (3, N)`, say which branch is the new-Python one."""
    if "version_info" not in ast.dump(test):
        return None
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return "both"  # version check we cannot read; stay conservative
    op = test.ops[0]
    if isinstance(op, (ast.GtE, ast.Gt)):
        return "body"
    if isinstance(op, (ast.Lt, ast.LtE)):
        return "orelse"
    return "both"


def _has_import_fallback(node):
    for handler in node.handlers:
        if handler.type is None:
            return True
        names = [n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name)]
        if "ImportError" in names or "Exception" in names:
            return True
    return False


def _scan(node, guarded, path, bad):
    """Walk the tree tracking whether we are on a 3.10-safe code path."""
    if isinstance(node, ast.ImportFrom) and node.module == "typing" and not guarded:
        for alias in node.names:
            if alias.name in POST_310_TYPING:
                bad.append(path + ":" + str(node.lineno) + " imports " + alias.name + " from typing")
        return

    if isinstance(node, ast.If):
        dumped = ast.dump(node.test)
        if "TYPE_CHECKING" in dumped:
            # never executed at runtime, so it cannot break the import
            body_guarded, else_guarded = True, guarded
        else:
            branch = _version_branch(node.test)
            if branch == "body":
                body_guarded, else_guarded = True, guarded
            elif branch == "orelse":
                body_guarded, else_guarded = guarded, True
            elif branch == "both":
                body_guarded, else_guarded = True, True
            else:
                body_guarded, else_guarded = guarded, guarded
        for child in node.body:
            _scan(child, body_guarded, path, bad)
        for child in node.orelse:
            _scan(child, else_guarded, path, bad)
        return

    if isinstance(node, ast.Try):
        # `try: from typing import X / except ImportError: from typing_extensions import X`
        # is the intended fallback. Only the try body gets the pass.
        fallback = _has_import_fallback(node)
        for child in node.body:
            _scan(child, fallback or guarded, path, bad)
        for handler in node.handlers:
            for child in handler.body:
                _scan(child, guarded, path, bad)
        for child in node.orelse + node.finalbody:
            _scan(child, guarded, path, bad)
        return

    for child in ast.iter_child_nodes(node):
        _scan(child, guarded, path, bad)


def _find_bad_typing_imports():
    bad = []
    for dirpath, _, filenames in os.walk(LITELLM_ROOT):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            if not (_TYPING_IMPORT_RE.search(source) and _SYMBOL_RE.search(source)):
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            _scan(tree, False, os.path.relpath(path, REPO_ROOT).replace(os.sep, "/"), bad)
    return bad


def test_no_unguarded_post_310_typing_imports():
    bad = _find_bad_typing_imports()
    assert not bad, (
        "These import 3.11+ symbols from stdlib `typing` on a code path that "
        "runs under Python 3.10 (requires-python >=3.10). Import them from "
        "typing_extensions instead:\n" + "\n".join(bad)
    )
