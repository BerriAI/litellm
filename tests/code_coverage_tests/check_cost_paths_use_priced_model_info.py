"""
Code quality check: cost-calculation code must resolve model info through
``get_priced_model_info`` / ``_cached_get_priced_model_info_helper``.

Plain ``get_model_info`` (and the ``*_get_model_info_helper`` pair) can return a
pricing-free capability-rule match, which bills $0 instead of raising for an
unmapped name. Fails listing file:line for any such call under the cost paths.
"""

import ast
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BANNED_CALLEES = frozenset({"get_model_info", "_cached_get_model_info_helper", "_get_model_info_helper"})


def _iter_cost_files():
    litellm_root = os.path.join(REPO_ROOT, "litellm")
    for dirpath, _dirnames, filenames in os.walk(litellm_root):
        for filename in filenames:
            if filename.startswith("cost_calculat") and filename.endswith(".py"):
                yield os.path.join(dirpath, filename)
    cost_calc_dir = os.path.join(litellm_root, "litellm_core_utils", "llm_cost_calc")
    for dirpath, _dirnames, filenames in os.walk(cost_calc_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)
    for filename in ("savings.py", "budget_reservation.py"):
        yield os.path.join(litellm_root, "proxy", "spend_tracking", filename)


def _callee_name(call: ast.Call):
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def main() -> int:
    violations = []
    for path in sorted(set(_iter_cost_files())):
        with open(path) as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _callee_name(node) in BANNED_CALLEES:
                violations.append(f"{os.path.relpath(path, REPO_ROOT)}:{node.lineno}")
    if violations:
        print("Cost paths must use get_priced_model_info / _cached_get_priced_model_info_helper:")
        print("\n".join(violations))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
