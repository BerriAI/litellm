"""
Test that all guardrail hooks with async def apply_guardrail use @log_guardrail_information decorator.

This ensures consistent logging and observability across all guardrail implementations.
"""

import ast
from pathlib import Path
from typing import List, Tuple


def find_apply_guardrail_methods(file_path: Path) -> List[Tuple[str, int, bool]]:
    """
    Find all apply_guardrail methods and check if they have the decorator.

    Returns:
        List of tuples: (class_name, line_number, has_decorator)
    """
    with open(file_path, "r") as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    results = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name

            # Check if this class has apply_guardrail method
            for item in node.body:
                if (
                    isinstance(item, ast.AsyncFunctionDef)
                    and item.name == "apply_guardrail"
                ):
                    # Check if it has the log_guardrail_information decorator
                    has_decorator = False
                    for decorator in item.decorator_list:
                        if (
                            isinstance(decorator, ast.Name)
                            and decorator.id == "log_guardrail_information"
                        ):
                            has_decorator = True
                            break

                    results.append((class_name, item.lineno, has_decorator))

    return results


def test_guardrail_apply_decorator():
    """Test that all guardrail hooks with apply_guardrail have the decorator."""
    # Path to the guardrail hooks directory
    guardrail_hooks_dir = (
        Path(__file__).parent.parent.parent
        / "litellm"
        / "proxy"
        / "guardrails"
        / "guardrail_hooks"
    )

    # Find all Python files in the guardrail hooks directory
    python_files = list(guardrail_hooks_dir.rglob("*.py"))

    # Track violations
    violations = []

    for python_file in python_files:
        # Skip __init__.py files and test files
        if python_file.name == "__init__.py" or python_file.name.startswith("test_"):
            continue

        # Skip base files and primitives
        if python_file.name in ["base.py", "primitives.py", "patterns.py"]:
            continue

        # Skip bedrock_guardrails.py - it implements logging differently via
        # add_standard_logging_guardrail_information_to_request_data calls
        # in make_bedrock_api_request method instead of using the decorator
        if python_file.name == "bedrock_guardrails.py":
            continue

        # Skip content_filter.py - it implements its own detailed logging via
        # _log_guardrail_information with detections, masked_entity_count, etc.
        # Using the decorator would cause duplicate entries.
        if python_file.name == "content_filter.py":
            continue

        results = find_apply_guardrail_methods(python_file)

        for class_name, line_num, has_decorator in results:
            if not has_decorator:
                relative_path = python_file.relative_to(
                    Path(__file__).parent.parent.parent
                )
                violations.append((relative_path, class_name, line_num))

    # Assert no violations found
    if violations:
        print(
            f"\nFound {len(violations)} guardrail hook(s) without @log_guardrail_information decorator:"
        )
        print(
            "\nAll guardrail hooks must use @log_guardrail_information decorator on their apply_guardrail method."
        )
        print(
            "This ensures consistent logging and observability across all guardrails.\n"
        )

        for file_path, class_name, line_num in violations:
            print(f"  - {file_path}:{line_num} ({class_name}.apply_guardrail)")

        print("\nTo fix, add the decorator:")
        print(
            "  from litellm.integrations.custom_guardrail import log_guardrail_information"
        )
        print("  ")
        print("  @log_guardrail_information")
        print("  async def apply_guardrail(self, ...):")
        print("      ...")

        raise AssertionError(
            f"Found {len(violations)} guardrail hook(s) without @log_guardrail_information decorator"
        )


if __name__ == "__main__":
    test_guardrail_apply_decorator()
    print("✓ All guardrail hooks have @log_guardrail_information decorator")
