import ast
import os


class CopyDeepcopyKwargsDetector(ast.NodeVisitor):
    def __init__(self):
        self.violations = []

    def visit_Call(self, node):
        # Check if this is a copy.deepcopy call
        if self._is_copy_deepcopy_call(node):
            # Check if any argument contains 'kwargs' in its name
            for arg in node.args:
                if self._is_kwargs_related(arg):
                    # Get line number and argument name for reporting
                    arg_name = self._get_arg_name(arg)
                    self.violations.append(
                        {
                            "line": node.lineno,
                            "arg_name": arg_name,
                            "full_call": (
                                ast.unparse(node)
                                if hasattr(ast, "unparse")
                                else str(node)
                            ),
                        }
                    )

        self.generic_visit(node)

    def _is_copy_deepcopy_call(self, node):
        """Check if this is a copy.deepcopy() call"""
        if isinstance(node.func, ast.Attribute):
            # Case: copy.deepcopy()
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "copy"
                and node.func.attr == "deepcopy"
            ):
                return True
        elif isinstance(node.func, ast.Name):
            # Case: deepcopy() (if imported as 'from copy import deepcopy')
            if node.func.id == "deepcopy":
                return True
        return False

    def _is_kwargs_related(self, arg):
        """Check if the argument is kwargs-related"""
        if isinstance(arg, ast.Name):
            # Direct variable names containing 'kwargs'
            return "kwargs" in arg.id.lower()
        elif isinstance(arg, ast.Subscript):
            # Handle cases like kwargs['key']
            if isinstance(arg.value, ast.Name):
                return "kwargs" in arg.value.id.lower()
        elif isinstance(arg, ast.Attribute):
            # Handle cases like self.kwargs
            return "kwargs" in arg.attr.lower()
        return False

    def _get_arg_name(self, arg):
        """Get a readable name for the argument"""
        if isinstance(arg, ast.Name):
            return arg.id
        elif isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
            return f"{arg.value.id}[...]"
        elif isinstance(arg, ast.Attribute):
            return f"...{arg.attr}"
        else:
            return "unknown_kwargs_variable"


def find_copy_deepcopy_kwargs_in_file(file_path):
    """Find copy.deepcopy usage with kwargs in a single file"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read(), filename=file_path)
        detector = CopyDeepcopyKwargsDetector()
        detector.visit(tree)
        return detector.violations
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def find_copy_deepcopy_kwargs_in_directory(directory):
    """Find copy.deepcopy usage with kwargs in all Python files in directory"""
    violations = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"Checking file: {file_path}")
                file_violations = find_copy_deepcopy_kwargs_in_file(file_path)
                if file_violations:
                    violations[file_path] = file_violations

    return violations


if __name__ == "__main__":
    # Check for copy.deepcopy(kwargs) usage in the litellm directory
    directory_path = "./litellm"
    violations = find_copy_deepcopy_kwargs_in_directory(directory_path)

    print("\n" + "=" * 80)
    print("COPY.DEEPCOPY KWARGS VIOLATIONS FOUND:")
    print("=" * 80)

    if violations:
        total_violations = 0
        for file_path, file_violations in violations.items():
            print(f"\n📁 File: {file_path}")
            for violation in file_violations:
                total_violations += 1
                print(
                    f"  ❌ Line {violation['line']}: copy.deepcopy({violation['arg_name']})"
                )
                print(f"     Full call: {violation['full_call']}")

        print(f"\n{'='*80}")
        print(f"🚨 TOTAL VIOLATIONS: {total_violations}")
        print("🚨 USE safe_deep_copy() INSTEAD OF copy.deepcopy() FOR KWARGS!")
        print("🚨 Available imports:")
        print("   - from litellm.proxy.utils import safe_deep_copy")
        print("   - from litellm.litellm_core_utils.core_helpers import safe_deep_copy")
        print("=" * 80)

        # Get first violation for the exception message
        first_file = list(violations.keys())[0]
        first_violation = violations[first_file][0]

        raise Exception(
            f"🚨 Found {total_violations} copy.deepcopy(kwargs) violations! "
            f"First violation: {first_file}:{first_violation['line']} - "
            f"copy.deepcopy({first_violation['arg_name']}). "
            f"Use safe_deep_copy() instead to handle non-serializable objects like OTEL spans."
        )
    else:
        print("✅ No copy.deepcopy(kwargs) violations found!")
        print("✅ All kwargs copying appears to use safe_deep_copy() correctly.")
