import os
import re
import ast
from pathlib import Path


class DataReplaceVisitor(ast.NodeVisitor):
    """AST visitor that finds calls to .replace("data:", ...) in the code."""

    def __init__(self):
        self.issues = []
        self.current_file = None

    def set_file(self, filename):
        self.current_file = filename

    def visit_Call(self, node):
        # Check for method calls like x.replace(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "replace":
            # Check if first argument is "data:"
            if (
                len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and "data:" in node.args[0].value
            ):

                self.issues.append(
                    {
                        "file": self.current_file,
                        "line": node.lineno,
                        "col": node.col_offset,
                        "text": f'Found .replace("data:", ...) at line {node.lineno}',
                    }
                )

        # Continue visiting child nodes
        self.generic_visit(node)


def check_file_with_ast(file_path):
    """Check a Python file for .replace("data:", ...) using AST parsing."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=file_path)
            visitor = DataReplaceVisitor()
            visitor.set_file(file_path)
            visitor.visit(tree)
            return visitor.issues
        except SyntaxError:
            return [
                {
                    "file": file_path,
                    "line": 0,
                    "col": 0,
                    "text": f"Syntax error in file, could not parse",
                }
            ]


def check_file_with_regex(file_path):
    """Check any file for .replace("data:", ...) using regex."""
    issues = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            matches = re.finditer(r'\.replace\(\s*[\'"]data:[\'"]', line)
            for match in matches:
                issues.append(
                    {
                        "file": file_path,
                        "line": i,
                        "col": match.start(),
                        "text": f'Found .replace("data:", ...) at line {i}',
                    }
                )
    return issues


def scan_directory(base_dir):
    """Scan a directory recursively for files containing .replace("data:", ...)."""
    all_issues = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            print("checking file: ", file)
            file_path = os.path.join(root, file)

            # Skip directories we don't want to check
            if any(
                d in file_path for d in [".git", "__pycache__", ".venv", "node_modules"]
            ):
                continue

            # For Python files, use AST for more accurate parsing
            if file.endswith(".py"):
                issues = check_file_with_ast(file_path)
            # For other files that might contain code, use regex
            elif file.endswith((".js", ".ts", ".jsx", ".tsx", ".md", ".ipynb")):
                issues = check_file_with_regex(file_path)
            else:
                continue

            all_issues.extend(issues)

    return all_issues


def main():
    # Start from the project root directory

    base_dir = "./litellm"

    # Local testing
    # base_dir = "../../litellm"

    print(f"Scanning for .replace('data:', ...) usage in {base_dir}")
    issues = scan_directory(base_dir)

    if issues:
        print(f"\n⚠️ Found {len(issues)} instances of .replace('data:', ...):")
        for issue in issues:
            print(f"{issue['file']}:{issue['line']} - {issue['text']}")

        # Fail the test if issues are found
        raise Exception(
            f"Found {len(issues)} instances of .replace('data:', ...) which may be unsafe. Use litellm.CustomStreamWrapper._strip_sse_data_from_chunk instead."
        )
    else:
        print("✅ No instances of .replace('data:', ...) found.")


if __name__ == "__main__":
    main()
