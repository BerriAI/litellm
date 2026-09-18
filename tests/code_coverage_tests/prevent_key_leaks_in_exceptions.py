import ast
import os
import re


def is_venv_directory(path):
    """
    Check if the path contains virtual environment directories.
    Common virtual environment directory names: venv, env, .env, myenv, .venv
    """
    venv_indicators = [
        "venv",
        "env",
        ".env",
        "myenv",
        ".venv",
        "virtualenv",
        "site-packages",
    ]

    path_parts = path.lower().split(os.sep)
    return any(indicator in path_parts for indicator in venv_indicators)


class ArgsStringVisitor(ast.NodeVisitor):
    """
    AST visitor that finds all instances of '{args}' string usage.
    """

    def __init__(self):
        self.args_locations = []
        self.current_file = None

    def set_file(self, filename):
        self.current_file = filename
        self.args_locations = []

    def visit_Str(self, node):
        """Check string literals for {args}"""
        if "{args}" in node.s:
            self.args_locations.append(
                {
                    "line": node.lineno,
                    "col": node.col_offset,
                    "text": node.s,
                    "file": self.current_file,
                }
            )

    def visit_JoinedStr(self, node):
        """Check f-strings for {args}"""
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                # Check if the formatted value uses 'args'
                if isinstance(value.value, ast.Name) and value.value.id == "args":
                    self.args_locations.append(
                        {
                            "line": node.lineno,
                            "col": node.col_offset,
                            "text": "f-string with {args}",
                            "file": self.current_file,
                        }
                    )


def check_file_for_args_string(file_path):
    """
    Analyzes a Python file for any usage of '{args}'.

    Args:
        file_path (str): Path to the Python file to analyze

    Returns:
        list: List of dictionaries containing information about {args} usage
    """
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            content = file.read()
            tree = ast.parse(content)

            # First check using AST for more accurate detection in strings
            visitor = ArgsStringVisitor()
            visitor.set_file(file_path)
            visitor.visit(tree)
            ast_locations = visitor.args_locations

            # Also check using regex for any instances we might have missed
            # (like in comments or docstrings)
            line_number = 1
            additional_locations = []

            for line in content.split("\n"):
                if "{args}" in line:
                    # Only add if it's not already caught by the AST visitor
                    if not any(loc["line"] == line_number for loc in ast_locations):
                        additional_locations.append(
                            {
                                "line": line_number,
                                "col": line.index("{args}"),
                                "text": line.strip(),
                                "file": file_path,
                            }
                        )
                line_number += 1

            return ast_locations + additional_locations

        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return []


def check_directory_for_args_string(directory_path):
    """
    Recursively checks all Python files in a directory for '{args}' usage,
    excluding virtual environment directories.

    Args:
        directory_path (str): Path to the directory to check
    """
    all_violations = []

    for root, dirs, files in os.walk(directory_path):
        # Skip virtual environment directories
        if is_venv_directory(root):
            continue

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                violations = check_file_for_args_string(file_path)
                all_violations.extend(violations)

    return all_violations


def main():
    # Update this path to point to your codebase root directory
    # codebase_path = "../../litellm"  # Adjust as needed
    codebase_path = "./litellm"

    violations = check_directory_for_args_string(codebase_path)

    if violations:
        print("Found '{args}' usage in the following locations:")
        for violation in violations:
            print(f"- {violation['file']}:{violation['line']} - {violation['text']}")
        raise Exception(
            f"Found {len(violations)} instances of '{{args}}' usage in the codebase"
        )
    else:
        print("No '{args}' usage found in the codebase.")


if __name__ == "__main__":
    main()
