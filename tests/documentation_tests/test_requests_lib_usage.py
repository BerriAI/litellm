"""
Prevent usage of 'requests' library in the codebase.
"""

import os
import ast
import sys
from typing import List, Tuple


def find_requests_usage(directory: str) -> List[Tuple[str, int, str]]:
    """
    Recursively search for Python files in the given directory
    and find usages of the 'requests' library.

    Args:
        directory (str): The root directory to search for Python files

    Returns:
        List of tuples containing (file_path, line_number, usage_type)
    """
    requests_usages = []

    def scan_file(file_path: str):
        """
        Scan a single Python file for requests library usage
        """
        try:
            # Use utf-8-sig to handle files with BOM, ignore errors
            with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as file:
                tree = ast.parse(file.read())

            for node in ast.walk(tree):
                # Check import statements
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("requests"):
                            requests_usages.append(
                                (file_path, node.lineno, f"Import: {alias.name}")
                            )

                # Check import from statements
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("requests"):
                        requests_usages.append(
                            (file_path, node.lineno, f"Import from: {node.module}")
                        )

                # Check method calls
                elif isinstance(node, ast.Call):
                    # Check function calls that might be from requests
                    try:
                        func_name = ast.unparse(node.func)
                        if "requests." in func_name and func_name.startswith(
                            "requests."
                        ):
                            requests_usages.append(
                                (file_path, node.lineno, f"Method Call: {func_name}")
                            )
                    except:
                        pass

                # Check attribute access
                elif isinstance(node, ast.Attribute):
                    try:
                        attr_name = ast.unparse(node)
                        if "requests." in attr_name and attr_name.startswith(
                            "requests."
                        ):
                            requests_usages.append(
                                (
                                    file_path,
                                    node.lineno,
                                    f"Attribute Access: {attr_name}",
                                )
                            )
                    except:
                        pass

        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    # Recursively walk through directory
    for root, dirs, files in os.walk(directory):
        # Remove virtual environment and cache directories from search
        dirs[:] = [
            d
            for d in dirs
            if not any(
                venv in d
                for venv in [
                    "venv",
                    "env",
                    "myenv",
                    ".venv",
                    "__pycache__",
                    ".pytest_cache",
                ]
            )
        ]

        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                # Skip files in virtual environment or cache directories
                if not any(
                    venv in full_path
                    for venv in [
                        "venv",
                        "env",
                        "myenv",
                        ".venv",
                        "__pycache__",
                        ".pytest_cache",
                    ]
                ):
                    scan_file(full_path)

    return requests_usages


def main():
    # Get directory from command line argument or use current directory
    directory = "../../litellm"

    # Find requests library usages
    results = find_requests_usage(directory)

    # Print results
    if results:
        print("Requests Library Usages Found:")
        for file_path, line_num, usage_type in results:
            print(f"{file_path}:{line_num} - {usage_type}")
    else:
        print("No requests library usages found.")


if __name__ == "__main__":
    main()
