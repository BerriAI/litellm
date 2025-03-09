import ast
import os

MAX_FUNCTION_LINES = 100


def get_function_line_counts(file_path):
    """
    Extracts all function names and their line counts from a given Python file.
    """
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())

    function_line_counts = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Top-level functions
            line_count = node.end_lineno - node.lineno + 1
            function_line_counts.append((node.name, line_count))
        elif isinstance(node, ast.ClassDef):
            # Functions inside classes
            for class_node in node.body:
                if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    line_count = class_node.end_lineno - class_node.lineno + 1
                    function_line_counts.append((class_node.name, line_count))

    return function_line_counts


ignored_functions = [
    "__init__",
]


def check_function_lengths(router_file):
    """
    Checks if any function in the specified file exceeds the maximum allowed length.
    """
    function_line_counts = get_function_line_counts(router_file)
    long_functions = [
        (name, count)
        for name, count in function_line_counts
        if count > MAX_FUNCTION_LINES and name not in ignored_functions
    ]

    if long_functions:
        print("The following functions exceed the allowed line count:")
        for name, count in long_functions:
            print(f"- {name}: {count} lines")
        raise Exception(
            f"{len(long_functions)} functions in {router_file} exceed {MAX_FUNCTION_LINES} lines"
        )
    else:
        print("All functions in the router file are within the allowed line limit.")


def main():
    # Update this path to point to the correct location of router.py
    router_file = "../../litellm/router.py"  # LOCAL TESTING

    check_function_lengths(router_file)


if __name__ == "__main__":
    main()
