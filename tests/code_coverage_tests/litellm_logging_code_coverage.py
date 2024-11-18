import ast
import os
from typing import List


def get_function_names_from_file(file_path: str) -> List[str]:
    """
    Extracts all static method names from litellm_logging.py
    """
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())

    function_names = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            # Functions inside classes
            for class_node in node.body:
                if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check if the function has @staticmethod decorator
                    for decorator in class_node.decorator_list:
                        if (
                            isinstance(decorator, ast.Name)
                            and decorator.id == "staticmethod"
                        ):
                            function_names.append(class_node.name)

    return function_names


def get_all_functions_called_in_tests(base_dir: str) -> set:
    """
    Returns a set of function names that are called in test functions
    inside test files containing the word 'logging'.
    """
    called_functions = set()

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py") and "logging" in file.lower():
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    try:
                        tree = ast.parse(f.read())
                    except SyntaxError:
                        print(f"Warning: Syntax error in file {file_path}")
                        continue

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Name):
                                called_functions.add(node.func.id)
                            elif isinstance(node.func, ast.Attribute):
                                called_functions.add(node.func.attr)

    return called_functions


# Functions that can be ignored in test coverage
ignored_function_names = [
    "__init__",
    # Add other functions to ignore here
]


def main():
    logging_file = "./litellm/litellm_core_utils/litellm_logging.py"
    tests_dir = "./tests/"

    # LOCAL TESTING
    # logging_file = "../../litellm/litellm_core_utils/litellm_logging.py"
    # tests_dir = "../../tests/"

    logging_functions = get_function_names_from_file(logging_file)
    print("logging_functions:", logging_functions)

    called_functions_in_tests = get_all_functions_called_in_tests(tests_dir)
    untested_functions = [
        fn
        for fn in logging_functions
        if fn not in called_functions_in_tests and fn not in ignored_function_names
    ]

    if untested_functions:
        untested_perc = len(untested_functions) / len(logging_functions)
        print(f"untested_percentage: {untested_perc * 100:.2f}%")
        raise Exception(
            f"{untested_perc * 100:.2f}% of functions in litellm_logging.py are not tested: {untested_functions}"
        )
    else:
        print("All functions in litellm_logging.py are covered by tests.")


if __name__ == "__main__":
    main()
