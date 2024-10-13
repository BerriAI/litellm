import ast
import os


def get_function_names_from_file(file_path):
    """
    Extracts all function names from a given Python file.
    """
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())

    return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


def get_all_functions_called_in_tests(tests_dir):
    """
    Returns a set of function names that are called in any test function inside the tests directory.
    """
    called_functions = set()

    for root, _, files in os.walk(tests_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        called_functions.add(node.func.id)

    return called_functions


def get_functions_from_router(file_path):
    """
    Extracts all functions defined in router.py.
    """
    return get_function_names_from_file(file_path)


ignored_function_names = ["__init__"]


def main():
    router_file = "./litellm/router.py"  # Update this path if it's located elsewhere
    tests_dir = (
        "./tests/"  # Update this path if your tests directory is located elsewhere
    )

    router_functions = get_functions_from_router(router_file)
    called_functions_in_tests = get_all_functions_called_in_tests(tests_dir)

    untested_functions = [
        fn for fn in router_functions if fn not in called_functions_in_tests
    ]

    if untested_functions:
        print("The following functions in router.py are not tested:")
        all_untested_functions = []
        for func in untested_functions:
            if func not in ignored_function_names:
                all_untested_functions.append(func)
        if len(all_untested_functions) > 0:
            raise Exception(f"Functions not tested: {all_untested_functions}")
    else:
        print("All functions in router.py are covered by tests.")


if __name__ == "__main__":
    main()
