import ast
import os


def get_function_names_from_file(file_path):
    """
    Extracts all function names from a given Python file.
    """
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())

    function_names = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.append(node.name)

    return function_names


def get_all_functions_called_in_tests(base_dir):
    """
    Returns a set of function names that are called in test functions
    inside 'local_testing' and 'router_unit_test' directories,
    specifically in files containing the word 'router'.
    """
    called_functions = set()
    test_dirs = ["local_testing", "router_unit_tests"]

    for test_dir in test_dirs:
        dir_path = os.path.join(base_dir, test_dir)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist.")
            continue

        print("dir_path: ", dir_path)
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".py") and "router" in file.lower():
                    print("file: ", file)
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        try:
                            tree = ast.parse(f.read())
                        except SyntaxError:
                            print(f"Warning: Syntax error in file {file_path}")
                            continue
                    if file == "test_router_validate_fallbacks.py":
                        print(f"tree: {tree}")
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call) and isinstance(
                            node.func, ast.Name
                        ):
                            called_functions.add(node.func.id)
                        elif isinstance(node, ast.Call) and isinstance(
                            node.func, ast.Attribute
                        ):
                            called_functions.add(node.func.attr)

    return called_functions


def get_functions_from_router(file_path):
    """
    Extracts all functions defined in router.py.
    """
    return get_function_names_from_file(file_path)


ignored_function_names = [
    "__init__",
    "_acreate_file",
    "_acreate_batch",
    "acreate_assistants",
    "adelete_assistant",
    "aget_assistants",
    "acreate_thread",
    "aget_thread",
    "a_add_message",
    "aget_messages",
    "arun_thread",
]


def main():
    router_file = "./litellm/router.py"  # Update this path if it's located elsewhere
    # router_file = "../../litellm/router.py"  ## LOCAL TESTING
    tests_dir = (
        "./tests/"  # Update this path if your tests directory is located elsewhere
    )
    # tests_dir = "../../tests/"  # LOCAL TESTING

    router_functions = get_functions_from_router(router_file)
    print("router_functions: ", router_functions)
    called_functions_in_tests = get_all_functions_called_in_tests(tests_dir)
    untested_functions = [
        fn for fn in router_functions if fn not in called_functions_in_tests
    ]

    if untested_functions:
        all_untested_functions = []
        for func in untested_functions:
            if func not in ignored_function_names:
                all_untested_functions.append(func)
        untested_perc = (len(all_untested_functions)) / len(router_functions)
        print("perc_covered: ", untested_perc)
        if untested_perc < 0.3:
            print("The following functions in router.py are not tested:")
            raise Exception(
                f"{untested_perc * 100:.2f}% of functions in router.py are not tested: {all_untested_functions}"
            )
    else:
        print("All functions in router.py are covered by tests.")


if __name__ == "__main__":
    main()
