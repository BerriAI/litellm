import ast
import os

pass_through_classes = [
    "AnthropicPassthroughLoggingHandler",
]


def get_function_names_from_file(file_path):
    """
    Extracts all function names from a given Python file.
    """
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())

    function_names = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            if node.name in pass_through_classes:
                for class_node in node.body:
                    if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        function_names.append(class_node.name)

    return function_names


def get_all_functions_called_in_tests(base_dir):
    """
    Returns a set of function names that are called in test functions
    inside 'local_testing' and 'router_unit_test' directories,
    specifically in files containing the word 'router'.
    """
    called_functions = set()
    test_dirs = ["pass_through_unit_tests"]

    for test_dir in test_dirs:
        dir_path = os.path.join(base_dir, test_dir)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist.")
            continue

        print("dir_path: ", dir_path)
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".py"):
                    print("file: ", file)
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        try:
                            tree = ast.parse(f.read())
                        except SyntaxError:
                            print(f"Warning: Syntax error in file {file_path}")
                            continue

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Attribute) and isinstance(
                                node.func.value, ast.Name
                            ):
                                if node.func.value.id in pass_through_classes:
                                    # If it's called via the class, add both the original and non-underscore versions
                                    method_name = node.func.attr
                                    called_functions.add(method_name)
                                    if method_name.startswith("_"):
                                        called_functions.add(method_name.lstrip("_"))

    return called_functions


def get_functions_from_router(file_path):
    """
    Extracts all functions defined in router.py.
    """
    return get_function_names_from_file(file_path)


ignored_function_names = [
    "__init__",
]


def main():
    router_file = [
        "../../litellm/proxy/pass_through_endpoints/llm_provider_handlers/anthropic_passthrough_logging_handler.py",
    ]
    tests_dir = "../../tests/"

    router_functions = []
    for file in router_file:
        router_functions.extend(get_functions_from_router(file))
    print("router_functions: ", router_functions)
    called_functions_in_tests = get_all_functions_called_in_tests(tests_dir)
    print("called_functions_in_tests: ", called_functions_in_tests)

    untested_functions = []
    for fn in router_functions:
        # Check if the function is called either with or without leading underscore
        clean_name = fn.lstrip("_")
        if (
            fn not in called_functions_in_tests
            and clean_name not in called_functions_in_tests
        ):
            untested_functions.append(fn)

    if untested_functions:
        all_untested_functions = []
        for func in untested_functions:
            if func not in ignored_function_names:
                all_untested_functions.append(func)
        untested_perc = (len(all_untested_functions)) / len(router_functions)
        print("untested_perc: ", untested_perc)
        if untested_perc > 0:
            print("The following functions in pass_through/ are not tested:")
            raise Exception(
                f"{untested_perc * 100:.2f}% of functions in pass_through/ are not tested: {all_untested_functions}"
            )
    else:
        print("All functions in router.py are covered by tests.")


if __name__ == "__main__":
    main()
