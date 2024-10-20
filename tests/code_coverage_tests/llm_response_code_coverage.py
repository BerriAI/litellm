import ast
import os


def get_function_names_from_file(file_path):
    """
    Extracts all function names from a given Python file.
    """
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())

    function_names = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Top-level functions
            function_names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            # Functions inside classes
            for class_node in node.body:
                if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function_names.append(class_node.name)

    return function_names


def get_all_functions_called_in_tests(base_dir):
    """
    Returns a set of function names that are called in test functions
    inside 'llm_translation' directory.
    """
    called_functions = set()
    test_dirs = ["llm_translation/tests_llm_response_utils"]

    for test_dir in test_dirs:
        dir_path = os.path.join(base_dir, test_dir)
        print("dir_path: ", dir_path)
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
                        if isinstance(node, ast.Call) and isinstance(
                            node.func, ast.Name
                        ):
                            called_functions.add(node.func.id)
                        elif isinstance(node, ast.Call) and isinstance(
                            node.func, ast.Attribute
                        ):
                            called_functions.add(node.func.attr)

    return called_functions


def get_functions_from_llm_response_utils(file_path):
    """
    Extracts all functions defined in llm_response_utils.py.
    """
    return get_function_names_from_file(file_path)


ignored_function_names = [
    "__init__",
]


def main():
    llm_response_utils_folder = "./litellm/litellm_core_utils/llm_response_utils"
    llm_response_utils_files = []

    # get all files in llm_response_utils folder
    for root, _, files in os.walk(llm_response_utils_folder):
        for file in files:
            if file.endswith(".py"):
                llm_response_utils_files.append(os.path.join(root, file))

    print("all llm_response_utils_files: ", llm_response_utils_files)

    tests_dir = (
        "./tests/"  # Update this path if your tests directory is located elsewhere
    )
    # tests_dir = "../../tests/"  # LOCAL TESTING

    llm_response_utils_functions = []
    for file in llm_response_utils_files:
        llm_response_utils_functions.extend(get_functions_from_llm_response_utils(file))
    print("llm_response_utils_functions: ", llm_response_utils_functions)
    called_functions_in_tests = get_all_functions_called_in_tests(tests_dir)
    print("called_functions_in_tests: ", called_functions_in_tests)
    untested_functions = [
        fn for fn in llm_response_utils_functions if fn not in called_functions_in_tests
    ]

    if untested_functions:
        all_untested_functions = []
        for func in untested_functions:
            if func not in ignored_function_names:
                all_untested_functions.append(func)
        untested_perc = (len(all_untested_functions)) / len(
            llm_response_utils_functions
        )
        print("untested_perc: ", untested_perc)
        if untested_perc > 0:
            print("The following functions in llm_response_utils.py are not tested:")
            raise Exception(
                f"{untested_perc * 100:.2f}% of functions in llm_response_utils.py are not tested: {all_untested_functions}"
            )
    else:
        print("All functions in llm_response_utils.py are covered by tests.")


if __name__ == "__main__":
    main()
