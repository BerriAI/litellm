import ast
import os

IGNORE_Functions = []


class RecursiveFunctionFinder(ast.NodeVisitor):
    def __init__(self):
        self.recursive_functions = []

    def visit_FunctionDef(self, node):
        # Check if the function calls itself
        if any(self._is_recursive_call(node, call) for call in ast.walk(node)):
            self.recursive_functions.append(node.name)
        self.generic_visit(node)

    def _is_recursive_call(self, func_node, call_node):
        # Check if the call node is a function call
        if not isinstance(call_node, ast.Call):
            return False

        # Case 1: Direct function call (e.g., my_func())
        if isinstance(call_node.func, ast.Name) and call_node.func.id == func_node.name:
            return True

        # Case 2: Method call with self (e.g., self.my_func())
        if isinstance(call_node.func, ast.Attribute) and isinstance(
            call_node.func.value, ast.Name
        ):
            return (
                call_node.func.value.id == "self"
                and call_node.func.attr == func_node.name
            )

        return False


def find_recursive_functions_in_file(file_path):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)
    finder = RecursiveFunctionFinder()
    finder.visit(tree)
    return finder.recursive_functions


def find_recursive_functions_in_directory(directory):
    recursive_functions = {}
    for root, _, files in os.walk(directory):
        for file in files:
            print("file: ", file)
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                functions = find_recursive_functions_in_file(file_path)
                if functions:
                    recursive_functions[file_path] = functions
    return recursive_functions


# Example usage
directory_path = "../../litellm"
recursive_functions = find_recursive_functions_in_directory(directory_path)
for file, functions in recursive_functions.items():
    print(f"File: {file}")
    for function in functions:
        print(f"  Recursive Function: {function}")
