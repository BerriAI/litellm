"""
Test that all cache calls in async functions in router_strategy/ are async

"""

import os
import sys
from typing import Dict, List, Tuple
import ast

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import os


class AsyncCacheCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.async_functions: Dict[str, List[Tuple[str, int]]] = {}
        self.current_function = None

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definitions and store their cache calls"""
        self.current_function = node.name
        self.async_functions[node.name] = []
        self.generic_visit(node)
        self.current_function = None

    def visit_Call(self, node):
        """Visit function calls and check for cache operations"""
        if self.current_function is not None:
            # Check if it's a cache-related call
            if isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
                if any(keyword in method_name.lower() for keyword in ["cache"]):
                    # Get the full method call path
                    if isinstance(node.func.value, ast.Name):
                        full_call = f"{node.func.value.id}.{method_name}"
                    elif isinstance(node.func.value, ast.Attribute):
                        # Handle nested attributes like self.router_cache.get
                        parts = []
                        current = node.func.value
                        while isinstance(current, ast.Attribute):
                            parts.append(current.attr)
                            current = current.value
                        if isinstance(current, ast.Name):
                            parts.append(current.id)
                        parts.reverse()
                        parts.append(method_name)
                        full_call = ".".join(parts)
                    else:
                        full_call = method_name
                    # Store both the call and its line number
                    self.async_functions[self.current_function].append(
                        (full_call, node.lineno)
                    )
        self.generic_visit(node)


def get_python_files(directory: str) -> List[str]:
    """Get all Python files in the router_strategy directory"""
    python_files = []
    for file in os.listdir(directory):
        if file.endswith(".py") and not file.startswith("__"):
            python_files.append(os.path.join(directory, file))
    return python_files


def analyze_file(file_path: str) -> Dict[str, List[Tuple[str, int]]]:
    """Analyze a Python file for async functions and their cache calls"""
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())

    visitor = AsyncCacheCallVisitor()
    visitor.visit(tree)
    return visitor.async_functions


def test_router_strategy_async_cache_calls():
    """Test that all cache calls in async functions are properly async"""
    strategy_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "litellm",
        "router_strategy",
    )

    # Get all Python files in the router_strategy directory
    python_files = get_python_files(strategy_dir)

    print("python files:", python_files)

    all_async_functions: Dict[str, Dict[str, List[Tuple[str, int]]]] = {}

    for file_path in python_files:
        file_name = os.path.basename(file_path)
        async_functions = analyze_file(file_path)

        if async_functions:
            all_async_functions[file_name] = async_functions
            print(f"\nAnalyzing {file_name}:")

            for func_name, cache_calls in async_functions.items():
                print(f"\nAsync function: {func_name}")
                print(f"Cache calls found: {cache_calls}")

                # Assert that cache calls in async functions use async methods
                for call, line_number in cache_calls:
                    if any(keyword in call.lower() for keyword in ["cache"]):
                        assert (
                            "async" in call.lower()
                        ), f"VIOLATION: Cache call '{call}' in async function '{func_name}' should be async. file path: {file_path}, line number: {line_number}"

    # Assert we found async functions to analyze
    assert (
        len(all_async_functions) > 0
    ), "No async functions found in router_strategy directory"


if __name__ == "__main__":
    test_router_strategy_async_cache_calls()
