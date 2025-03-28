import ast
from typing import List, Dict, Set, Optional
import os
from dataclasses import dataclass
import argparse
import re
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm


@dataclass
class FunctionInfo:
    """Store function information."""

    name: str
    docstring: Optional[str]
    parameters: Set[str]
    file_path: str
    line_number: int


class FastAPIDocVisitor(ast.NodeVisitor):
    """AST visitor to find FastAPI endpoint functions."""

    def __init__(self, target_functions: Set[str]):
        self.target_functions = target_functions
        self.functions: Dict[str, FunctionInfo] = {}
        self.current_file = ""

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Visit function definitions (both async and sync) and collect info if they match target functions."""
        if node.name in self.target_functions:
            # Extract docstring
            docstring = ast.get_docstring(node)

            # Extract parameters
            parameters = set()
            for arg in node.args.args:
                if arg.annotation is not None:
                    # Get the parameter type from annotation
                    if isinstance(arg.annotation, ast.Name):
                        parameters.add((arg.arg, arg.annotation.id))
                    elif isinstance(arg.annotation, ast.Subscript):
                        if isinstance(arg.annotation.value, ast.Name):
                            parameters.add((arg.arg, arg.annotation.value.id))

            self.functions[node.name] = FunctionInfo(
                name=node.name,
                docstring=docstring,
                parameters=parameters,
                file_path=self.current_file,
                line_number=node.lineno,
            )

    # Also need to add this to handle async functions
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handle async functions by delegating to the regular function visitor."""
        return self.visit_FunctionDef(node)


def find_functions_in_file(
    file_path: str, target_functions: Set[str]
) -> Dict[str, FunctionInfo]:
    """Find target functions in a Python file using AST."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        visitor = FastAPIDocVisitor(target_functions)
        visitor.current_file = file_path
        tree = ast.parse(content)
        visitor.visit(tree)
        return visitor.functions

    except Exception as e:
        print(f"Error parsing {file_path}: {str(e)}")
        return {}


def extract_docstring_params(docstring: Optional[str]) -> Set[str]:
    """Extract parameter names from docstring."""
    if not docstring:
        return set()

    params = set()
    # Match parameters in format:
    # - parameter_name: description
    # or
    # parameter_name: description
    param_pattern = r"-?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:"

    for match in re.finditer(param_pattern, docstring):
        params.add(match.group(1))

    return params


def analyze_function(func_info: FunctionInfo) -> Dict:
    """Analyze function documentation and return validation results."""

    docstring_params = extract_docstring_params(func_info.docstring)

    print(f"func_info.parameters: {func_info.parameters}")
    pydantic_params = set()

    for name, type_name in func_info.parameters:
        if type_name.endswith("Request") or type_name.endswith("Response"):
            pydantic_model = getattr(litellm.proxy._types, type_name, None)
            if pydantic_model is not None:
                for param in pydantic_model.model_fields.keys():
                    pydantic_params.add(param)

    print(f"pydantic_params: {pydantic_params}")

    missing_params = pydantic_params - docstring_params

    return {
        "function": func_info.name,
        "file_path": func_info.file_path,
        "line_number": func_info.line_number,
        "has_docstring": bool(func_info.docstring),
        "pydantic_params": list(pydantic_params),
        "documented_params": list(docstring_params),
        "missing_params": list(missing_params),
        "is_valid": len(missing_params) == 0,
    }


def print_validation_results(results: Dict) -> None:
    """Print validation results in a readable format."""
    print(f"\nChecking function: {results['function']}")
    print(f"File: {results['file_path']}:{results['line_number']}")
    print("-" * 50)

    if not results["has_docstring"]:
        print("❌ No docstring found!")
        return

    if not results["pydantic_params"]:
        print("ℹ️  No Pydantic input models found.")
        return

    if results["is_valid"]:
        print("✅ All Pydantic parameters are documented!")
    else:
        print("❌ Missing documentation for parameters:")
        for param in sorted(results["missing_params"]):
            print(f"  - {param}")


def main():
    function_names = [
        "new_end_user",
        "end_user_info",
        "update_end_user",
        "delete_end_user",
        "generate_key_fn",
        "info_key_fn",
        "update_key_fn",
        "delete_key_fn",
        "new_user",
        "new_team",
        "team_info",
        "update_team",
        "delete_team",
        "new_organization",
        "update_organization",
        "delete_organization",
        "list_organization",
        "user_update",
        "new_budget",
        "info_budget",
        "update_budget",
        "delete_budget",
        "list_budget",
    ]
    # directory = "../../litellm/proxy/management_endpoints"  # LOCAL
    directory = "./litellm/proxy/management_endpoints"

    # Convert function names to set for faster lookup
    target_functions = set(function_names)
    found_functions: Dict[str, FunctionInfo] = {}

    # Walk through directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                found = find_functions_in_file(file_path, target_functions)
                found_functions.update(found)

    # Analyze and output results
    for func_name in function_names:
        if func_name in found_functions:
            result = analyze_function(found_functions[func_name])
            if not result["is_valid"]:
                raise Exception(print_validation_results(result))
    #         results.append(result)
    #         print_validation_results(result)

    # # Exit with error code if any validation failed
    # if any(not r["is_valid"] for r in results):
    #     exit(1)


if __name__ == "__main__":
    main()
