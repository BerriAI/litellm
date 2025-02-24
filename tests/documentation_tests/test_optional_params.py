import ast
from typing import List, Set, Dict, Optional
import sys


class ConfigChecker(ast.NodeVisitor):
    def __init__(self):
        self.errors: List[str] = []
        self.current_provider_block: Optional[str] = None
        self.param_assignments: Dict[str, Set[str]] = {}
        self.map_openai_calls: Set[str] = set()
        self.class_inheritance: Dict[str, List[str]] = {}

    def get_full_name(self, node):
        """Recursively extract the full name from a node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self.get_full_name(node.value)
            if base:
                return f"{base}.{node.attr}"
        return None

    def visit_ClassDef(self, node: ast.ClassDef):
        # Record class inheritance
        bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
        print(f"Found class {node.name} with bases {bases}")
        self.class_inheritance[node.name] = bases
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Check for map_openai_params calls
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "map_openai_params"
        ):
            if isinstance(node.func.value, ast.Name):
                config_name = node.func.value.id
                self.map_openai_calls.add(config_name)
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        # Detect custom_llm_provider blocks
        provider = self._extract_provider_from_if(node)
        if provider:
            old_provider = self.current_provider_block
            self.current_provider_block = provider
            self.generic_visit(node)
            self.current_provider_block = old_provider
        else:
            self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Track assignments to optional_params
        if self.current_provider_block and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                if target.value.id == "optional_params":
                    if isinstance(target.slice, ast.Constant):
                        key = target.slice.value
                        if self.current_provider_block not in self.param_assignments:
                            self.param_assignments[self.current_provider_block] = set()
                        self.param_assignments[self.current_provider_block].add(key)
        self.generic_visit(node)

    def _extract_provider_from_if(self, node: ast.If) -> Optional[str]:
        """Extract the provider name from an if condition checking custom_llm_provider"""
        if isinstance(node.test, ast.Compare):
            if len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq):
                if (
                    isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "custom_llm_provider"
                ):
                    if isinstance(node.test.comparators[0], ast.Constant):
                        return node.test.comparators[0].value
        return None

    def check_patterns(self) -> List[str]:
        # Check if all configs using map_openai_params inherit from BaseConfig
        for config_name in self.map_openai_calls:
            print(f"Checking config: {config_name}")
            if (
                config_name not in self.class_inheritance
                or "BaseConfig" not in self.class_inheritance[config_name]
            ):
                # Retrieve the associated class name, if any
                class_name = next(
                    (
                        cls
                        for cls, bases in self.class_inheritance.items()
                        if config_name in bases
                    ),
                    "Unknown Class",
                )
                self.errors.append(
                    f"Error: {config_name} calls map_openai_params but doesn't inherit from BaseConfig. "
                    f"It is used in the class: {class_name}"
                )

        # Check for parameter assignments in provider blocks
        for provider, params in self.param_assignments.items():
            # You can customize which parameters should raise warnings for each provider
            for param in params:
                if param not in self._get_allowed_params(provider):
                    self.errors.append(
                        f"Warning: Parameter '{param}' is directly assigned in {provider} block. "
                        f"Consider using a config class instead."
                    )

        return self.errors

    def _get_allowed_params(self, provider: str) -> Set[str]:
        """Define allowed direct parameter assignments for each provider"""
        # You can customize this based on your requirements
        common_allowed = {"stream", "api_key", "api_base"}
        provider_specific = {
            "anthropic": {"api_version"},
            "openai": {"organization"},
            # Add more providers and their allowed params here
        }
        return common_allowed.union(provider_specific.get(provider, set()))


def check_file(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())

    checker = ConfigChecker()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_optional_params":
            checker.visit(node)
            break  # No need to visit other functions
    return checker.check_patterns()


def main():
    file_path = "../../litellm/utils.py"
    errors = check_file(file_path)

    if errors:
        print("\nFound the following issues:")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)
    else:
        print("No issues found!")
        sys.exit(0)


if __name__ == "__main__":
    main()
