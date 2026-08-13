import ast
import os
import re


def find_azure_files(base_dir):
    """
    Find all Python files in the Azure directory.
    """
    azure_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                azure_files.append(os.path.join(root, file))
    return azure_files


def check_direct_instantiation(file_path):
    """
    Check if a file directly instantiates AzureOpenAI or AsyncAzureOpenAI
    outside of the BaseAzureLLM class methods.
    """
    with open(file_path, "r") as file:
        content = file.read()

    # Parse the file
    tree = ast.parse(content)

    # Track issues found
    issues = []

    # Find all class definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name

            # Skip BaseAzureLLM class since it's allowed to define the client creation methods
            if class_name == "BaseAzureLLM":
                continue

            # Check method bodies for direct instantiation
            for method in node.body:
                if isinstance(method, ast.FunctionDef) or isinstance(
                    method, ast.AsyncFunctionDef
                ):
                    method_name = method.name

                    # Skip methods that are specifically for client creation
                    if method_name in [
                        "get_azure_openai_client",
                        "initialize_azure_sdk_client",
                    ]:
                        continue

                    # Look for direct instantiation in the method body
                    for subnode in ast.walk(method):
                        if isinstance(subnode, ast.Call):
                            if hasattr(subnode, "func") and hasattr(subnode.func, "id"):
                                if subnode.func.id in [
                                    "AzureOpenAI",
                                    "AsyncAzureOpenAI",
                                ]:
                                    issues.append(
                                        f"Direct instantiation of {subnode.func.id} in {class_name}.{method_name}"
                                    )
                            elif hasattr(subnode, "func") and hasattr(
                                subnode.func, "attr"
                            ):
                                if subnode.func.attr in [
                                    "AzureOpenAI",
                                    "AsyncAzureOpenAI",
                                ]:
                                    issues.append(
                                        f"Direct instantiation of {subnode.func.attr} in {class_name}.{method_name}"
                                    )

    return issues


def main():
    """
    Main function to run the test.
    """
    # local
    base_dir = "../../litellm/llms/azure"
    azure_files = find_azure_files(base_dir)
    print(f"Found {len(azure_files)} Azure Python files to check")

    all_issues = []

    for file_path in azure_files:
        issues = check_direct_instantiation(file_path)
        if issues:
            all_issues.extend([f"{file_path}: {issue}" for issue in issues])

    if all_issues:
        print("Found direct instantiations of AzureOpenAI or AsyncAzureOpenAI:")
        for issue in all_issues:
            print(f"  - {issue}")
        raise Exception(
            f"Found {len(all_issues)} direct instantiations of AzureOpenAI or AsyncAzureOpenAI classes. Use get_azure_openai_client instead."
        )
    else:
        print("All Azure modules are correctly using get_azure_openai_client!")


if __name__ == "__main__":
    main()
