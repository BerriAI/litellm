import ast
import os

ALLOWED_FILES = [
    # local files
    "../../litellm/litellm_core_utils/litellm.logging_callback_manager.py",
    "../../litellm/proxy/common_utils/callback_utils.py",
    # when running on ci/cd
    "./litellm/litellm_core_utils/litellm.logging_callback_manager.py",
    "./litellm/proxy/common_utils/callback_utils.py",
]

warning_msg = "this is a serious violation. Callbacks must only be modified through LoggingCallbackManager"


def check_for_callback_modifications(file_path):
    """
    Checks if any direct modifications to specific litellm callback lists are made in the given file.
    Also prints the violating line of code.
    """
    print("..checking file=", file_path)
    if file_path in ALLOWED_FILES:
        return []

    violations = []
    with open(file_path, "r") as file:
        try:
            lines = file.readlines()
            tree = ast.parse("".join(lines))
        except SyntaxError:
            print(f"Warning: Syntax error in file {file_path}")
            return violations

    protected_lists = [
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ]

    forbidden_operations = ["append", "extend", "insert"]

    for node in ast.walk(tree):
        # Check for attribute calls like litellm.callbacks.append()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # Get the full attribute chain
            attr_chain = []
            current = node.func
            while isinstance(current, ast.Attribute):
                attr_chain.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                attr_chain.append(current.id)

            # Reverse to get the chain from root to leaf
            attr_chain = attr_chain[::-1]

            # Check if the attribute chain starts with 'litellm' and modifies a protected list
            if (
                len(attr_chain) >= 3
                and attr_chain[0] == "litellm"
                and attr_chain[2] in forbidden_operations
            ):
                protected_list = attr_chain[1]
                operation = attr_chain[2]
                if (
                    protected_list in protected_lists
                    and operation in forbidden_operations
                ):
                    violating_line = lines[node.lineno - 1].strip()
                    violations.append(
                        f"Found violation in file {file_path} line {node.lineno}: '{violating_line}'. "
                        f"Direct modification of 'litellm.{protected_list}' using '{operation}' is not allowed. "
                        f"Please use LoggingCallbackManager instead. {warning_msg}"
                    )

    return violations


def scan_directory_for_callback_modifications(base_dir):
    """
    Scans all Python files in the directory tree for unauthorized callback list modifications.
    """
    all_violations = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                violations = check_for_callback_modifications(file_path)
                all_violations.extend(violations)
    return all_violations


def test_no_unauthorized_callback_modifications():
    """
    Test to ensure callback lists are not modified directly anywhere in the codebase.
    """
    base_dir = "./litellm"  # Adjust this path as needed
    # base_dir = "../../litellm"  # LOCAL TESTING

    violations = scan_directory_for_callback_modifications(base_dir)
    if violations:
        print(f"\nFound {len(violations)} callback modification violations:")
        for violation in violations:
            print("\n" + violation)
        raise AssertionError(
            "Found unauthorized callback modifications. See above for details."
        )


if __name__ == "__main__":
    test_no_unauthorized_callback_modifications()
