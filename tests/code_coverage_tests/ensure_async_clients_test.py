import ast
import os

ALLOWED_FILES = [
    # local files
    "../../litellm/__init__.py",
    "../../litellm/llms/custom_httpx/http_handler.py",
    "../../litellm/router_utils/client_initalization_utils.py",
    "../../litellm/llms/custom_httpx/http_handler.py",
    "../../litellm/llms/huggingface_restapi.py",
    "../../litellm/llms/base.py",
    "../../litellm/llms/custom_httpx/httpx_handler.py",
    "../../litellm/llms/openai/common_utils.py",
    "../../litellm/experimental_mcp_client/client.py",
    # when running on ci/cd
    "./litellm/__init__.py",
    "./litellm/llms/custom_httpx/http_handler.py",
    "./litellm/router_utils/client_initalization_utils.py",
    "./litellm/llms/custom_httpx/http_handler.py",
    "./litellm/llms/huggingface_restapi.py",
    "./litellm/llms/base.py",
    "./litellm/llms/custom_httpx/httpx_handler.py",
    "./litellm/llms/openai/common_utils.py",
    "./litellm/experimental_mcp_client/client.py",
]

warning_msg = "this is a serious violation that can impact latency. Creating Async clients per request can add +500ms per request"


def check_for_async_http_handler(file_path):
    """
    Checks if AsyncHttpHandler is instantiated in the given file.
    Returns a list of line numbers where AsyncHttpHandler is used.
    """
    print("..checking file=", file_path)
    if file_path in ALLOWED_FILES:
        return []
    with open(file_path, "r") as file:
        try:
            tree = ast.parse(file.read())
        except SyntaxError:
            print(f"Warning: Syntax error in file {file_path}")
            return []

    violations = []
    target_names = [
        "AsyncHttpHandler",
        "AsyncHTTPHandler",
        "AsyncClient",
        "httpx.AsyncClient",
    ]  # Add variations here
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id.lower() in [
                name.lower() for name in target_names
            ]:
                raise ValueError(
                    f"found violation in file {file_path} line: {node.lineno}. Please use `get_async_httpx_client` instead. {warning_msg}"
                )
            # Check for attribute calls like httpx.AsyncClient()
            elif isinstance(node.func, ast.Attribute):
                full_name = ""
                current = node.func
                while isinstance(current, ast.Attribute):
                    full_name = "." + current.attr + full_name
                    current = current.value
                if isinstance(current, ast.Name):
                    full_name = current.id + full_name
                    if full_name.lower() in [name.lower() for name in target_names]:
                        raise ValueError(
                            f"found violation in file {file_path} line: {node.lineno}. Please use `get_async_httpx_client` instead. {warning_msg}"
                        )
    return violations


def scan_directory_for_async_handler(base_dir):
    """
    Scans all Python files in the directory tree for AsyncHttpHandler usage.
    Returns a dict of files and line numbers where violations were found.
    """
    violations = {}

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                file_violations = check_for_async_http_handler(file_path)
                if file_violations:
                    violations[file_path] = file_violations

    return violations


def test_no_async_http_handler_usage():
    """
    Test to ensure AsyncHttpHandler is not used anywhere in the codebase.
    """
    base_dir = "./litellm"  # Adjust this path as needed

    # base_dir = "../../litellm"  # LOCAL TESTING
    violations = scan_directory_for_async_handler(base_dir)

    if violations:
        violation_messages = []
        for file_path, line_numbers in violations.items():
            violation_messages.append(
                f"Found AsyncHttpHandler in {file_path} at lines: {line_numbers}"
            )
        raise AssertionError(
            "AsyncHttpHandler usage detected:\n" + "\n".join(violation_messages)
        )


if __name__ == "__main__":
    test_no_async_http_handler_usage()
