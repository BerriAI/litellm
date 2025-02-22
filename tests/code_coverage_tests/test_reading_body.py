import ast
import os


class RequestBodyFinder(ast.NodeVisitor):
    def __init__(self):
        self.request_body_calls = []
        self.current_file = ""

    def visit_Attribute(self, node):
        # Check for request.body or request.json
        if isinstance(node.value, ast.Name) and node.value.id == "request":
            if node.attr in ["body", "json"]:
                # Get the line number for better tracking
                line_no = getattr(node, "lineno", 0)
                self.request_body_calls.append(
                    {
                        "file": self.current_file,
                        "line": line_no,
                        "access_type": node.attr,
                    }
                )
        self.generic_visit(node)


def find_request_body_in_file(file_path):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)
    finder = RequestBodyFinder()
    finder.current_file = file_path
    finder.visit(tree)
    return finder.request_body_calls


def test_find_request_body_usage():
    # Focus on the proxy directory
    proxy_dir = "../../litellm/proxy"
    request_body_instances = []

    for root, _, files in os.walk(proxy_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                instances = find_request_body_in_file(file_path)
                if instances:
                    request_body_instances.extend(instances)

    # Print findings
    print("\n=== Found request.body/json usage ===")
    for instance in request_body_instances:
        print(f"File: {instance['file']}")
        print(f"Line: {instance['line']}")
        print(f"Access Type: request.{instance['access_type']}")
        print("---")

    # Optionally raise an exception if you want to enforce specific rules
    # if len(request_body_instances) > 0:
    #     raise Exception(f"Found {len(request_body_instances)} instances of request.body/json usage")


if __name__ == "__main__":
    test_find_request_body_usage()
