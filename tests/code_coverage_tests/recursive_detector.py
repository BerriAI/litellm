import ast
import os

IGNORE_FUNCTIONS = [
    "_format_type",
    "_remove_additional_properties",
    "_remove_strict_from_schema",
    "filter_schema_fields",
    "text_completion",
    "_check_for_os_environ_vars",
    "clean_message",
    "unpack_defs",
    "convert_anyof_null_to_nullable",  # has a set max depth
    "add_object_type",
    "strip_field",
    "_transform_prompt",
    "mask_dict",
    "_serialize",  # we now set a max depth for this
    "_sanitize_request_body_for_spend_logs_payload",  # testing added for circular reference
    "_sanitize_value",  # testing added for circular reference
    "set_schema_property_ordering",  # testing added for infinite recursion
    "process_items",  # testing added for infinite recursion + max depth set.
    "_can_object_call_model",  # max depth set.
    "encode_unserializable_types",  # max depth set.
    "filter_value_from_dict",  # max depth set.
    "normalize_json_schema_types",  # max depth set.
    "_extract_fields_recursive",  # max depth set.
    "_remove_json_schema_refs",  # max depth set.,
    "_convert_schema_types",  # max depth set.,
    "_fix_enum_empty_strings",  # max depth set.,
    "get_access_token",  # max depth set.,
    "_redact_base64",  # max depth set.
    "_contains_vision_content",  # max depth set.
    "_read_all_bytes",  # max depth set.
    "_fix_enum_types",  # max depth set.
    "_collect_argument_paths",  # max depth set.
    "_split_text",  # max depth set.
    "_delete_nested_value_custom",  # max depth set (bounded by number of path segments).
    "filter_exceptions_from_params",  # max depth set (default 20) to prevent infinite recursion.
]


class RecursiveFunctionFinder(ast.NodeVisitor):
    def __init__(self):
        self.recursive_functions = []
        self.ignored_recursive_functions = []

    def visit_FunctionDef(self, node):
        # Check if the function calls itself
        if any(self._is_recursive_call(node, call) for call in ast.walk(node)):
            if node.name in IGNORE_FUNCTIONS:
                self.ignored_recursive_functions.append(node.name)
            else:
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
    return finder.recursive_functions, finder.ignored_recursive_functions


def find_recursive_functions_in_directory(directory):
    recursive_functions = {}
    ignored_recursive_functions = {}
    for root, _, files in os.walk(directory):
        for file in files:
            print("file: ", file)
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                functions, ignored = find_recursive_functions_in_file(file_path)
                if functions:
                    recursive_functions[file_path] = functions
                if ignored:
                    ignored_recursive_functions[file_path] = ignored
    return recursive_functions, ignored_recursive_functions


if __name__ == "__main__":
    # Example usage
    # raise exception if any recursive functions are found, except for the ignored ones
    # this is used in the CI/CD pipeline to prevent recursive functions from being merged

    directory_path = "./litellm"
    recursive_functions, ignored_recursive_functions = (
        find_recursive_functions_in_directory(directory_path)
    )
    print("UNIGNORED RECURSIVE FUNCTIONS: ", recursive_functions)
    print("IGNORED RECURSIVE FUNCTIONS: ", ignored_recursive_functions)

    if len(recursive_functions) > 0:
        # raise exception if any recursive functions are found
        for file, functions in recursive_functions.items():
            print(
                f"🚨 Unignored recursive functions found in {file}: {functions}. THIS IS REALLY BAD, it has caused CPU Usage spikes in the past. Only keep this if it's ABSOLUTELY necessary."
            )
        file, functions = list(recursive_functions.items())[0]
        raise Exception(
            f"🚨 Unignored recursive functions found include {file}: {functions}. THIS IS REALLY BAD, it has caused CPU Usage spikes in the past. Only keep this if it's ABSOLUTELY necessary."
        )
