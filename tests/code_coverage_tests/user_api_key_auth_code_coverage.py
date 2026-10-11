"""
Enforce all functions in user_api_key_auth.py are covered by tests
"""

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
    inside the unit/proxy directory,
    specifically in files containing the word 'router'.
    """
    called_functions = set()
    test_dirs = ["unit", "integration"]

    for test_dir in test_dirs:
        dir_path = os.path.join(base_dir, test_dir)
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
                    if file == "test_router_validate_fallbacks.py":
                        print(f"tree: {tree}")
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


def get_functions_from_router(file_path):
    """
    Extracts all functions defined in user_api_key_auth.py.
    """
    return get_function_names_from_file(file_path)


ignored_function_names = [
    "__init__",
    "_normalize_public_auth_route",
    "_get_model_from_request_context",
    "_resolve_router_settings_model_group_alias",
    "_get_model_names_for_budget_checks",
    "_budget_fallback_target_model",
    "team_models",
    "_raw_cache",
    "_token_team_models",
    "model_max_budget",
    "end_user_model_max_budget",
    "_token_model_budget_fields",
    "fallbacks",
    "_router_fallbacks",
    "_check_team_model_budget",
    "_is_within_every_model_budget",
    "_is_fallback_model_authorized",
    "_is_within_all_fallback_model_budgets",
    "_check_team_member_model_budget_for_model",
    "_try_team_member_model_budget_fallback",
    "_get_bearer_token_or_received_api_key",
    "_should_route_jwt_to_oauth2_override",
    "get_global_proxy_spend",
    "get_rbac_role",
    "_claim_identifies_user",
    "_reusable_key_hash_for_user",
    "_lookup_jwt_mapping_token_hash",
    "_record_unparsable_body_failure",
    "_refresh_session_token_grants",
    "_resolve_object_permission_for_unresolvable_team",
    "validate_resolved_virtual_key",
    "_safe_fetch",
    "_team_obj_from_token",
    "_token_can_vouch_for_team",
    "_inherit_org_identity",
    "is_no_auth_dev_mode",
    "_noop_none",
    "_apply_key_end_user_default_budget_to_token",
    "_spend_counter_redis_cache",
    "_prefetch_referenced_auth_objects",
    "authorize_internal_virtual_key",
]


def main():
    # router_file = [
    #     "./litellm/user_api_key_auth.py",
    # ]
    router_file = [
        "../../litellm/proxy/auth/user_api_key_auth.py",
    ]
    # router_file = [
    #     "../../litellm/router.py",
    #     "../../litellm/router_utils/pattern_match_deployments.py",
    #     "../../litellm/router_utils/batch_utils.py",
    # ]  ## LOCAL TESTING
    # tests_dir = (
    #     "./tests/"  # Update this path if your tests directory is located elsewhere
    # )
    tests_dir = "../../tests/"  # LOCAL TESTING

    router_functions = []
    for file in router_file:
        router_functions.extend(get_functions_from_router(file))
    print("router_functions: ", router_functions)
    called_functions_in_tests = get_all_functions_called_in_tests(tests_dir)
    untested_functions = [
        fn for fn in router_functions if fn not in called_functions_in_tests
    ]

    if untested_functions:
        all_untested_functions = []
        for func in untested_functions:
            if func not in ignored_function_names:
                all_untested_functions.append(func)
        untested_perc = (len(all_untested_functions)) / len(router_functions)
        print("untested_perc: ", untested_perc)
        if untested_perc > 0:
            print("The following functions in user_api_key_auth.py are not tested:")
            raise Exception(
                f"{untested_perc * 100:.2f}% of functions in user_api_key_auth.py are not tested: {all_untested_functions}"
            )
    else:
        print("All functions in user_api_key_auth.py are covered by tests.")


if __name__ == "__main__":
    main()
