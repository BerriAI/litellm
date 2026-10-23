import ast
import os


def get_function_names_from_file(file_path):
    """
    Extracts all function names from a given Python file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
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
    inside the unit test directories,
    specifically in files containing the word 'router'.
    """
    called_functions = set()
    test_dirs = ["unit"]

    for test_dir in test_dirs:
        dir_path = os.path.join(base_dir, test_dir)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist.")
            continue

        print("dir_path: ", dir_path)
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".py") and (
                    "router" in os.path.relpath(file_path, dir_path).lower() or test_dir == "unit"
                ):
                    print("file: ", file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            tree = ast.parse(f.read())
                        except SyntaxError:
                            print(f"Warning: Syntax error in file {file_path}")
                            continue
                    if file == "test_router_validate_fallbacks.py":
                        print(f"tree: {tree}")
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                            called_functions.add(node.func.id)
                        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                            called_functions.add(node.func.attr)

    return called_functions


def get_functions_from_router(file_path):
    """
    Extracts all functions defined in router.py.
    """
    return get_function_names_from_file(file_path)


ignored_function_names = [
    "_acancel_batch",
    "_acreate_batch",
    "_acreate_file",
    "__init__",
    "avector_store_create",  # Tested via proxy vector_store_endpoints (files lack "router" in name)
    "_override_vector_store_methods_for_router",  # No-op placeholder, called during Router init
    "_merge_tools_from_deployment",  # Tested indirectly via _update_kwargs_with_deployment (test files lack "router" in name)
    "_invalidate_access_groups_cache",  # Tested indirectly via set_model_list, upsert_model etc. (test files lack "router" in name)
    "has_buffered_provider_output",  # Property, so its reads in test_router.py are never an ast.Call
    "chunks",  # Property on FallbackAwareAnthropicMessagesStream, so its reads in tests are never an ast.Call
    "messages",  # Property on FallbackAwareAnthropicMessagesStream, so its reads in tests are never an ast.Call
    "model",  # Property on FallbackAwareAnthropicMessagesStream, so its reads in tests are never an ast.Call
    "_routing_groups",  # Property getter and setter reads are never ast.Call nodes
    "_request_header",  # Tested through Claude Code session routing in test_router.py
    "_claude_code_session_router_cache_key",  # Tested through Claude Code session routing in test_router.py
    "_delete_claude_code_session_router_binding",  # Tested through Redis cleanup failure in test_router.py
    "_resolve_claude_code_session_router",  # Tested through Claude Code session routing in test_router.py
    "_get_claude_code_session_router_binding",  # Tested through the two-worker session routing test in test_router.py
    "_apply_updated_routing_strategy_args",  # Tested via update_settings in test_lowest_latency.py (file lacks "router" in name)
    "_wait_for_scheduler_turn",  # Tested through prioritized acompletion and atext_completion in test_router.py
    "arm_routing_read_prefetch",  # Tested in tests/unit/caching/test_request_redis_batch_pre_call.py (file lacks "router" in name)
    "_deployment_params_base_model",  # Tested through log_retry in test_router_helper_utils.py
    "_configured_model_info",  # Tested through get_configured_service_tiers in test_router.py
    "_routable_deployments",  # Tested through get_configured_service_tiers and get_routable_upstream_model in test_router.py
    "_async_get_available_deployment",  # Body of the `route {model}` phase wrapper, exercised through async_get_available_deployment in test_router.py
    "_async_get_available_deployment_for_pass_through",  # Same, through async_get_available_deployment_for_pass_through in test_router.py
    "_embedding",
    "_aembedding",
    "_stream_pre_content_error",  # Tested through the non-retriable retry error tests in test_router.py
    "_narrow_to_video_deployment",  # Tested through video_status and avideo_status polls in test_router_provider_endpoints.py
    "_video_call_pin",  # Tested through the video_status pin tests and the chat video_id test in test_router_provider_endpoints.py
    "_deployment_num_retries",  # Tested through the deployment num_retries mid-stream budget test in test_router.py
    "_request_fallback_list",  # Tested through every mid-stream retry test in test_router.py
    "_request_model_group",  # Tested through test_anthropic_messages_retry_budget_precedence_direct_call
    "_mid_stream_retry_trigger",  # Tested through the retry policy mid-stream budget test in test_router.py
    "_anthropic_messages_group_retry_policy",  # Tested through the retry budget precedence test in test_router.py
    "_anthropic_messages_resolved_retry_policy",  # Tested through the malformed retry policy tests in test_router.py
    "_anthropic_messages_plain_retry_budget",  # Tested through the retry budget precedence test in test_router.py
    "_stream_should_retry",  # Tested through every mid-stream retry test in test_router.py
    "_astream_retry_same_group",  # Tested through the dropped-before-content retry tests in test_router.py
    "_aanthropic_messages_yield_recovered",  # Tested through every mid-stream retry and fallback test in test_router.py
    "_anthropic_messages_policy_retries",  # Tested through the retry budget precedence test in test_router.py
    "_get_wildcard_deployments",  # Tested through the get_model_list_of_routed_group wildcard test in test_router.py
    "_prompt_management_acompletion",  # Tested through the scheduled prompt-management model test in test_router.py
    "_request_deployment_params",  # Tested through the team-route and deployment-id drop_params priority tests in test_router.py
    "split_request_priority",  # Tested through the acompletion, atext_completion and queue route priority tests
    "_pinned_deployment",  # Tested through the deployment-id scheduler test in test_router.py
    "_is_fallback_hop",  # Tested through the order fallback hop tests in test_router_order_fallback.py
    "_deployment_that_just_failed",  # Tested through the same-boundary hop test in test_router_order_fallback.py
    "_merged_tools",  # Tested through the _update_kwargs_with_deployment tools merge tests in test_router.py
    "_drops_malformed_request_tools",  # Tested through the _update_kwargs_with_deployment tools merge tests in test_router.py
    "_collect_ptu_share_teams",  # Cached behind team_ptu_ceiling, tested through the live model list test in test_router.py
    "_resolve_team_ptu_ceiling",  # Cached behind team_ptu_ceiling, tested through the live model list test in test_router.py
    "_register_continuation_prefill_check",  # Tested through the continuation tests in test_router.py
    "_set_mid_stream_fallback_continuation",  # Tested through the update_settings continuation test in test_router.py
    "_is_flag_enabled",  # Tested through the string-flag continuation tests in test_router.py
    "_params_sent_to_deployment",  # Tested through the configured-constraint continuation tests in test_router.py
    "_names_deployment_that_cannot_continue",  # Tested through the deployment-id continuation tests in test_router.py
    "_set_retry_policy",  # Tested through update_settings in test_router_retry_policy_update.py
    "_resolve_pre_call_check_input_tokens",  # Tested through the token-count failure pre-call check tests in test_router.py
    "_params_neither_dropped_nor_allowed",  # Tested through the allowed-or-dropped param pre-call check tests in test_router.py
    "_deployment_rejects_params",  # Tested through the unsupported-param pre-call check tests in test_router.py
    "_deployment_lacks_params",  # Tested through the unsupported-param pre-call check tests in test_router.py
    "_pre_call_check_requested_chat_params",  # Tested through the unsupported-param pre-call check tests in test_router.py
    "_deployment_supports_param",  # Tested through the deployment-capability pre-call check tests in test_router.py
    "_capability_model_and_provider",  # Tested through the authenticating-provider pre-call check test in test_router.py
    "_exact_base_model_max_input_tokens",  # Tested through get_router_model_info in test_router_model_cost_isolation.py
    "_base_model_widened_window",  # Tested through get_router_model_info in test_router_model_cost_isolation.py
    "_provider_model_id",  # Tested through get_router_model_info in test_router_model_cost_isolation.py
    "_names_route_model",  # Tested through get_router_model_info in test_router_model_cost_isolation.py
    "_declared_base_model",  # Tested through get_router_model_info in test_router_model_cost_isolation.py
    "async_admit_deployment_turn",  # Bound as the WebSocket turn_admission partial, tested through it in test_router_helper_utils.py
    "_deployment_as_configured_now",  # Read by async_admit_deployment_turn, tested through turn_admission in test_router_helper_utils.py
    "_admit_turn_through_routing_strategy",  # Run by async_admit_deployment_turn, tested through turn_admission in test_router_helper_utils.py
    "_callback_selectors",  # Run by discard, tested through the discard tests in test_router_helper_utils.py
    "bind_to_request_logging",  # Run by the proxy user_config route, tested through the user_config tests in test_route_llm_request.py
    "serve_through_request_logging",  # Run by the proxy user_config route, tested through the user_config tests in test_route_llm_request.py
    "_request_adaptive_router_hooks",  # Run by serve_through_request_logging, tested through the user_config adaptive test
    "_release_global_callbacks",  # Run by discard, tested through the discard tests in test_router_helper_utils.py
    "_routing_callbacks",  # Read by the pre-call checks, tested through the user_config limit tests in test_route_llm_request.py
    "_request_policies_not_set_globally",  # Read by bind_to_request_logging, tested through the user_config budget test in test_route_llm_request.py
    "_optional_loggers",  # Read by discard and serve_through_request_logging, tested through the budget limiter discard test
    "retire_routing_state",  # Run by discard, tested through the budget limiter discard test
    "_slack_alerting_loggers",  # Run by discard, tested through the Slack alerting discard test in test_router_helper_utils.py
    "_adaptive_router_hooks",  # Run by discard, tested through the adaptive router discard tests in test_router_helper_utils.py
    "_global_callback_candidates",  # Run by discard, tested through the survivor discard tests in test_router_helper_utils.py
    "without_deployments_that_cannot_batch",  # Tested through the Azure and OpenAI mixed-group acreate_batch tests in test_router.py
    "batch_create_routing",  # Tested through the mixed-group acreate_batch tests in test_router.py
    "is_batch_create_routing",  # Tested through the chat-after-batch-create test in test_router.py
    "is_batch_capable",  # Tested through the mixed-group acreate_batch and model group info tests in test_router.py
    "deployments_the_caller_may_use",  # Tested through the usage-based-routing team batch create test in test_router.py
    "deployment_supported_endpoints",  # Tested through the model group info batch endpoint tests in test_router.py
    "_endpoint_paths",  # Read by deployment_supported_endpoints, tested through the malformed endpoint list group info test in test_router.py
    "_deployment_field",  # Read by is_batch_capable, tested through the mixed-group tests in test_router.py
    "_deployment_provider",  # Read by without_deployments_that_cannot_batch, tested through the mixed-group tests in test_router.py
    "_cannot_run_batches",  # Tested through the Azure and OpenAI mixed-group acreate_batch tests in test_router.py
]


def main():
    router_file = [
        "./litellm/router.py",
        "./litellm/router_utils/batch_utils.py",
        "./litellm/router_utils/pattern_match_deployments.py",
    ]
    # router_file = [
    #     "../../litellm/router.py",
    #     "../../litellm/router_utils/pattern_match_deployments.py",
    #     "../../litellm/router_utils/batch_utils.py",
    # ]  ## LOCAL TESTING
    tests_dir = "./tests/"  # Update this path if your tests directory is located elsewhere
    # tests_dir = "../../tests/"  # LOCAL TESTING

    router_functions = []
    for file in router_file:
        router_functions.extend(get_functions_from_router(file))
    print("router_functions: ", router_functions)
    called_functions_in_tests = get_all_functions_called_in_tests(tests_dir)
    untested_functions = [fn for fn in router_functions if fn not in called_functions_in_tests]

    if untested_functions:
        all_untested_functions = []
        for func in untested_functions:
            if func not in ignored_function_names:
                all_untested_functions.append(func)
        untested_perc = (len(all_untested_functions)) / len(router_functions)
        print("untested_perc: ", untested_perc)
        if untested_perc > 0:
            print("The following functions in router.py are not tested:")
            raise Exception(
                f"{untested_perc * 100:.2f}% of functions in router.py are not tested: {all_untested_functions}"
            )
    else:
        print("All functions in router.py are covered by tests.")


if __name__ == "__main__":
    main()
