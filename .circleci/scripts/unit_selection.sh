#!/usr/bin/env bash
set -euo pipefail

flag="${1:?usage: unit_selection.sh <codecov flag>}"

legacy_flags=(
  caching-local
  enterprise-package
  enterprise-routing
  integrations
  llm-other-providers
  llm-vertex-ai
  mcp-integration
  misc
  proxy-db-auth-checks
  proxy-db-budgets
  proxy-db-custom-logging
  proxy-db-db-and-spend
  proxy-db-endpoints-and-responses
  proxy-db-guardrails-hooks
  proxy-db-jwt-and-keys
  proxy-db-key-generation
  proxy-db-logging-misc
  proxy-db-proxy-runtime
  proxy-db-proxy-server-core
  proxy-db-proxy-utils
  proxy-extras
  proxy-infra
  responses-caching-types
)

legacy_paths() {
  case "$1" in
    caching-local) echo tests/unit/caching ;;
    enterprise-package)
      echo tests/unit/enterprise/integrations
      echo tests/unit/enterprise/proxy/auth
      echo tests/unit/enterprise/proxy/guardrails
      echo tests/unit/enterprise/proxy/hooks
      echo tests/unit/enterprise/proxy/management_endpoints
      echo tests/unit/enterprise/proxy/test_audit_logging_endpoints.py
      echo tests/unit/enterprise/enterprise_callbacks/test_prometheus_logging_callbacks.py ;;
    enterprise-routing)
      echo tests/unit/google_genai
      echo tests/unit/enterprise/enterprise_callbacks/send_emails
      echo tests/unit/enterprise/proxy/test_afile_retrieve_returns_unified_id.py
      echo tests/unit/enterprise/proxy/test_batch_retrieve_input_file_id.py
      echo tests/unit/enterprise/proxy/test_batch_retrieve_registers_missing_output_file_id.py
      echo tests/unit/enterprise/proxy/test_batch_retrieve_returns_unified_input_file_id.py
      echo tests/unit/enterprise/proxy/test_batch_update_db_managed_output_file_id.py
      echo tests/unit/enterprise/proxy/test_deleted_file_returns_403_not_404.py
      echo tests/unit/enterprise/proxy/test_enterprise_routes.py
      echo tests/unit/enterprise/proxy/test_file_deletion_blocking.py
      echo tests/unit/enterprise/proxy/test_managed_files_access_check.py
      echo tests/unit/enterprise/proxy/test_managed_files_hook.py ;;
    integrations) echo tests/unit/integrations ;;
    llm-other-providers) find tests/unit/llms -name 'test_*.py' -not -path 'tests/unit/llms/vertex_ai/*' ;;
    llm-vertex-ai) echo tests/unit/llms/vertex_ai ;;
    mcp-integration)
      echo tests/unit/experimental_mcp_client
      echo tests/unit/proxy/_experimental/mcp_server
      echo tests/unit/responses/mcp
      echo tests/mcp_tests/test_proxy_mcp_e2e.py ;;
    misc)
      find tests/unit -maxdepth 1 -name 'test_*.py'
      echo tests/unit/test_router
      echo tests/unit/a2a_protocol
      echo tests/unit/batches
      echo tests/unit/chat_completions
      echo tests/unit/completion_extras
      echo tests/unit/containers
      echo tests/unit/embeddings
      echo tests/unit/endpoints
      echo tests/unit/files
      echo tests/unit/images
      echo tests/unit/interactions
      echo tests/unit/messages
      echo tests/unit/rag
      echo tests/unit/rerank_api
      echo tests/unit/secret_managers
      echo tests/unit/vector_stores
      echo tests/unit/videos ;;
    proxy-db-auth-checks)
      echo tests/unit/proxy/auth/test_auth_checks.py
      echo tests/unit/proxy/auth/test_user_api_key_auth.py
      echo tests/unit/proxy/test_deprecated_key_grace_period.py ;;
    proxy-db-budgets)
      echo tests/unit/proxy/auth/test_default_end_user_budget_simple.py
      echo tests/unit/proxy/hooks/test_unit_test_max_model_budget_limiter.py
      echo tests/unit/proxy/test_zero_cost_model_budget_bypass.py ;;
    proxy-db-custom-logging)
      echo tests/unit/proxy/test_custom_callback_input.py
      echo tests/unit/proxy/test_custom_logger_s3_gcs.py ;;
    proxy-db-db-and-spend)
      echo tests/unit/proxy/common_utils/test_proxy_encrypt_decrypt.py
      echo tests/unit/proxy/db/db_transaction_queue/test_e2e_pod_lock_manager.py
      echo tests/unit/proxy/db/test_update_daily_tag_spend.py
      echo tests/unit/proxy/test_db_schema_changes.py
      echo tests/unit/proxy/test_prisma_client_backoff_retry.py
      echo tests/unit/proxy/test_update_spend.py
      echo tests/unit/skills/test_skills_db.py ;;
    proxy-db-endpoints-and-responses)
      echo tests/unit/proxy/auth/test_models_fallback_endpoint.py
      echo tests/unit/proxy/common_utils/test_check_batch_cost.py
      echo tests/unit/proxy/common_utils/test_check_responses_cost.py
      echo tests/unit/proxy/common_utils/test_realtime_cache.py
      echo tests/unit/proxy/google_endpoints/test_gemini_agents_endpoints.py
      echo tests/unit/proxy/google_endpoints/test_google_endpoint_routing.py
      echo tests/unit/proxy/google_endpoints/test_google_gemini_proxy_request.py
      echo tests/unit/proxy/public_endpoints/test_blog_posts_endpoint.py
      echo tests/unit/proxy/response_polling/test_response_polling_handler.py
      echo tests/unit/proxy/test_custom_tokenizer_bug.py
      echo tests/unit/proxy/test_get_favicon.py
      echo tests/unit/proxy/test_get_image.py
      echo tests/unit/proxy/test_prompt_test_endpoint.py
      echo tests/unit/proxy/test_reducto_ocr_route.py
      echo tests/unit/proxy/test_response_polling_pre_call_checks.py
      echo tests/unit/proxy/test_ui_path_detection.py ;;
    proxy-db-guardrails-hooks)
      echo tests/unit/proxy/hooks/test_banned_keyword_list.py
      echo tests/unit/proxy/test_proxy_setting_guardrails.py
      echo tests/unit/proxy/test_unit_test_proxy_hooks.py ;;
    proxy-db-jwt-and-keys)
      echo tests/unit/proxy/auth/test_jwt.py
      echo tests/unit/proxy/management_endpoints/test_jwt_key_mapping.py
      echo tests/unit/proxy/test_proxy_custom_auth.py ;;
    proxy-db-key-generation) echo tests/unit/proxy/management_endpoints/test_key_generate_prisma.py ;;
    proxy-db-logging-misc)
      echo tests/unit/proxy/management_helpers/test_audit_logs_proxy.py
      echo tests/unit/proxy/spend_tracking/test_search_api_logging.py
      echo tests/unit/proxy/test_proxy_reject_logging.py ;;
    proxy-db-proxy-runtime)
      echo tests/unit/proxy/auth/test_multipart_bypass_repro.py
      echo tests/unit/proxy/auth/test_proxy_routes.py
      echo tests/unit/proxy/middleware/test_request_size_limit_middleware.py
      echo tests/unit/proxy/test_proxy_config_unit_test.py
      echo tests/unit/proxy/test_proxy_token_counter.py
      echo tests/unit/proxy/test_server_root_path.py ;;
    proxy-db-proxy-server-core)
      echo tests/unit/proxy/test_aproxy_startup.py
      echo tests/unit/proxy/test_proxy_server.py ;;
    proxy-db-proxy-utils) echo tests/unit/proxy/test_proxy_utils.py ;;
    proxy-extras) echo tests/unit/litellm_proxy_extras ;;
    proxy-infra) echo tests/unit/gateway ;;
    responses-caching-types) echo tests/unit/types ;;
    *) echo "unit_selection.sh: unknown flag $1" >&2; exit 1 ;;
  esac
}

expand() {
  while read -r path; do
    if [ -d "$path" ]; then
      find "$path" -name 'test_*.py'
    elif [ -f "$path" ]; then
      echo "$path"
    else
      echo "unit_selection.sh: $path does not exist" >&2
      exit 1
    fi
  done
}

if [ "$flag" = unit ]; then
  comm -23 \
    <(find tests/unit -name 'test_*.py' | sort) \
    <(for legacy in "${legacy_flags[@]}"; do legacy_paths "$legacy"; done | expand | sort)
  exit 0
fi

legacy_paths "$flag" | expand | sort
