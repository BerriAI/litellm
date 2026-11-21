INSERT INTO "LiteLLM_SpendLogs" (
    "request_id", "call_type", "api_key", "spend", "total_tokens", "prompt_tokens",
    "completion_tokens", "startTime", "endTime", "request_duration_ms", "completionStartTime",
    "model", "model_id", "model_group", "custom_llm_provider", "api_base", "user", "metadata",
    "cache_hit", "cache_key", "request_tags", "team_id", "organization_id", "end_user",
    "requester_ip_address", "messages", "response", "session_id", "status",
    "mcp_namespaced_tool_name", "billing_agent_id", "agent_id", "proxy_server_request",
    "litellm_call_id", "created_at", "updated_at"
)
SELECT
    "request_id", "call_type", "api_key", "spend", "total_tokens", "prompt_tokens",
    "completion_tokens", "startTime", "endTime", "request_duration_ms", "completionStartTime",
    "model", "model_id", "model_group", "custom_llm_provider", "api_base", "user", "metadata",
    "cache_hit", "cache_key", "request_tags", "team_id", "organization_id", "end_user",
    "requester_ip_address", "messages", "response", "session_id", "status",
    "mcp_namespaced_tool_name", "billing_agent_id", "agent_id", "proxy_server_request",
    "litellm_call_id", "created_at", "updated_at"
FROM jsonb_populate_recordset(NULL::"LiteLLM_SpendLogs", $1::jsonb)
ON CONFLICT DO NOTHING RETURNING request_id
