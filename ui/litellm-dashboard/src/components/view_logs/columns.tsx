/** API sort field mapping for /spend/logs/ui endpoint */
export const LOGS_SORT_FIELD_MAP = {
  startTime: "startTime",
  spend: "spend",
  total_tokens: "total_tokens",
  request_duration_ms: "request_duration_ms",
  model: "model",
  ttft_ms: "ttft_ms",
} as const;

export type LogsSortField = keyof typeof LOGS_SORT_FIELD_MAP;

export type LogEntry = {
  request_id: string;
  api_key: string;
  team_id: string;
  model: string;
  model_id: string;
  api_base?: string;
  call_type: string;
  spend: number;
  total_tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
  startTime: string;
  endTime: string;
  user?: string;
  end_user?: string;
  custom_llm_provider?: string;
  metadata?: Record<string, any>;
  cache_hit: string;
  cache_key?: string;
  request_tags?: Record<string, any>;
  requester_ip_address?: string;
  messages: string | any[] | Record<string, any>;
  response: string | any[] | Record<string, any>;
  proxy_server_request?: string | any[] | Record<string, any>;
  session_id?: string;
  status?: string;
  completionStartTime?: string;
  request_duration_ms?: number;
  session_total_count?: number;
  session_total_spend?: number;
  mcp_tool_call_count?: number;
  mcp_tool_call_spend?: number;
  session_llm_count?: number;
  session_mcp_count?: number;
  session_agent_count?: number;
};
