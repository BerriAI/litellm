import type { LogEntry } from "@/components/view_logs/columns";

export type AdminObservabilityRow = LogEntry & {
  /** Top-level MCP tool name returned by /spend/logs/ui but not yet in the shared LogEntry type. */
  mcp_namespaced_tool_name?: string | null;
};

export interface AdminObservabilityFilters {
  user_id?: string;
  model?: string;
  status_filter?: string;
  request_id?: string;
}
