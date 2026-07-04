import type { LogEntry } from "@/components/view_logs/columns";

export type AdminObservabilityRow = LogEntry;

export interface AdminObservabilityFilters {
  user_id?: string;
  model?: string;
  status_filter?: string;
  request_id?: string;
}
