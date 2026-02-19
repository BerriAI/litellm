export interface GuardrailMetrics {
  totalRequests: number;
  successCount: number;
  intervenedCount: number;
  failedCount: number;
  notRunCount: number;
  failRate: number;
  avgLatencyMs: number;
}

export interface GuardrailSummary {
  guardrail_name: string;
  provider: string;
  total_requests: number;
  fail_rate: number;
  avg_latency_ms: number;
}

export interface GuardrailMetricsResponse {
  results: GuardrailSummary[];
  metadata: {
    page: number;
    total_pages: number;
    has_more: boolean;
    total_count: number;
  };
}

export interface GuardrailDailyMetrics {
  date: string;
  total_requests: number;
  intervened_count: number;
  success_count: number;
  fail_rate: number;
  avg_latency_ms: number;
}

export interface GuardrailDetailMetrics {
  requests_evaluated: number;
  fail_rate: number;
  avg_latency_ms: number;
  blocked_count: number;
  daily_metrics: GuardrailDailyMetrics[];
}

export interface GuardrailLogEntry {
  request_id: string;
  timestamp: string;
  model: string;
  status: "blocked" | "passed";
  guardrail_response?: any;
  request_content?: string;
  latency_ms: number;
}

export interface GuardrailLogsResponse {
  logs: GuardrailLogEntry[];
  total_count: number;
  page: number;
  page_size: number;
}
