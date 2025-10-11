export interface SpendMetrics {
  spend: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  api_requests: number;
  successful_requests: number;
  failed_requests: number;
  cache_read_input_tokens: number;
  cache_creation_input_tokens: number;
}

export interface DailyData {
  date: string;
  metrics: SpendMetrics;
  breakdown: BreakdownMetrics;
}

export interface BreakdownMetrics {
  models: { [key: string]: MetricWithMetadata };
  model_groups: { [key: string]: MetricWithMetadata };
  mcp_servers: { [key: string]: MetricWithMetadata };
  providers: { [key: string]: MetricWithMetadata };
  api_keys: { [key: string]: KeyMetricWithMetadata };
  entities: { [key: string]: MetricWithMetadata };
}

export interface MetricWithMetadata {
  metrics: SpendMetrics;
  metadata: object;
  api_key_breakdown: { [key: string]: KeyMetricWithMetadata };
}

export interface KeyMetricWithMetadata {
  metrics: SpendMetrics;
  metadata: KeyMetadata;
}

export interface KeyMetadata {
  key_alias: string | null;
  team_id: string | null;
  tags?: { tag: string; usage: number }[];
}

export interface TopApiKeyData {
  api_key: string;
  key_alias: string | null;
  team_id: string | null;
  spend: number;
  requests: number;
  tokens: number;
}

export interface ModelActivityData {
  label: string;
  total_requests: number;
  total_successful_requests: number;
  total_failed_requests: number;
  total_cache_read_input_tokens: number;
  total_cache_creation_input_tokens: number;
  total_tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_spend: number;
  top_api_keys: TopApiKeyData[];
  daily_data: {
    date: string;
    metrics: {
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
      api_requests: number;
      spend: number;
      successful_requests: number;
      failed_requests: number;
      cache_read_input_tokens: number;
      cache_creation_input_tokens: number;
    };
  }[];
}

export interface EntityMetadata {
  alias: string;
  id: string;
}

export interface EntityMetricWithMetadata {
  metrics: SpendMetrics;
  metadata: EntityMetadata;
}

export interface TagUsage {
  tag: string;
  usage: number;
}
