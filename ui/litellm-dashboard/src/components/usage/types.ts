export interface SpendMetrics {
  spend: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  api_requests: number;
  successful_requests: number;
  failed_requests: number;
}

export interface DailyData {
  date: string;
  metrics: SpendMetrics;
  breakdown: BreakdownMetrics;
}

export interface BreakdownMetrics {
  models: { [key: string]: MetricWithMetadata };
  providers: { [key: string]: MetricWithMetadata };
  api_keys: { [key: string]: KeyMetricWithMetadata };
}

export interface MetricWithMetadata {
  metrics: SpendMetrics;
  metadata: object;
}

export interface KeyMetricWithMetadata {
  metrics: SpendMetrics;
  metadata: {
    key_alias: string | null;
  };
}

export interface ModelActivityData {
  total_requests: number;
  total_successful_requests: number;
  total_failed_requests: number;
  total_tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_spend: number;
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
    };
  }[];
}

export interface KeyMetadata {
  key_alias: string | null;
}

export interface KeyMetricWithMetadata {
  metrics: SpendMetrics;
  metadata: KeyMetadata;
}

export interface MetricWithMetadata {
  metrics: SpendMetrics;
  metadata: object;
}

export interface BreakdownMetrics {
  models: { [key: string]: MetricWithMetadata };
  providers: { [key: string]: MetricWithMetadata };
  api_keys: { [key: string]: KeyMetricWithMetadata };
}
