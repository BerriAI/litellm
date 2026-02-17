export interface PricingCalculatorProps {
  accessToken: string | null;
  models: string[];
}

export interface PricingFormValues {
  model: string;
  input_tokens: number;
  output_tokens: number;
  num_requests_per_day?: number;
  num_requests_per_month?: number;
}

export interface ModelEntry {
  id: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  num_requests_per_day?: number;
  num_requests_per_month?: number;
}

export interface MultiModelResult {
  entries: Array<{
    entry: ModelEntry;
    result: import("../types").CostEstimateResponse | null;
    loading: boolean;
    error: string | null;
  }>;
  totals: {
    cost_per_request: number;
    daily_cost: number | null;
    monthly_cost: number | null;
    margin_per_request: number;
    daily_margin: number | null;
    monthly_margin: number | null;
  };
}

