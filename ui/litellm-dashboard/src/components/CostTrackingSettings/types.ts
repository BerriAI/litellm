export interface CostTrackingSettingsProps {
  userID: string | null;
  userRole: string | null;
  accessToken: string | null;
}

export interface DiscountConfig {
  [provider: string]: number;
}

export interface CostDiscountResponse {
  values: DiscountConfig;
}

export interface MarginConfig {
  [provider: string]: number | { percentage?: number; fixed_amount?: number };
}

export interface CostMarginResponse {
  values: MarginConfig;
}

export interface CostEstimateRequest {
  model: string;
  input_tokens: number;
  output_tokens: number;
  num_requests_per_day?: number | null;
  num_requests_per_month?: number | null;
}

export interface CostEstimateResponse {
  model: string;
  input_tokens: number;
  output_tokens: number;
  num_requests_per_day: number | null;
  num_requests_per_month: number | null;
  cost_per_request: number;
  input_cost_per_request: number;
  output_cost_per_request: number;
  margin_cost_per_request: number;
  daily_cost: number | null;
  daily_input_cost: number | null;
  daily_output_cost: number | null;
  daily_margin_cost: number | null;
  monthly_cost: number | null;
  monthly_input_cost: number | null;
  monthly_output_cost: number | null;
  monthly_margin_cost: number | null;
  input_cost_per_token: number | null;
  output_cost_per_token: number | null;
  provider: string | null;
}

