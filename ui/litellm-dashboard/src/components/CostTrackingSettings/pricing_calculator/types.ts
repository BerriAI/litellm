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

