export interface SubscriptionFee {
  subscription_account_id: string;
  custom_llm_provider: string;
  account_id: string;
  label: string | null;
  monthly_fee: number;
  currency: string;
  billing_period_start: string;
  created_at: string;
  updated_at: string;
  created_by: string;
  updated_by: string;
}

export interface BillingPeriod {
  start: string;
  end: string;
}

export interface SubscriptionAccountUsage {
  custom_llm_provider: string;
  account_id: string | null;
  deployments: string[];
  fee: SubscriptionFee | null;
  billing_periods: BillingPeriod[];
  fixed_cost: number | null;
}

export interface SubscriptionUsageResponse {
  subscription_providers: string[];
  accounts: SubscriptionAccountUsage[];
}

export interface SubscriptionAccountCreateRequest {
  custom_llm_provider: string;
  account_id: string;
  monthly_fee: number;
  currency: string;
  billing_period_start: string;
  label: string | null;
}

export interface SubscriptionAccountUpdateRequest {
  subscription_account_id: string;
  monthly_fee?: number;
  currency?: string;
  billing_period_start?: string;
  label?: string | null;
}

export interface SubscriptionAccountDeleteRequest {
  subscription_account_id: string;
}

export interface SubscriptionFeeFormValues {
  monthly_fee: number;
  currency: string;
  billing_period_start: string;
  label: string | null;
}

export const isSubscriptionCovered = (metadata: object | undefined): boolean =>
  metadata !== undefined && "subscription_covered" in metadata && metadata.subscription_covered === true;
