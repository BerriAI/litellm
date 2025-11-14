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

