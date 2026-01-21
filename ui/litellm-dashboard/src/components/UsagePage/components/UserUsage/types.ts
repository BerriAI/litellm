/**
 * TypeScript types for Admin User Usage components
 */

export interface UserUsageSummary {
  total_users: number;
  total_spend: number;
  total_requests: number;
  total_successful_requests: number;
  total_failed_requests: number;
  total_tokens: number;
  avg_spend_per_user: number;
  power_users_count: number;
  low_users_count: number;
}

export interface UserMetrics {
  user_id: string;
  user_email: string;
  spend: number;
  requests: number;
  successful_requests: number;
  failed_requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  tokens: number;
  days_active: number;
  first_request_date: string;
  last_request_date: string;
  tags: string[];
  models_used: string[];
}

export interface UserUsagePagination {
  page: number;
  page_size: number;
  total_count: number;
  total_pages: number;
}

export interface UserUsageResponse {
  summary: UserUsageSummary;
  top_users: UserMetrics[];
  users: UserMetrics[];
  pagination: UserUsagePagination;
}

export interface UserUsageFiltersState {
  tagFilters: string[];
  minSpend: number | null;
  maxSpend: number | null;
  sortBy: "spend" | "requests" | "tokens";
  sortOrder: "asc" | "desc";
}
