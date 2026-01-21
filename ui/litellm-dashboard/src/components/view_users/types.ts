export interface UserInfo {
  user_id: string;
  user_email: string;
  user_alias: string | null;
  user_role: string;
  spend: number;
  max_budget: number | null;
  models: string[];
  key_count: number;
  created_at: string;
  updated_at: string;
  sso_user_id: string | null;
  budget_duration: string | null;
}
