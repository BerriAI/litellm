export interface UserInfo {
  user_id: string;
  user_email: string;
  user_role: string;
  spend: number;
  max_budget: number | null;
  key_count: number;
  created_at: string;
  updated_at: string;
} 