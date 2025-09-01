export interface VirtualKey {
  key_id: string;
  key_name?: string;
  team_id?: string;
  models?: string[];
  spend?: number;
  max_budget?: number;
  max_parallel_requests?: number;
  metadata?: Record<string, any>;
  expires?: string;
  created_at?: string;
  updated_at?: string;
}
