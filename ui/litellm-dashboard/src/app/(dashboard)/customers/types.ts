export interface CustomerObjectPermission {
  mcp_servers?: string[];
  mcp_access_groups?: string[];
  mcp_tool_permissions?: Record<string, string[]>;
  agents?: string[];
  agent_access_groups?: string[];
}

export interface Customer {
  user_id: string;
  alias: string | null;
  spend: number;
  allowed_model_region: string | null;
  default_model: string | null;
  budget_id: string | null;
  blocked: boolean;
  max_budget?: number | null;
  budget_duration?: string | null;
  litellm_budget_table?: {
    max_budget: number | null;
    budget_duration: string | null;
  } | null;
  object_permission?: CustomerObjectPermission | null;
}

export interface NewCustomerData {
  user_id: string;
  alias?: string;
  max_budget?: string;
  budget_id?: string;
  default_model?: string;
  allowed_model_region?: string;
  budget_duration?: string;
  metadata?: string;
  object_permission?: CustomerObjectPermission;
}
