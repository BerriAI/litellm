import type { components } from "@/lib/http/schema";

export interface AgentAttachedKey {
  token: string;
  key_alias?: string | null;
  key_name?: string | null;
}

export type AgentObjectPermission = components["schemas"]["AgentObjectPermission"];

export interface Agent {
  identity?: components["schemas"]["AgentIdentityBinding"] | null;
  identity_managed?: boolean;
  enabled?: boolean;
  execution_mode?: components["schemas"]["AgentResponse"]["execution_mode"];
  litellm_budget_table?: components["schemas"]["AgentBudgetState"] | null;
  jwt_auth_configured?: boolean;
  agent_id: string;
  agent_name: string;
  litellm_params: {
    model: string;
    [key: string]: any;
  };
  agent_card_params?: {
    description?: string;
    url?: string;
    [key: string]: any;
  };
  object_permission?: AgentObjectPermission;
  access_group_ids?: string[] | null;
  keys?: AgentAttachedKey[] | null;
  spend?: number;
  tpm_limit?: number | null;
  rpm_limit?: number | null;
  session_tpm_limit?: number | null;
  session_rpm_limit?: number | null;
  created_at?: string;
  updated_at?: string;
  created_by?: string;
  updated_by?: string;
}

export interface AgentsResponse {
  agents: Agent[];
}
