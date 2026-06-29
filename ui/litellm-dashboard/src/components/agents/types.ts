export interface AgentAttachedKey {
  token: string;
  key_alias?: string | null;
  key_name?: string | null;
}

export interface AgentObjectPermission {
  mcp_servers?: string[];
  mcp_access_groups?: string[];
  mcp_tool_permissions?: Record<string, string[]>;
}

export interface Agent {
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
