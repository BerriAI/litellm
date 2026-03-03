export interface AgentKeyInfo {
  key_alias?: string;
  token_prefix?: string;
  has_key: boolean;
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
  created_at?: string;
  updated_at?: string;
  created_by?: string;
  updated_by?: string;
}

export interface AgentsResponse {
  agents: Agent[];
}
