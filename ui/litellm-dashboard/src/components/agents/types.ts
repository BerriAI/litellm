export interface Agent {
  agent_id: string;
  agent_name: string;
  litellm_params: {
    model: string;
    [key: string]: any;
  };
  agent_card_params?: {
    description?: string;
    [key: string]: any;
  };
  created_at?: string;
  updated_at?: string;
  created_by?: string;
  updated_by?: string;
}

export interface AgentsResponse {
  agents: Agent[];
}
