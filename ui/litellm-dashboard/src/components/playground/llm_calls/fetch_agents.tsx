// fetch_agents.tsx

import { getProxyBaseUrl, getGlobalLitellmHeaderName, modelInfoCall } from "../../networking";

export interface Agent {
  agent_id: string;
  agent_name: string;
  description?: string;
  agent_card_params?: {
    name?: string;
    description?: string;
    url?: string;
  };
}

/** MCP tool entry in the same format as chat completions API (litellm_params.tools) */
export interface MCPToolEntry {
  type: "mcp";
  server_label?: string;
  server_url: string;
  require_approval?: string;
  allowed_tools?: string[];
}

/** Agent model from /model/info where litellm_params.model starts with "litellm_agent/" */
export interface AgentModel {
  model_name: string;
  litellm_params: {
    model: string;
    litellm_system_prompt?: string;
    /** Saved MCP tools array (same shape as chat completions API tools) */
    tools?: MCPToolEntry[];
    [key: string]: unknown;
  };
  model_info?: Record<string, unknown> | null;
}

/**
 * Fetches available A2A agents from /v1/agents endpoint.
 */
export const fetchAvailableAgents = async (
  accessToken: string,
  customBaseUrl?: string,
): Promise<Agent[]> => {
  try {
    const proxyBaseUrl = customBaseUrl || getProxyBaseUrl();
    const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/agents` : `/v1/agents`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Failed to fetch agents");
    }

    const agents: Agent[] = await response.json();
    console.log("Fetched agents:", agents);

    // Sort agents alphabetically by name
    agents.sort((a, b) => {
      const nameA = a.agent_name || a.agent_id;
      const nameB = b.agent_name || b.agent_id;
      return nameA.localeCompare(nameB);
    });

    return agents;
  } catch (error) {
    console.error("Error fetching agents:", error);
    throw error;
  }
};

/**
 * Fetches available litellm_agent models from /v2/model/info.
 * Filters for models where litellm_params.model starts with "litellm_agent/".
 */
export const fetchAvailableAgentModels = async (
  accessToken: string,
  userID: string,
  userRole: string,
  customBaseUrl?: string,
): Promise<AgentModel[]> => {
  try {
    const size = 200;
    const response = await modelInfoCall(accessToken, userID, userRole, 1, size);
    const data = response?.data ?? [];
    const list = Array.isArray(data) ? data : [];

    const agentModels: AgentModel[] = list
      .filter(
        (m: { litellm_params?: { model?: string } }) =>
          typeof m?.litellm_params?.model === "string" &&
          m.litellm_params.model.startsWith("litellm_agent/"),
      )
      .map((m: any) => ({
        model_name: m.model_name ?? m.model_group ?? "",
        litellm_params: {
          ...m.litellm_params,
          model: m.litellm_params.model,
          litellm_system_prompt: m.litellm_params?.litellm_system_prompt,
          tools: Array.isArray(m.litellm_params?.tools) ? m.litellm_params.tools : undefined,
        },
        model_info: m.model_info ?? null,
      }));

    agentModels.sort((a, b) => a.model_name.localeCompare(b.model_name));
    return agentModels;
  } catch (error) {
    console.error("Error fetching agent models:", error);
    throw error;
  }
};
