// fetch_agents.tsx

import { getProxyBaseUrl } from "../../networking";

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

/**
 * Fetches available A2A agents from /v1/agents endpoint.
 */
export const fetchAvailableAgents = async (accessToken: string): Promise<Agent[]> => {
  try {
    const proxyBaseUrl = getProxyBaseUrl();
    const url = proxyBaseUrl ? `${proxyBaseUrl}/v1/agents` : `/v1/agents`;

    const response = await fetch(url, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${accessToken}`,
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
