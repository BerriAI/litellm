import { Agent } from "./types";
import { AgentCreateInfo } from "../networking";

/**
 * Detects the agent type from an agent's litellm_params.
 * Returns the agent_type string (e.g., "langgraph", "azure_ai_foundry", "bedrock_agentcore", or "a2a")
 */
export const detectAgentType = (agent: Agent): string => {
  const model = agent.litellm_params?.model || "";
  const customProvider = agent.litellm_params?.custom_llm_provider;

  // Check by custom_llm_provider first
  if (customProvider === "langgraph") return "langgraph";
  if (customProvider === "azure_ai") return "azure_ai_foundry";
  if (customProvider === "bedrock") return "bedrock_agentcore";

  // Check by model prefix
  if (model.startsWith("langgraph/")) return "langgraph";
  if (model.startsWith("azure_ai/agents/")) return "azure_ai_foundry";
  if (model.startsWith("bedrock/agentcore/")) return "bedrock_agentcore";

  // Default to a2a
  return "a2a";
};

/**
 * Parses agent data for dynamic form fields (non-A2A agents).
 * Extracts values from litellm_params based on the agent type metadata.
 */
export const parseDynamicAgentForForm = (
  agent: Agent,
  agentTypeInfo: AgentCreateInfo
): Record<string, any> => {
  const values: Record<string, any> = {
    agent_name: agent.agent_name,
    description: agent.agent_card_params?.description || "",
  };

  // Extract credential field values from litellm_params
  for (const field of agentTypeInfo.credential_fields) {
    if (field.include_in_litellm_params !== false) {
      values[field.key] = agent.litellm_params?.[field.key] || field.default_value || "";
    } else {
      // For fields not in litellm_params (like agent_id), try to extract from model string
      if (agentTypeInfo.model_template && agent.litellm_params?.model) {
        const model = agent.litellm_params.model;
        const templateParts = agentTypeInfo.model_template.split("/");
        const modelParts = model.split("/");
        
        // Find the placeholder position and extract the value
        templateParts.forEach((part, index) => {
          if (part === `{${field.key}}` && modelParts[index]) {
            values[field.key] = modelParts[index];
          }
        });
      }
    }
  }

  // Extract cost configuration
  values.cost_per_query = agent.litellm_params?.cost_per_query;
  values.input_cost_per_token = agent.litellm_params?.input_cost_per_token;
  values.output_cost_per_token = agent.litellm_params?.output_cost_per_token;

  return values;
};

