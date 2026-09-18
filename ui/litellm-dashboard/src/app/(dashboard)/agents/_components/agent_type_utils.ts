import { Agent } from "@/components/agents/types";
import { AgentCreateInfo } from "@/components/networking";

/**
 * Detects the agent type from an agent's litellm_params.
 * Returns the agent_type string (e.g., "langgraph", "azure_ai_foundry", "bedrock_agentcore", or "a2a")
 */
export const detectAgentType = (agent: Agent): string => {
  const model = agent.litellm_params?.model || "";
  const customProvider = agent.litellm_params?.custom_llm_provider;

  // Check by custom_llm_provider first
  if (customProvider === "langflow") return "langflow";
  if (customProvider === "langgraph") return "langgraph";
  if (customProvider === "azure_ai") return "azure_ai_foundry";
  if (customProvider === "bedrock") return "bedrock_agentcore";

  // Check by model prefix
  if (model.startsWith("langflow/")) return "langflow";
  if (model.startsWith("langgraph/")) return "langgraph";
  if (model.startsWith("azure_ai/agents/")) return "azure_ai_foundry";
  if (model.startsWith("bedrock/agentcore/")) return "bedrock_agentcore";

  // Default to a2a
  return "a2a";
};

const escapeRegExp = (segment: string): string => segment.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * Reverses a `model_template` (e.g. "bedrock/agentcore/{agent_runtime_arn}") against a stored
 * `model` string to recover the placeholder values that produced it. Builds a regex from the
 * template's literal segments rather than matching by split("/") position, because a
 * placeholder's value can itself contain "/" (an AWS ARN's "runtime/<runtime-id>" resource path,
 * a Vertex AI reasoning engine's "projects/.../reasoningEngines/..." resource id) and would
 * otherwise be cut off at the first one.
 */
export const extractModelTemplateValues = (template: string, model: string): Record<string, string> => {
  // Splitting on a regex with a capturing group interleaves the captured placeholder
  // names between the surrounding literal segments, e.g. "a/{x}/b" -> ["a/", "x", "/b"].
  const parts = template.split(/\{([a-zA-Z0-9_]+)\}/g);
  const fieldNames = parts.filter((_part, index) => index % 2 === 1);
  const pattern = parts.map((part, index) => (index % 2 === 1 ? "(.+)" : escapeRegExp(part))).join("");

  const match = model.match(new RegExp(`^${pattern}$`));
  if (!match) return {};

  return Object.fromEntries(fieldNames.map((name, index) => [name, match[index + 1]]));
};

/**
 * Parses agent data for dynamic form fields (non-A2A agents).
 * Extracts values from litellm_params based on the agent type metadata.
 */
export const parseDynamicAgentForForm = (agent: Agent, agentTypeInfo: AgentCreateInfo): Record<string, any> => {
  const values: Record<string, any> = {
    agent_name: agent.agent_name,
    description: agent.agent_card_params?.description || "",
  };

  const templateValues =
    agentTypeInfo.model_template && agent.litellm_params?.model
      ? extractModelTemplateValues(agentTypeInfo.model_template, agent.litellm_params.model)
      : {};

  // Extract credential field values from litellm_params
  for (const field of agentTypeInfo.credential_fields) {
    if (field.include_in_litellm_params !== false) {
      values[field.key] = agent.litellm_params?.[field.key] || field.default_value || "";
    } else if (templateValues[field.key] !== undefined) {
      // For fields not in litellm_params (like agent_runtime_arn), recover from the model string
      values[field.key] = templateValues[field.key];
    }
  }

  // Extract cost configuration
  values.cost_per_query = agent.litellm_params?.cost_per_query;
  values.input_cost_per_token = agent.litellm_params?.input_cost_per_token;
  values.output_cost_per_token = agent.litellm_params?.output_cost_per_token;

  return values;
};
