import { z } from "zod";
import type { components } from "@/lib/http/schema";

export type AgentAttachedKey = components["schemas"]["AgentKeySummary"];

export type AgentObjectPermission = components["schemas"]["AgentObjectPermission"];
export type AgentKillSwitchConfig = components["schemas"]["AgentKillSwitchConfig"];
export type AgentKillSwitchResult = components["schemas"]["AgentKillSwitchResult"];

type ApiAgent = components["schemas"]["AgentResponse"];

const agentParamsShape = {
  model: z.string().nullish(),
  custom_llm_provider: z.string().nullish(),
  make_public: z.boolean().nullish(),
  cost_per_query: z.number().nullish(),
  input_cost_per_token: z.number().nullish(),
  output_cost_per_token: z.number().nullish(),
};

const agentParamsSchema = z.object(agentParamsShape).passthrough().nullish();

const agentCardShape = {
  name: z.string().nullish(),
  description: z.string().nullish(),
  url: z.string().nullish(),
  version: z.string().nullish(),
  protocolVersion: z.string().nullish(),
  iconUrl: z.string().nullish(),
  documentationUrl: z.string().nullish(),
  defaultInputModes: z.array(z.string()).nullish(),
  defaultOutputModes: z.array(z.string()).nullish(),
  provider: z.object({ organization: z.string().nullish(), url: z.string().nullish() }).passthrough().nullish(),
  capabilities: z
    .object({
      streaming: z.boolean().nullish(),
      pushNotifications: z.boolean().nullish(),
      stateTransitionHistory: z.boolean().nullish(),
    })
    .passthrough()
    .nullish(),
  skills: z
    .array(
      z
        .object({
          id: z.string().optional(),
          name: z.string().optional(),
          description: z.string().optional(),
          tags: z.array(z.string()).nullish(),
          examples: z.array(z.string()).nullish(),
        })
        .passthrough(),
    )
    .nullish(),
};

const agentCardSchema = z.object(agentCardShape).passthrough();

const agentPermissionShape = {
  agents: z.array(z.string()).nullish(),
  models: z.array(z.string()).nullish(),
  mcp_servers: z.array(z.string()).nullish(),
  mcp_access_groups: z.array(z.string()).nullish(),
  mcp_toolsets: z.array(z.string()).nullish(),
  mcp_tool_permissions: z.record(z.string(), z.array(z.string())).nullish(),
};

const agentPermissionSchema: z.ZodType<AgentObjectPermission | null | undefined> = z
  .object(agentPermissionShape)
  .passthrough()
  .nullish();

export type Agent = Omit<ApiAgent, "litellm_params" | "agent_card_params" | "object_permission"> & {
  litellm_params: z.output<typeof agentParamsSchema>;
  agent_card_params: z.output<typeof agentCardSchema>;
  object_permission: AgentObjectPermission | null | undefined;
};

export const toAgentCard = (card: ApiAgent["agent_card_params"]): Agent["agent_card_params"] =>
  agentCardSchema.parse(card);

export const toAgent = (agent: ApiAgent): Agent => ({
  ...agent,
  litellm_params: agentParamsSchema.parse(agent.litellm_params),
  agent_card_params: toAgentCard(agent.agent_card_params),
  object_permission: agentPermissionSchema.parse(agent.object_permission),
});

export interface AgentsResponse {
  agents: Agent[];
}
