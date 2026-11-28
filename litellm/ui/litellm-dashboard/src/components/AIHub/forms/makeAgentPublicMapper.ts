import type { MakeAgentPublicValues } from "./makeAgentPublicSchema";

export const toAgentPublicationRequest = ({ agentIds }: MakeAgentPublicValues) => ({
  agent_ids: agentIds,
});
