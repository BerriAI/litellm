import type { MakeMcpPublicValues } from "./makeMcpPublicSchema";

export const toMcpPublicationRequest = ({ serverIds }: MakeMcpPublicValues) => ({
  mcp_server_ids: serverIds,
});
