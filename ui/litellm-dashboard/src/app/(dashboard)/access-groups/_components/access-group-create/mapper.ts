import type { components } from "@/lib/http/schema";

import type { AccessGroupCreateFormValues } from "./schema";

export type AccessGroupCreateBody = components["schemas"]["AccessGroupCreateRequest"];

export const emptyAccessGroupFormValues: AccessGroupCreateFormValues = {
  name: "",
  description: "",
  modelIds: [],
  mcpServerIds: [],
  agentIds: [],
};

export const buildAccessGroupCreateBody = (values: AccessGroupCreateFormValues): AccessGroupCreateBody => ({
  access_group_name: values.name.trim(),
  ...(values.description.trim() !== "" && { description: values.description.trim() }),
  ...(values.modelIds.length > 0 && { access_model_names: values.modelIds }),
  ...(values.mcpServerIds.length > 0 && { access_mcp_server_ids: values.mcpServerIds }),
  ...(values.agentIds.length > 0 && { access_agent_ids: values.agentIds }),
});
