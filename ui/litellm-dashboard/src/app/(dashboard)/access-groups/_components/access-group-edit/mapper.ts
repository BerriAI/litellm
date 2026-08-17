import type { AccessGroupResponse } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";
import type { components } from "@/lib/http/schema";

import type { AccessGroupFormValues } from "../access-group-form/schema";

export type AccessGroupPatchBody = components["schemas"]["AccessGroupUpdateRequest"];

export const formValuesFromAccessGroup = (group: AccessGroupResponse): AccessGroupFormValues => ({
  name: group.access_group_name,
  description: group.description ?? "",
  modelIds: group.access_model_names ?? [],
  mcpServerIds: group.access_mcp_server_ids ?? [],
  agentIds: group.access_agent_ids ?? [],
});

// The endpoint writes exactly the keys it receives, so only dirty fields are mapped and a blank description clears
export const buildAccessGroupPatchBody = (dirty: Partial<AccessGroupFormValues>): AccessGroupPatchBody => ({
  ...(dirty.name !== undefined && { access_group_name: dirty.name.trim() }),
  ...(dirty.description !== undefined && {
    description: dirty.description.trim() === "" ? null : dirty.description.trim(),
  }),
  ...(dirty.modelIds !== undefined && { access_model_names: dirty.modelIds }),
  ...(dirty.mcpServerIds !== undefined && { access_mcp_server_ids: dirty.mcpServerIds }),
  ...(dirty.agentIds !== undefined && { access_agent_ids: dirty.agentIds }),
});
