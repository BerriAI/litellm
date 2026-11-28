import type { MountedFormValues } from "../../common_components/MountedFormField";

export const keyCreateDefaults = (teamId: string | null): MountedFormValues => ({
  team_id: teamId,
  key_type: "llm_api",
  tpm_limit_type: null,
  rpm_limit_type: null,
  mcp_tool_permissions: {},
  duration: "",
});
