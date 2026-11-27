import type { TeamDefaultSettingsFields } from "./schema";

export const emptyTeamDefaultSettings: TeamDefaultSettingsFields = {
  max_budget: null,
  budget_duration: null,
  tpm_limit: null,
  rpm_limit: null,
  models: [],
  team_member_permissions: [],
  organization_id: null,
  access_group_ids: [],
};

export const teamDefaultSettingsToFields = (stored: object | null | undefined): TeamDefaultSettingsFields => ({
  ...emptyTeamDefaultSettings,
  ...(stored || {}),
});
