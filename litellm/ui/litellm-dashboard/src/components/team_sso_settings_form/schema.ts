import { z } from "zod";

const teamDefaultSettingsShape = {
  max_budget: z.number().nullable(),
  budget_duration: z.string().nullable(),
  tpm_limit: z.number().nullable(),
  rpm_limit: z.number().nullable(),
  models: z.array(z.string()),
  team_member_permissions: z.array(z.string()),
  organization_id: z.string().nullable(),
  access_group_ids: z.array(z.string()),
};

export const teamDefaultSettingsSchema = z.looseObject(teamDefaultSettingsShape);

export type TeamDefaultSettingsFields = z.infer<typeof teamDefaultSettingsSchema>;
