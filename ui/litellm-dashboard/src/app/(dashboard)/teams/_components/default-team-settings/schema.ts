import { z } from "zod/v4";

import { isSelectablePermission, type KeyManagementRoute } from "./permissions";

const isBlank = (value: string): boolean => value.trim() === "";

const amountOrEmpty = z
  .string()
  .refine(
    (value) => isBlank(value) || (Number.isFinite(Number(value)) && Number(value) >= 0),
    "Must be a non-negative number",
  );

const wholeNumberOrEmpty = z
  .string()
  .refine(
    (value) => isBlank(value) || (Number.isInteger(Number(value)) && Number(value) >= 0),
    "Must be a non-negative whole number",
  );

const selectablePermission = z.custom<KeyManagementRoute>(
  (value) => typeof value === "string" && isSelectablePermission(value),
  "Unknown permission",
);

export const defaultTeamSettingsSchema = z.object({
  max_budget: amountOrEmpty,
  budget_duration: z.string(),
  tpm_limit: wholeNumberOrEmpty,
  rpm_limit: wholeNumberOrEmpty,
  models: z.array(z.string()),
  team_member_permissions: z.array(selectablePermission),
});

export type DefaultTeamSettingsFormValues = z.output<typeof defaultTeamSettingsSchema>;
