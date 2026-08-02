import { z } from "zod/v4";

import type { components } from "@/lib/http/schema";

import { isSelectablePermission } from "./permissions";
import type { DefaultTeamSettingsFormValues } from "./schema";

export type DefaultTeamSettings = components["schemas"]["DefaultTeamSettingsResponse"];
export type DefaultTeamParams = components["schemas"]["DefaultTeamSSOParams"];

const serverValuesShape = {
  max_budget: z.number().nullish().catch(null),
  budget_duration: z.string().nullish().catch(null),
  tpm_limit: z.number().nullish().catch(null),
  rpm_limit: z.number().nullish().catch(null),
  models: z.array(z.string()).nullish().catch(null),
  team_member_permissions: z.array(z.string()).nullish().catch(null),
};

const serverValuesSchema = z.object(serverValuesShape);

export const settingsToForm = (values: DefaultTeamSettings["values"]): DefaultTeamSettingsFormValues => {
  const parsed = serverValuesSchema.parse(values);

  return {
    max_budget: parsed.max_budget?.toString() ?? "",
    budget_duration: parsed.budget_duration ?? "",
    tpm_limit: parsed.tpm_limit?.toString() ?? "",
    rpm_limit: parsed.rpm_limit?.toString() ?? "",
    models: parsed.models ?? [],
    team_member_permissions: (parsed.team_member_permissions ?? []).filter(isSelectablePermission),
  };
};

const numberOrNull = (raw: string): number | null => (raw.trim() === "" ? null : Number(raw));

const textOrNull = (raw: string): string | null => (raw.trim() === "" ? null : raw);

const listOrNull = <T>(items: readonly T[]): T[] | null => (items.length === 0 ? null : [...items]);

export const buildBody = (values: DefaultTeamSettingsFormValues): DefaultTeamParams => ({
  max_budget: numberOrNull(values.max_budget),
  budget_duration: textOrNull(values.budget_duration),
  tpm_limit: numberOrNull(values.tpm_limit),
  rpm_limit: numberOrNull(values.rpm_limit),
  models: [...values.models],
  team_member_permissions: listOrNull(values.team_member_permissions),
});
