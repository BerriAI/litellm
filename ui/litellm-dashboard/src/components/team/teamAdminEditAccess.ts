import { z } from "zod";
import { numberOrNull } from "@/lib/forms/numberOrNull";

export const TEAM_ADMIN_EDITABLE_TEAM_FIELDS_SETTING = "team_admin_editable_team_fields";

export const TEAM_ADMIN_RAISE_MAX_BUDGET_PERMISSION = "raise_max_budget";

export const TEAM_ADMIN_EDITING_DISABLED_TITLE = "Team admins cannot edit team settings on this proxy";
export const TEAM_ADMIN_EDITING_DISABLED_DESCRIPTION =
  "Ask a proxy admin to enable fields under Settings > UI > Team admin editable fields.";

const callerEditAccessSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("unrestricted") }),
  z.object({
    kind: z.literal("team_admin"),
    editable_fields: z.array(z.string()),
    may_raise_max_budget: z.boolean().catch(false),
  }),
  z.object({ kind: z.literal("team_admin_disabled") }),
  z.object({ kind: z.literal("none") }),
]);

export type CallerEditAccess = z.infer<typeof callerEditAccessSchema>;

export type TeamEditAccess =
  | { readonly kind: "unrestricted" }
  | { readonly kind: "team_admin"; readonly editableFields: ReadonlySet<string>; readonly mayRaiseMaxBudget: boolean }
  | { readonly kind: "team_admin_disabled" }
  | { readonly kind: "none" };

const fieldListSchema = z.array(z.string()).catch([]);

export const parseTeamAdminEditableFields = (uiSettingsValues: unknown): readonly string[] => {
  const values = z.record(z.string(), z.unknown()).catch({}).parse(uiSettingsValues);
  return fieldListSchema.parse(values[TEAM_ADMIN_EDITABLE_TEAM_FIELDS_SETTING]);
};

export const parseSupportedTeamAdminEditableFields = (uiSettingsFieldSchema: unknown): readonly string[] => {
  const property = z
    .object({ properties: z.object({ [TEAM_ADMIN_EDITABLE_TEAM_FIELDS_SETTING]: z.object({ items: z.unknown() }) }) })
    .safeParse(uiSettingsFieldSchema);
  if (!property.success) return [];
  const items = z
    .object({ enum: z.unknown() })
    .safeParse(property.data.properties[TEAM_ADMIN_EDITABLE_TEAM_FIELDS_SETTING].items);
  return items.success ? fieldListSchema.parse(items.data.enum) : [];
};

export const TEAM_ADMIN_SETTINGS_FIELDS = ["tpm_limit", "rpm_limit", "max_budget"] as const;

export type TeamAdminSettingsField = (typeof TEAM_ADMIN_SETTINGS_FIELDS)[number];

const TEAM_ADMIN_FIELD_LABELS: ReadonlyMap<string, string> = new Map([
  ["tpm_limit", "Tokens per minute Limit (TPM)"],
  ["rpm_limit", "Requests per minute Limit (RPM)"],
  ["max_budget", "Max Budget (USD)"],
  [TEAM_ADMIN_RAISE_MAX_BUDGET_PERMISSION, "Raise the team's max budget"],
  ["projects", "Create and update projects"],
  ["member_key_budgets", "Update budgets on team members' keys"],
]);

export const teamAdminFieldLabel = (field: string): string => TEAM_ADMIN_FIELD_LABELS.get(field) ?? field;

export type TeamAdminSettingsValues = { readonly [F in TeamAdminSettingsField]?: string | number | null };

export type TeamAdminSettingsChanges = { readonly [F in TeamAdminSettingsField]?: number | null };

export const teamAdminSettingsChanges = (
  values: TeamAdminSettingsValues,
  initialValues: TeamAdminSettingsValues,
  editableFields: ReadonlySet<string>,
): TeamAdminSettingsChanges =>
  Object.fromEntries(
    TEAM_ADMIN_SETTINGS_FIELDS.flatMap((field) => {
      const value = numberOrNull(values[field]);
      return editableFields.has(field) && value !== numberOrNull(initialValues[field]) ? [[field, value]] : [];
    }),
  );

export const parseTeamEditAccess = (callerEditAccess: unknown): TeamEditAccess => {
  const parsed = callerEditAccessSchema.safeParse(callerEditAccess);
  if (!parsed.success) return { kind: "none" };
  if (parsed.data.kind === "team_admin") {
    return {
      kind: "team_admin",
      editableFields: new Set(parsed.data.editable_fields),
      mayRaiseMaxBudget: parsed.data.may_raise_max_budget,
    };
  }
  return parsed.data;
};
