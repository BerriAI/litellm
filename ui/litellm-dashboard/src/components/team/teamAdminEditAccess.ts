import { z } from "zod/v4";

export const TEAM_ADMIN_EDITABLE_TEAM_FIELDS_SETTING = "team_admin_editable_team_fields";

export const TEAM_ADMIN_EDITING_DISABLED_TITLE = "Team admins cannot edit team settings on this proxy";
export const TEAM_ADMIN_EDITING_DISABLED_DESCRIPTION =
  "Ask a proxy admin to enable fields under Settings > UI > Team admin editable fields.";

export type TeamEditAccess =
  | { readonly kind: "unrestricted" }
  | { readonly kind: "team_admin"; readonly editableFields: ReadonlySet<string> }
  | { readonly kind: "team_admin_disabled" };

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

export const resolveTeamEditAccess = (editsAsTeamAdmin: boolean, uiSettingsValues: unknown): TeamEditAccess => {
  if (!editsAsTeamAdmin) return { kind: "unrestricted" };
  const editableFields = parseTeamAdminEditableFields(uiSettingsValues);
  return editableFields.length === 0
    ? { kind: "team_admin_disabled" }
    : { kind: "team_admin", editableFields: new Set(editableFields) };
};
