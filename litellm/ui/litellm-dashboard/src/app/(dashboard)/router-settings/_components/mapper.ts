import type { generalSettingsItem } from "./general_settings";
import { GeneralSettingsRows } from "./schema";

export function mapGeneralSetting(
  rows: generalSettingsItem[],
  fieldName: string,
  update: (setting: generalSettingsItem) => generalSettingsItem,
) {
  return new GeneralSettingsRows(rows.map((setting) => (setting.field_name === fieldName ? update(setting) : setting)));
}

export function toGeneralSettingUpdate(rows: generalSettingsItem[], fieldName: string) {
  const setting = rows.find((row) => row.field_name === fieldName);
  const fieldValue: unknown = setting?.field_value;

  if (fieldValue != null) return { kind: "update", value: fieldValue } as const;
  return { kind: setting?.field_type === "Select" || setting?.field_type === "List" ? "reset" : "skip" } as const;
}

export function resetGeneralSetting(rows: generalSettingsItem[], fieldName: string) {
  return mapGeneralSetting(rows, fieldName, (setting) => ({
    ...setting,
    stored_in_db: null,
    field_value: setting.field_default_value ?? null,
  }));
}
