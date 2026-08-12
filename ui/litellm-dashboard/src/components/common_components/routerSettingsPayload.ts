import { RouterSettingsAccordionValue } from "./RouterSettingsAccordion";

export type RouterSettings = RouterSettingsAccordionValue["router_settings"];

const isMeaningfulRouterSetting = (value: unknown): boolean => {
  if (value === null || value === undefined || value === "" || value === false) return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return true;
};

export const hasRouterSettings = (settings: Record<string, unknown> | null | undefined): boolean =>
  settings != null && Object.values(settings).some(isMeaningfulRouterSetting);

/**
 * Router settings to put on a /key/update payload, or undefined to leave the field off.
 * Clearing every field must reach the server, while a key that never had router settings
 * must not start sending an all-null object, so the value is only sent when the editor
 * holds something or there are stored settings to overwrite.
 */
export const routerSettingsUpdate = (
  edited: RouterSettings | null | undefined,
  stored: Record<string, unknown> | null | undefined,
): RouterSettings | undefined => {
  if (!edited) return undefined;
  const editedRecord: Record<string, unknown> = edited;
  return hasRouterSettings(editedRecord) || hasRouterSettings(stored) ? edited : undefined;
};
