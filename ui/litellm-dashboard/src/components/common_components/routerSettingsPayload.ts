import { RouterSettingsAccordionValue } from "./RouterSettingsAccordion";

export type RouterSettings = RouterSettingsAccordionValue["router_settings"];

const EDITOR_OWNED_FIELDS: Record<keyof RouterSettings, true> = {
  routing_strategy: true,
  allowed_fails: true,
  cooldown_time: true,
  num_retries: true,
  timeout: true,
  retry_after: true,
  fallbacks: true,
  context_window_fallbacks: true,
  retry_policy: true,
  model_group_alias: true,
  enable_tag_filtering: true,
  routing_strategy_args: true,
};

const EDITOR_OWNED_KEYS = Object.keys(EDITOR_OWNED_FIELDS) as Array<keyof RouterSettings>;

const isMeaningfulRouterSetting = (value: unknown): boolean => {
  if (value === null || value === undefined || value === "" || value === false) return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return true;
};

export const hasRouterSettings = (settings: Record<string, unknown> | null | undefined): boolean =>
  settings != null && Object.values(settings).some(isMeaningfulRouterSetting);

/**
 * The stored settings narrowed to what the editor renders. The stored blob is untyped JSON
 * from /key/info, so this projection is the one place it is read as RouterSettings.
 */
export const routerSettingsEditorValue = (
  stored: Record<string, unknown> | null | undefined,
): RouterSettingsAccordionValue | undefined =>
  stored
    ? {
        router_settings: Object.fromEntries(
          EDITOR_OWNED_KEYS.filter((key) => key in stored).map((key) => [key, stored[key]]),
        ) as RouterSettings,
      }
    : undefined;

/**
 * Router settings to put on a /key/update payload, or undefined to leave the field off.
 * The editor only renders EDITOR_OWNED_FIELDS, so its value is merged over the stored object
 * rather than replacing it, and routing fields the editor cannot show survive an unrelated edit.
 * Emptying every field sends {}, which the proxy reads as "no key-level override" so the key
 * falls back to its team and global settings, where an all-null blob would pin it to nulls.
 */
export const routerSettingsUpdate = (
  edited: RouterSettings | null | undefined,
  stored: Record<string, unknown> | null | undefined,
): Record<string, unknown> | undefined => {
  if (!edited) return undefined;
  const editorOwned = Object.fromEntries(EDITOR_OWNED_KEYS.map((key) => [key, edited[key] ?? null]));
  const merged: Record<string, unknown> = { ...stored, ...editorOwned };
  if (hasRouterSettings(merged)) return merged;
  return hasRouterSettings(stored) ? {} : undefined;
};
