import { z } from "zod";
import { storageKey } from "@/lib/storage";

export interface AliasItem {
  id: string;
  aliasName: string;
  targetModelGroup: string;
}

const aliasDisplayFields = {
  sortBy: z.enum(["aliasName", "targetModelGroup"]),
  descending: z.boolean(),
  pinned: z.array(z.string()).transform((names) => [...new Set(names)]),
  lastPrefix: z.string(),
};
const aliasDisplaySchema = z.object(aliasDisplayFields);

export type AliasDisplayPreferences = z.infer<typeof aliasDisplaySchema>;

export const DEFAULT_ALIAS_DISPLAY: AliasDisplayPreferences = {
  sortBy: "aliasName",
  descending: false,
  pinned: [],
  lastPrefix: "",
};

export const aliasDisplayStorage = (scope: string) =>
  storageKey("local", `litellm_alias_display_v1_${scope}`, aliasDisplaySchema, DEFAULT_ALIAS_DISPLAY);

const alphabetical = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });

export function orderAliases(aliases: readonly AliasItem[], preferences: AliasDisplayPreferences): AliasItem[] {
  const ranks = new Map(preferences.pinned.map((name, index) => [name, index]));
  const group = (name: string): number => {
    if (ranks.has(name)) return 0;
    return preferences.lastPrefix && name.startsWith(preferences.lastPrefix) ? 2 : 1;
  };
  return [...aliases].sort((a, b) => {
    const groupDifference = group(a.aliasName) - group(b.aliasName);
    if (groupDifference) return groupDifference;
    if (ranks.has(a.aliasName) && ranks.has(b.aliasName)) {
      return ranks.get(a.aliasName)! - ranks.get(b.aliasName)!;
    }
    const fieldDifference = alphabetical.compare(a[preferences.sortBy], b[preferences.sortBy]);
    return fieldDifference * (preferences.descending ? -1 : 1) || a.aliasName.localeCompare(b.aliasName);
  });
}

export function movePinnedAlias(pinned: readonly string[], name: string, direction: -1 | 1): string[] {
  const index = pinned.indexOf(name);
  const target = index + direction;
  if (index < 0 || target < 0 || target >= pinned.length) return [...pinned];
  return pinned.map((value, position) => {
    if (position === index) return pinned[target];
    return position === target ? name : value;
  });
}
