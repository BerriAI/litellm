import type { Settings } from "../model/types";

export function normalizeFilters(filters: NonNullable<Settings["filters"]>): Settings["filters"] {
  return filters.map((f) => {
    if (!f.key.trim() || !f.value.trim()) throw new Error("Choose a key and value for every condition, or remove it");
    return { key: f.key.trim(), value: f.value.trim() };
  });
}
