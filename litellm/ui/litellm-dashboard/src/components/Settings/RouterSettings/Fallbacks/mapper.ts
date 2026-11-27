import type { z } from "zod";
import type { FallbackGroup } from "./FallbackGroupConfig";
import type { addFallbacksSchema, editFallbacksSchema } from "./schema";

export type FallbackEntry = Record<string, string[]>;
export type Fallbacks = FallbackEntry[];

export const newFallbackGroups = (): FallbackGroup[] => [{ id: "1", primaryModel: null, fallbackModels: [] }];

export function toEditFallbackGroup(entry: FallbackEntry): FallbackGroup {
  const primaryModel = Object.keys(entry)[0] ?? null;
  return {
    id: "edit",
    primaryModel,
    fallbackModels: primaryModel ? [...(entry[primaryModel] ?? [])] : [],
  };
}

export function appendFallbacks(value: Fallbacks, { groups }: z.output<typeof addFallbacksSchema>): Fallbacks {
  return [...(value || []), ...groups.map((group) => ({ [group.primaryModel]: group.fallbackModels }))];
}

export function replaceFallbacks(value: Fallbacks, { group }: z.output<typeof editFallbacksSchema>): Fallbacks {
  const { primaryModel, fallbackModels } = group;
  return (value || []).map((entry) => (primaryModel in entry ? { ...entry, [primaryModel]: fallbackModels } : entry));
}
