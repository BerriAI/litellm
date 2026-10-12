import type { ModelGroup } from "@/components/llm_calls/fetch_models";
import type { FallbackGroup } from "./FallbackGroupConfig";

export function providerModels(models: ModelGroup[], provider: string): string[] {
  return [
    ...new Set(
      models
        .filter((model) => model.providers?.includes(provider) && !model.model_group.includes("*"))
        .map((model) => model.model_group),
    ),
  ].sort();
}

export function primaryModels(group: FallbackGroup, models: ModelGroup[]): string[] {
  if (group.primaryProvider) return providerModels(models, group.primaryProvider);
  return group.primaryModel ? [group.primaryModel] : [];
}

export function buildFallbackEntries(
  groups: FallbackGroup[],
  models: ModelGroup[],
  current: Record<string, string[]>[],
): { entries: Record<string, string[]>[]; error?: never } | { error: string; entries?: never } {
  const expanded = groups.map((group) => ({
    primaries: primaryModels(group, models),
    targets: [...new Set(group.fallbackModels)],
  }));
  if (expanded.some(({ primaries, targets }) => primaries.length === 0 || targets.length === 0)) {
    return { error: "Please complete configuration for all groups." };
  }
  if (expanded.some(({ targets }) => targets.length > 10)) {
    return { error: "Each chain can contain at most 10 fallback models." };
  }
  const allPrimaries = expanded.flatMap(({ primaries }) => primaries);
  const existing = new Set(current.flatMap((entry) => Object.keys(entry)));
  const duplicate = allPrimaries.find((model, index) => existing.has(model) || allPrimaries.indexOf(model) !== index);
  if (duplicate) {
    return {
      error: `Fallbacks are already configured for ${duplicate}. Edit its existing chain or remove overlapping groups.`,
    };
  }
  const entries = expanded.flatMap(({ primaries, targets }) =>
    primaries.map((primary) => ({ [primary]: targets.filter((target) => target !== primary) })),
  );
  if (entries.some((entry) => Object.values(entry)[0].length === 0)) {
    return { error: "Choose a fallback different from each primary model." };
  }
  return { entries };
}
