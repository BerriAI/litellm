import { ModelGroup } from "@/components/llm_calls/fetch_models";
import { getProviderLogoAndName, provider_map, resolveLitellmProviderSlug } from "@/components/provider_info_helpers";

export interface FallbackOption {
  label: string;
  value: string;
}

const KNOWN_PROVIDER_SLUGS: ReadonlySet<string> = new Set(
  Object.values(provider_map).map((slug) => slug.toLowerCase()),
);

const modelGroupProviderPrefix = (modelGroup: string): string | null => {
  const slashIndex = modelGroup.indexOf("/");
  if (slashIndex <= 0) {
    return null;
  }
  const prefix = modelGroup.slice(0, slashIndex).toLowerCase();
  return KNOWN_PROVIDER_SLUGS.has(prefix) ? prefix : null;
};

export const providerSlugsFor = (modelGroups: ModelGroup[]): string[] => {
  const slugs = new Set<string>();
  for (const modelGroup of modelGroups) {
    for (const provider of modelGroup.providers ?? []) {
      slugs.add(resolveLitellmProviderSlug(provider));
    }
    const prefix = modelGroupProviderPrefix(modelGroup.model_group);
    if (prefix) {
      slugs.add(prefix);
    }
  }
  return Array.from(slugs).sort((a, b) => a.localeCompare(b));
};

export const providerWildcardOption = (slug: string): FallbackOption => {
  const { displayName } = getProviderLogoAndName(slug);
  return { value: `${slug}/*`, label: `All ${displayName} Models` };
};

export const providerWildcardOptions = (modelGroups: ModelGroup[]): FallbackOption[] =>
  providerSlugsFor(modelGroups).map(providerWildcardOption);

export const primaryModelOptions = (modelGroups: ModelGroup[]): FallbackOption[] => {
  const wildcardValues = new Set(providerWildcardOptions(modelGroups).map((option) => option.value));
  const groupOptions = Array.from(new Set(modelGroups.map((modelGroup) => modelGroup.model_group)))
    .filter((name) => !wildcardValues.has(name))
    .sort((a, b) => a.localeCompare(b))
    .map((name) => ({ label: name, value: name }));
  return [...providerWildcardOptions(modelGroups), ...groupOptions];
};

export const fallbackChainOptions = (modelGroups: ModelGroup[]): FallbackOption[] =>
  Array.from(new Set(modelGroups.map((modelGroup) => modelGroup.model_group)))
    .filter((name) => !name.includes("*"))
    .sort((a, b) => a.localeCompare(b))
    .map((name) => ({ label: name, value: name }));

export const resolveFallbackProvider = (
  model: string,
  modelCostMap: Record<string, { litellm_provider?: string }> | null | undefined,
  modelGroups: ModelGroup[],
): string => {
  const fromCostMap = modelCostMap?.[model]?.litellm_provider;
  if (typeof fromCostMap === "string" && fromCostMap !== "") {
    return fromCostMap;
  }
  const provider = modelGroups.find((modelGroup) => modelGroup.model_group === model)?.providers?.[0];
  if (provider) {
    return resolveLitellmProviderSlug(provider);
  }
  return modelGroupProviderPrefix(model) ?? "";
};

export const fallbackPrimaryLabel = (model: string): string => {
  if (model.endsWith("/*")) {
    const slug = model.slice(0, -2);
    if (KNOWN_PROVIDER_SLUGS.has(slug.toLowerCase())) {
      const { displayName } = getProviderLogoAndName(slug);
      return `All ${displayName} Models`;
    }
  }
  return model;
};
