import type { ModelGroup } from "@/components/llm_calls/fetch_models";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";

const WILDCARD_SUFFIX = "/*";

export type ModelProviders = Readonly<Record<string, readonly string[]>>;

export interface FallbackModelOption {
  label: string;
  value: string;
  sublabel?: string;
}

export const isProviderWildcard = (value: string): boolean =>
  value.endsWith(WILDCARD_SUFFIX) && value.length > WILDCARD_SUFFIX.length;

export const wildcardProvider = (value: string): string => value.slice(0, -WILDCARD_SUFFIX.length);

export const fallbackModelLabel = (value: string): string => {
  if (value === "*") {
    return "All models";
  }
  if (!isProviderWildcard(value)) {
    return value;
  }
  return `All ${getProviderLogoAndName(wildcardProvider(value)).displayName} models`;
};

export const toModelProviders = (models: readonly ModelGroup[]): ModelProviders =>
  Object.fromEntries(models.map((model) => [model.model_group, model.providers ?? []]));

const providersOf = (model: string, modelProviders: ModelProviders): readonly string[] => {
  const known = modelProviders[model] ?? [];
  if (known.length > 0) {
    return known;
  }
  const [prefix, ...rest] = model.split("/");
  return rest.length > 0 && prefix ? [prefix] : [];
};

export const modelsForProvider = (
  provider: string,
  models: readonly string[],
  modelProviders: ModelProviders,
): string[] =>
  models.filter((model) => !isProviderWildcard(model) && providersOf(model, modelProviders).includes(provider));

export const providerWildcardValues = (models: readonly string[], modelProviders: ModelProviders): string[] => {
  const providers = new Set(
    models.filter((model) => !isProviderWildcard(model)).flatMap((model) => providersOf(model, modelProviders)),
  );
  return Array.from(providers)
    .sort()
    .map((provider) => `${provider}${WILDCARD_SUFFIX}`);
};

export const toFallbackModelOption = (value: string): FallbackModelOption =>
  isProviderWildcard(value) ? { label: fallbackModelLabel(value), value, sublabel: value } : { label: value, value };

export const expandFallbackSelection = (
  values: readonly string[],
  models: readonly string[],
  modelProviders: ModelProviders,
  { primaryModel, maxFallbacks }: { primaryModel: string | null; maxFallbacks: number },
): string[] => {
  const expanded = values.flatMap((value) =>
    isProviderWildcard(value) && !models.includes(value)
      ? modelsForProvider(wildcardProvider(value), models, modelProviders)
      : [value],
  );
  return Array.from(new Set(expanded))
    .filter((model) => model !== primaryModel)
    .slice(0, maxFallbacks);
};
