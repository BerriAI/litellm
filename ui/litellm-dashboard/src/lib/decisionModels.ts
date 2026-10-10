import { z } from "zod";

import { getProviderLogoAndName } from "@/components/provider_info_helpers";

export const DECISIONS_DOCS_URL = "https://docs.litellm.ai/docs/decisions";
export const DECISIONS_PROVIDERS_DOCS_URL = `${DECISIONS_DOCS_URL}#supported-providers`;
export const SYSTEM_ONE_PLAYGROUND_ROUTE = "playground?tab=system-one";

const DECISION_ENDPOINTS: ReadonlySet<string> = new Set(["/v1/decisions", "/v1/systemone"]);

const costMapEntrySchema = z.object({
  litellm_provider: z.string(),
  mode: z.string().nullish(),
  supported_endpoints: z.array(z.string()).nullish(),
});

type CostMapEntry = z.infer<typeof costMapEntrySchema>;

interface ProviderDecisionModels {
  readonly names: readonly string[];
  readonly decisionOnly: boolean;
}

export interface DecisionCatalog {
  readonly models: ReadonlySet<string>;
  readonly providers: ReadonlyMap<string, ProviderDecisionModels>;
}

export const EMPTY_DECISION_CATALOG: DecisionCatalog = { models: new Set(), providers: new Map() };

export const isDecisionMode = (mode: string | null | undefined): boolean => mode === "evaluation";

const isDecisionEntry = (entry: CostMapEntry): boolean =>
  isDecisionMode(entry.mode) || (entry.supported_endpoints ?? []).some((endpoint) => DECISION_ENDPOINTS.has(endpoint));

const withoutProviderPrefix = (key: string, provider: string): string =>
  key.startsWith(`${provider}/`) ? key.slice(provider.length + 1) : key;

const parsedEntries = (costMap: unknown): ReadonlyArray<readonly [string, CostMapEntry]> => {
  const record = z.record(z.string(), z.unknown()).safeParse(costMap);
  if (!record.success) {
    return [];
  }
  return Object.entries(record.data).flatMap(([key, value]) => {
    const entry = costMapEntrySchema.safeParse(value);
    return entry.success ? [[key, entry.data] as const] : [];
  });
};

export function buildDecisionCatalog(costMap: unknown): DecisionCatalog {
  const entries = parsedEntries(costMap);
  const decisionEntries = entries.filter(([, entry]) => isDecisionEntry(entry));
  const decisionProviders = new Set(decisionEntries.map(([, entry]) => entry.litellm_provider));
  const providers = new Map(
    [...decisionProviders].map((provider) => {
      const providerEntries = entries.filter(([, entry]) => entry.litellm_provider === provider);
      const names = decisionEntries
        .filter(([, entry]) => entry.litellm_provider === provider)
        .map(([key]) => withoutProviderPrefix(key, provider));
      return [provider, { names, decisionOnly: providerEntries.every(([, entry]) => isDecisionEntry(entry)) }] as const;
    }),
  );
  return { models: new Set(decisionEntries.map(([key]) => key)), providers };
}

export const decisionModelsSublabel = (catalog: DecisionCatalog, litellmProvider: string): string | undefined => {
  const names = catalog.providers.get(litellmProvider)?.names ?? [];
  return names.length > 0 ? `Decision models: ${names.join(", ")}` : undefined;
};

export const isDecisionSelection = (
  catalog: DecisionCatalog,
  litellmProvider: string | undefined,
  selectedModels: readonly string[],
): boolean =>
  (litellmProvider !== undefined && catalog.providers.get(litellmProvider)?.decisionOnly === true) ||
  selectedModels.some((model) => catalog.models.has(model));

export const decisionProviderNames = (catalog: DecisionCatalog): readonly string[] =>
  [...catalog.providers.keys()]
    .map((provider) => getProviderLogoAndName(provider).displayName)
    .sort((a, b) => a.localeCompare(b));
