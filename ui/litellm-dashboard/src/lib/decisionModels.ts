import { z } from "zod";

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
  readonly decisionOnly: boolean;
}

export interface DecisionCatalog {
  readonly models: ReadonlySet<string>;
  readonly providers: ReadonlyMap<string, ProviderDecisionModels>;
}

export const EMPTY_DECISION_CATALOG: DecisionCatalog = { models: new Set(), providers: new Map() };

export const DECISIONS_MODE_LABEL = "decisions";
export const DECISION_MODE_VALUES: readonly string[] = ["evaluation", DECISIONS_MODE_LABEL];

export const isDecisionMode = (mode: string | null | undefined): boolean =>
  mode !== null && mode !== undefined && DECISION_MODE_VALUES.includes(mode);

export const displayMode = (mode: string): string => (isDecisionMode(mode) ? DECISIONS_MODE_LABEL : mode);

export const expandModeFilterValues = (modes: readonly string[]): readonly string[] =>
  modes.flatMap((mode) => (mode === DECISIONS_MODE_LABEL ? DECISION_MODE_VALUES : [mode]));

const isDecisionEntry = (entry: CostMapEntry): boolean =>
  isDecisionMode(entry.mode) || (entry.supported_endpoints ?? []).some((endpoint) => DECISION_ENDPOINTS.has(endpoint));

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
      return [provider, { decisionOnly: providerEntries.every(([, entry]) => isDecisionEntry(entry)) }] as const;
    }),
  );
  return { models: new Set(decisionEntries.map(([key]) => key)), providers };
}

export interface DecisionGardenModel {
  readonly model: string;
  readonly provider: string;
}

const isHiddenBillingRow = (key: string): boolean => key.includes("/decisions/");

export const buildDecisionGarden = (costMap: unknown): readonly DecisionGardenModel[] =>
  parsedEntries(costMap)
    .filter(([key, entry]) => isDecisionMode(entry.mode) && !isHiddenBillingRow(key))
    .map(([key, entry]) => ({ model: key, provider: entry.litellm_provider }))
    .sort((a, b) => a.provider.localeCompare(b.provider) || a.model.localeCompare(b.model));

export const isDecisionSelection = (
  catalog: DecisionCatalog,
  litellmProvider: string | undefined,
  selectedModels: readonly string[],
): boolean =>
  (litellmProvider !== undefined && catalog.providers.get(litellmProvider)?.decisionOnly === true) ||
  selectedModels.some((model) => catalog.models.has(model));
