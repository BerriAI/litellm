import { z } from "zod";

export const DECISIONS_DOCS_URL = "https://docs.litellm.ai/docs/decisions";
export const DECISIONS_PROVIDERS_DOCS_URL = `${DECISIONS_DOCS_URL}#supported-providers`;
export const DECISIONS_SELF_HOSTED_DOCS_URL = `${DECISIONS_DOCS_URL}#self-hosted-laya-and-nimble`;
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

export const DECISIONS_MODE = "decisions";
const DECISION_MODE_VALUES: readonly string[] = ["evaluation", DECISIONS_MODE];

export const isDecisionMode = (mode: string | null | undefined): boolean =>
  mode !== null && mode !== undefined && DECISION_MODE_VALUES.includes(mode);

export const displayMode = (mode: string | null | undefined): string | undefined =>
  isDecisionMode(mode) ? DECISIONS_MODE : mode ?? undefined;

export const expandModeFilterValues = (modes: readonly string[]): readonly string[] =>
  modes.flatMap((mode) => (mode === DECISIONS_MODE ? DECISION_MODE_VALUES : [mode]));

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

export type DecisionRoute = "decisions" | "self-hosted" | "needs-upgrade";

const ROUTE_ORDER: Readonly<Record<DecisionRoute, number>> = { decisions: 0, "self-hosted": 1, "needs-upgrade": 2 };

const proxyDecisionProvidersSchema = z.object({ decision_model_providers: z.array(z.string()) });

export const parseProxyDecisionProviders = (settings: unknown): ReadonlySet<string> | null => {
  const parsed = proxyDecisionProvidersSchema.safeParse(settings);
  return parsed.success ? new Set(parsed.data.decision_model_providers) : null;
};

export interface DecisionGardenModel {
  readonly model: string;
  readonly provider: string;
  readonly route: DecisionRoute;
}

const decisionRoute = (entry: CostMapEntry, proxyProviders: ReadonlySet<string> | null): DecisionRoute => {
  if (proxyProviders?.has(entry.litellm_provider)) {
    return "decisions";
  }
  if (!(entry.supported_endpoints ?? ["/v1/decisions"]).includes("/v1/decisions")) {
    return "self-hosted";
  }
  return proxyProviders === null ? "decisions" : "needs-upgrade";
};

const shortestNameFirst = (a: DecisionGardenModel, b: DecisionGardenModel): number =>
  a.model.length - b.model.length || a.model.localeCompare(b.model);

const rankedRow = (row: DecisionGardenModel, index: number, rows: readonly DecisionGardenModel[]) => ({
  row,
  routeOrder: ROUTE_ORDER[row.route],
  rankInProvider: rows.slice(0, index).filter((earlier) => earlier.provider === row.provider).length,
});

export const buildDecisionGarden = (
  costMap: unknown,
  proxyProviders: ReadonlySet<string> | null,
): readonly DecisionGardenModel[] =>
  parsedEntries(costMap)
    .filter(([, entry]) => isDecisionEntry(entry))
    .map(([key, entry]) => ({
      model: key,
      provider: entry.litellm_provider,
      route: decisionRoute(entry, proxyProviders),
    }))
    .toSorted(shortestNameFirst)
    .map(rankedRow)
    .toSorted(
      (a, b) =>
        a.routeOrder - b.routeOrder ||
        a.rankInProvider - b.rankInProvider ||
        a.row.provider.localeCompare(b.row.provider),
    )
    .map(({ row }) => row);

export const isDecisionSelection = (
  catalog: DecisionCatalog,
  litellmProvider: string | undefined,
  selectedModels: readonly string[],
): boolean =>
  (litellmProvider !== undefined && catalog.providers.get(litellmProvider)?.decisionOnly === true) ||
  selectedModels.some((model) => catalog.models.has(model));
