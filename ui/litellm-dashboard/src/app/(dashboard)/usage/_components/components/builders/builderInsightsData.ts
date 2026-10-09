export interface BuilderInsightsWindow {
  start: string;
  end: string;
  sampleStart: string;
  sampleEnd: string;
  tz: string;
}

export interface HiddenBuilder {
  name: string;
  spend: number;
  prs: number;
}

export interface BuilderDailyEntry {
  date: string;
  spend: number;
  requests: number;
}

export interface BuilderModelEntry {
  model: string;
  spend: number;
  requests: number;
}

export interface BuilderAgentEntry {
  id: string;
  requests: number;
  spend: number;
  share: number;
}

export interface BuilderTopSession {
  spend: number;
  requests: number;
  durationMs: number;
  agent: string;
}

export interface BuilderInsightBuilder {
  id: string;
  name: string;
  verdict: BuilderVerdict;
  verdictLabel: string;
  verdictLine: string;
  email: string;
  archetype: string;
  tagline: string;
  uses: string;
  markdown: string;
  spend: number;
  requests: number;
  tokens: number;
  cacheHitRate: number;
  failed: number;
  activeDays: number;
  prsOwn: number;
  prsDevin: number;
  prs: number;
  spendPerPr: number | null;
  medianPromptTokens: number;
  p90PromptTokens: number;
  daily: readonly BuilderDailyEntry[];
  models: readonly BuilderModelEntry[];
  agents: readonly BuilderAgentEntry[];
  hoursPdt: readonly number[];
  heatPdt: readonly (readonly number[])[];
  topSessions: readonly BuilderTopSession[];
  prScopes: readonly (readonly [string, number])[];
}

export interface BuilderInsightsData {
  window: BuilderInsightsWindow;
  markdown: string;
  teamMedianCostPerPr: number;
  hidden: readonly HiddenBuilder[];
  builders: readonly BuilderInsightBuilder[];
}

export interface BuilderTotalsInput {
  spend: number;
  prs: number;
  prsDevin: number;
}

export interface TeamTotals {
  spend: number;
  prs: number;
  prsDevin: number;
  builderCount: number;
}

export type BuilderSort = "spend" | "prs" | "efficiency";
export type BuilderVerdict = "productive" | "mixed" | "expensive" | "none";

export interface BuilderSortInput {
  spend: number;
  prs: number;
  spendPerPr: number | null;
}

export const teamTotals = (builders: readonly BuilderTotalsInput[]): TeamTotals => {
  return {
    spend: builders.reduce((total, builder) => total + builder.spend, 0),
    prs: builders.reduce((total, builder) => total + builder.prs, 0),
    prsDevin: builders.reduce((total, builder) => total + builder.prsDevin, 0),
    builderCount: builders.length,
  };
};

export const sortBuilders = <T extends BuilderSortInput>(builders: readonly T[], by: BuilderSort): T[] =>
  [...builders].sort((left, right) => {
    if (by === "efficiency") {
      if (left.spendPerPr === null) return right.spendPerPr === null ? 0 : 1;
      if (right.spendPerPr === null) return -1;
      return left.spendPerPr - right.spendPerPr;
    }
    return by === "spend" ? right.spend - left.spend : right.prs - left.prs;
  });

export const aggregateModels = (models: readonly BuilderModelEntry[]): BuilderModelEntry[] =>
  [
    ...models
      .reduce<ReadonlyMap<string, BuilderModelEntry>>((totals, entry) => {
        const model = entry.model.replace(/^(openai|anthropic|bedrock|vertex_ai|azure)\//, "");
        const current = totals.get(model);
        return new Map<string, BuilderModelEntry>([
          ...totals,
          [
            model,
            {
              model,
              spend: (current?.spend ?? 0) + entry.spend,
              requests: (current?.requests ?? 0) + entry.requests,
            },
          ] as const,
        ]);
      }, new Map())
      .values(),
  ].sort((left, right) => right.spend - left.spend);

export const agentSpendShares = (agents: readonly BuilderAgentEntry[]): BuilderAgentEntry[] => {
  const totals = agents.reduce<ReadonlyMap<string, BuilderAgentEntry>>((current, agent) => {
    const entry = current.get(agent.id);
    return new Map([
      ...current,
      [
        agent.id,
        {
          id: agent.id,
          requests: (entry?.requests ?? 0) + agent.requests,
          spend: (entry?.spend ?? 0) + agent.spend,
          share: 0,
        },
      ] as const,
    ]);
  }, new Map());
  const totalSpend = [...totals.values()].reduce((sum, agent) => sum + agent.spend, 0);
  if (totalSpend <= 0) return [];
  const visible = [...totals.values()].filter((agent) => agent.spend / totalSpend >= 0.005);
  const visibleSpend = visible.reduce((sum, agent) => sum + agent.spend, 0);
  return visible
    .map((agent) => ({ ...agent, share: agent.spend / visibleSpend }))
    .sort((left, right) => right.spend - left.spend);
};

export const builderCostPerPr = (
  builder: Pick<BuilderInsightBuilder, "spend" | "prs" | "spendPerPr">,
): number | null => (builder.prs > 0 ? builder.spendPerPr ?? builder.spend / builder.prs : null);

export const builderInitials = (name: string): string => {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length > 1) return `${parts[0][0] ?? ""}${parts[parts.length - 1][0] ?? ""}`.toUpperCase();
  const first = parts[0] ?? "";
  return `${first[0]?.toUpperCase() ?? ""}${first[1]?.toLowerCase() ?? ""}`;
};

export const builderVerdictDotClass = (verdict: BuilderVerdict): string =>
  ({
    productive: "bg-emerald-500",
    mixed: "bg-amber-500",
    expensive: "bg-red-500",
    none: "bg-muted-foreground/40",
  })[verdict];
