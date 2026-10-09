import type { SeriesDay } from "../overview/overviewData";

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
  hidden: readonly HiddenBuilder[];
  builders: readonly BuilderInsightBuilder[];
}

export interface BuilderTotalsInput {
  spend: number;
  prs: number;
  prsDevin: number;
  spendPerPr: number | null;
  tokens: number;
  cacheHitRate: number;
}

export interface TeamTotals {
  spend: number;
  prs: number;
  prsDevin: number;
  builderCount: number;
  medianSpendPerPr: number | null;
  top3Share: number;
  cacheHitRate: number;
}

export type BuilderSort = "spend" | "prs" | "efficiency";

export interface BuilderSortInput {
  spend: number;
  prs: number;
  spendPerPr: number | null;
}

export interface TeamAgent {
  id: string;
  spend: number;
  requests: number;
  builders: number;
  share: number;
}

const median = (values: readonly number[]): number | null => {
  const sorted = [...values].sort((left, right) => left - right);
  if (sorted.length === 0) return null;
  const midpoint = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[midpoint - 1] + sorted[midpoint]) / 2 : sorted[midpoint];
};

export const teamTotals = (
  builders: readonly (BuilderTotalsInput & Pick<BuilderInsightBuilder, "prs">)[],
): TeamTotals => {
  const spend = builders.reduce((total, builder) => total + builder.spend, 0);
  const prs = builders.reduce((total, builder) => total + builder.prs, 0);
  const prsDevin = builders.reduce((total, builder) => total + builder.prsDevin, 0);
  const tokens = builders.reduce((total, builder) => total + builder.tokens, 0);
  const cacheHitTokens = builders.reduce((total, builder) => total + builder.tokens * builder.cacheHitRate, 0);
  const builderSpendPerPr = builders.flatMap((builder) =>
    builder.prs > 0 ? [builder.spendPerPr ?? builder.spend / builder.prs] : [],
  );
  const topThreeSpend = [...builders]
    .sort((left, right) => right.spend - left.spend)
    .slice(0, 3)
    .reduce((total, builder) => total + builder.spend, 0);

  return {
    spend,
    prs,
    prsDevin,
    builderCount: builders.length,
    medianSpendPerPr: median(builderSpendPerPr),
    top3Share: spend > 0 ? topThreeSpend / spend : 0,
    cacheHitRate: tokens > 0 ? cacheHitTokens / tokens : 0,
  };
};

export const teamAgents = (
  builders: readonly (Pick<BuilderInsightBuilder, "spend" | "agents">)[],
): TeamAgent[] => {
  const perAgent = builders
    .flatMap((builder) => builder.agents)
    .reduce<ReadonlyMap<string, Omit<TeamAgent, "share">>>((totals, agent) => {
      const current = totals.get(agent.id) ?? { id: agent.id, spend: 0, requests: 0, builders: 0 };
      return new Map([
        ...totals,
        [
          agent.id,
          {
            id: agent.id,
            spend: current.spend + agent.spend,
            requests: current.requests + agent.requests,
            builders: current.builders + Number(agent.share >= 0.05),
          },
        ],
      ]);
    }, new Map());
  const totalAgentSpend = [...perAgent.values()].reduce((total, agent) => total + agent.spend, 0);

  return [...perAgent.values()]
    .map((agent) => ({ ...agent, share: totalAgentSpend > 0 ? agent.spend / totalAgentSpend : 0 }))
    .sort((left, right) => right.spend - left.spend);
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

export const dailySeries = (
  builder: Pick<BuilderInsightBuilder, "daily">,
): (SeriesDay & Pick<BuilderDailyEntry, "spend" | "requests">)[] =>
  [...builder.daily]
    .sort((left, right) => left.date.localeCompare(right.date))
    .map((day) => {
      const parsed = new Date(`${day.date}T00:00:00`);
      const label = Number.isNaN(parsed.getTime())
        ? day.date
        : parsed.toLocaleDateString("en-US", { month: "short", day: "numeric" });
      return { date: day.date, label, spend: day.spend, requests: day.requests };
    });

export const aggregateModels = (models: readonly BuilderModelEntry[]): BuilderModelEntry[] =>
  [...models.reduce<ReadonlyMap<string, BuilderModelEntry>>((totals, entry) => {
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
  }, new Map()).values()].sort((left, right) => right.spend - left.spend);

export const builderInitials = (name: string): string => {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length > 1) return `${parts[0][0] ?? ""}${parts[parts.length - 1][0] ?? ""}`.toUpperCase();
  const first = parts[0] ?? "";
  return `${first[0]?.toUpperCase() ?? ""}${first[1]?.toLowerCase() ?? ""}`;
};
