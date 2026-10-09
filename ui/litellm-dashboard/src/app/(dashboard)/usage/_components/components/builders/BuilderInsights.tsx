"use client";

import { useEffect, useMemo, useState } from "react";
import { ChartSkeleton, Panel, Stat } from "../overview/Primitives";
import { formatCompactUsd } from "../overview/overviewData";
import { agentRowFor, type AgentRow } from "../overview/agentCatalog";
import { AgentMark } from "../overview/TopAgents";
import { BuilderDetail } from "./BuilderDetail";
import { BuilderList } from "./BuilderList";
import { MarkdownFile } from "./MarkdownFile";
import { sortBuilders, teamAgents, teamTotals, type BuilderSort } from "./builderInsightsData";
import { useBuilderInsights } from "./useBuilderInsights";

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

function TeamAgents({
  builders,
}: {
  builders: Parameters<typeof teamAgents>[0];
}) {
  const agents = teamAgents(builders);
  return (
    <Panel title="Agents" subtitle="Share of spend by User-Agent, 7-day sample of SpendLogs">
      <div className="grid gap-3">
        {agents.map((agent) => {
          const row: AgentRow = agentRowFor(agent.id);
          return (
            <div key={agent.id} className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3">
              <AgentMark agent={row} size="sm" />
              <div className="min-w-0">
                <div className="flex items-center justify-between gap-3">
                  <span className="truncate text-sm font-medium">{row.label}</span>
                  <span className="shrink-0 text-xs tabular-nums text-muted-foreground">{percent(agent.share)}</span>
                </div>
                <p className="truncate text-xs text-muted-foreground">{row.description}</p>
                <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-muted">
                  <span className="block h-full rounded-full bg-[var(--chart-1)]" style={{ width: `${agent.share * 100}%` }} />
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm tabular-nums">{formatCompactUsd(agent.spend)}</div>
                <div className="text-xs tabular-nums text-muted-foreground">{agent.builders} builders</div>
              </div>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}

export default function BuilderInsights() {
  const { data, isPending, isError } = useBuilderInsights();
  const [sort, setSort] = useState<BuilderSort>("spend");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const selectedBuilders = useMemo(() => (data ? sortBuilders(data.builders, "spend") : []), [data]);
  const selectedBuilder = data?.builders.find((builder) => builder.id === selectedId) ?? selectedBuilders[0];
  const totals = useMemo(() => teamTotals(data?.builders ?? []), [data]);

  useEffect(() => {
    if (!selectedId || !data) return;
    const first = data.builders[0]?.id;
    if (first === selectedId) return;
    document.getElementById("builder-insights-detail")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [data, selectedId]);

  if (isPending) {
    return (
      <div className="grid gap-3">
        <ChartSkeleton className="h-28" />
        <div className="grid gap-3 lg:grid-cols-5">
          <ChartSkeleton className="h-96 lg:col-span-3" />
          <ChartSkeleton className="h-96 lg:col-span-2" />
        </div>
      </div>
    );
  }

  if (isError || !data) return <p className="py-10 text-center text-sm text-muted-foreground">Builder Insights is unavailable</p>;

  return (
    <div className="grid gap-3">
      <div className="grid grid-cols-1 overflow-hidden rounded-xl border bg-card sm:grid-cols-2 lg:grid-cols-5 lg:divide-x">
        <div className="border-b px-5 py-4 sm:border-b-0">
          <Stat label="Team spend" value={formatCompactUsd(totals.spend)} hint={`${totals.builderCount} builders · 30 days`} />
        </div>
        <div className="border-b px-5 py-4 sm:border-b-0">
          <Stat label="Merged PRs" value={totals.prs.toLocaleString()} hint={`${totals.prsDevin} with Devin`} />
        </div>
        <div className="border-b px-5 py-4 lg:border-b-0">
          <Stat
            label="Median cost / PR"
            value={totals.medianSpendPerPr === null ? "—" : formatCompactUsd(totals.medianSpendPerPr)}
          />
        </div>
        <div className="border-b px-5 py-4 sm:border-b-0">
          <Stat label="Top 3 builders" value={percent(totals.top3Share)} hint="of team spend" />
        </div>
        <div className="px-5 py-4">
          <Stat label="Prompt cache hit" value={percent(totals.cacheHitRate)} />
        </div>
      </div>
      <div className="grid gap-3 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <BuilderList
            builders={data.builders}
            selectedId={selectedBuilder?.id ?? null}
            sort={sort}
            onSelect={setSelectedId}
            onSortChange={setSort}
          />
        </div>
        <div className="lg:col-span-2">
          <MarkdownFile filename="team-insights.md" markdown={data.markdown} />
        </div>
      </div>
      <TeamAgents builders={data.builders} />
      {selectedBuilder && (
        <BuilderDetail
          builder={selectedBuilder}
          sampleStart={data.window.sampleStart}
          sampleEnd={data.window.sampleEnd}
          detailId="builder-insights-detail"
        />
      )}
      <p className="text-xs text-muted-foreground">
        Built from SpendLogs aggregates and merged PR counts on litellm main. No prompts or code are read.{" "}
        {data.hidden.length} builders with under $100 attributed spend are hidden.
      </p>
    </div>
  );
}
