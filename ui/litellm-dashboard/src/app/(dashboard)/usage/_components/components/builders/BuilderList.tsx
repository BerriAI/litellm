"use client";

import { useMemo } from "react";
import { AgentMark } from "../overview/TopAgents";
import { agentRowFor } from "../overview/agentCatalog";
import { formatCompactUsd } from "../overview/overviewData";
import { Panel, Segmented } from "../overview/Primitives";
import {
  builderCostPerPr,
  builderInitials,
  sortBuilders,
  type BuilderInsightBuilder,
  type BuilderSort,
} from "./builderInsightsData";
import { BuilderVerdictBadge } from "./BuilderVerdictBadge";

const SORT_OPTIONS = [
  { value: "spend", label: "Spend" },
  { value: "prs", label: "PRs" },
  { value: "efficiency", label: "Cost per PR" },
] as const satisfies readonly { value: BuilderSort; label: string }[];

export function BuilderList({
  builders,
  selectedId,
  sort,
  onSelect,
  onSortChange,
}: {
  builders: readonly BuilderInsightBuilder[];
  selectedId: string | null;
  sort: BuilderSort;
  onSelect: (id: string) => void;
  onSortChange: (sort: BuilderSort) => void;
}) {
  const sortedBuilders = useMemo(() => sortBuilders(builders, sort), [builders, sort]);

  return (
    <Panel
      title="Builders"
      subtitle="Last 30 days. Select a builder to see their verdict"
      action={<Segmented label="Builder sort" value={sort} options={SORT_OPTIONS} onChange={onSortChange} />}
    >
      <div className="divide-y divide-border/60">
        {sortedBuilders.map((builder, index) => {
          const costPerPr = builderCostPerPr(builder);
          const agents = builder.agents
            .filter((agent) => agent.spend > 0 && agent.id !== "unlabeled" && agent.id !== "browser")
            .sort((left, right) => right.spend - left.spend)
            .slice(0, 2);
          const selected = builder.id === selectedId;
          return (
            <button
              key={builder.id}
              type="button"
              onClick={() => onSelect(builder.id)}
              title={builder.tagline.replace(/`/g, "")}
              className={`grid w-full grid-cols-[1.5rem_minmax(0,1fr)_auto] items-center gap-2 rounded-lg py-2.5 text-left transition-colors sm:grid-cols-[1.5rem_minmax(0,1fr)_3rem_auto] sm:gap-3 ${
                selected ? "bg-muted/60" : "hover:bg-muted/40"
              }`}
            >
              <span className="text-sm tabular-nums text-muted-foreground">{index + 1}.</span>
              <span className="flex min-w-0 items-center gap-3">
                <span className="flex size-9 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-semibold text-muted-foreground">
                  {builderInitials(builder.name)}
                </span>
                <span className="min-w-0">
                  <span className="block truncate text-sm font-medium text-foreground">{builder.name}</span>
                  <BuilderVerdictBadge verdict={builder.verdict} label={builder.verdictLabel} />
                </span>
              </span>
              <span className="hidden items-center justify-start -space-x-1.5 sm:flex">
                {agents.map((agent) => (
                  <AgentMark key={agent.id} agent={agentRowFor(agent.id)} size="sm" />
                ))}
              </span>
              <span className="text-right">
                <span className="block text-sm tabular-nums text-foreground">
                  {costPerPr === null ? "—" : formatCompactUsd(costPerPr)}
                </span>
                <span className="block text-xs tabular-nums text-muted-foreground">
                  {builder.prs.toLocaleString()} PRs · {formatCompactUsd(builder.spend)}
                </span>
              </span>
            </button>
          );
        })}
      </div>
    </Panel>
  );
}
