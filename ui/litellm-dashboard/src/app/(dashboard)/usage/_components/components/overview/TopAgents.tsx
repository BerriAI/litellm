"use client";

import React, { useMemo, useState } from "react";
import { ArrowRight, Bot } from "lucide-react";
import { StackedUsageChart } from "@/components/shared/charts";
import { Logo } from "@/components/molecules/logo/Logo";
import type { DailyData } from "@/components/UsagePage/types";
import { cn } from "@/lib/cva.config";
import {
  bucketTotals,
  labelForDate,
  formatCompact,
  formatCompactUsd,
  formatMetricValue,
  seriesBy,
  type UsageMetric,
} from "./overviewData";
import { ChartSkeleton, Panel, PANEL_INSET_X, Segmented } from "./Primitives";
import { agentDailyData, topAgents, type AgentRow, type TagSummaryRow } from "./agentCatalog";

const ROWS = 10;
const CHART_SERIES = 6;
/** Below this share of tokens carrying a User-Agent, say so rather than imply the list is the whole picture. */
const COVERAGE_NOTE_BELOW = 0.5;
const METRICS = [
  { value: "tokens", label: "Tokens" },
  { value: "spend", label: "Spend" },
  { value: "requests", label: "Requests" },
] as const satisfies readonly { value: UsageMetric; label: string }[];

/** Brand-tinted monograms for runtimes that have no logo file, so a top spender never shows a blank tile. */
const MONOGRAMS: Readonly<Record<string, { text: string; className: string }>> = {
  python: { text: "Py", className: "bg-[#3776ab] text-[#ffd43b]" },
  node: { text: "JS", className: "bg-[#3c873a] text-white" },
  curl: { text: "cu", className: "bg-[#073551] text-white" },
};

/** Claude Code's mark: a white spark on Claude terracotta. Drawn here since no Claude asset ships in the repo. */
function ClaudeSpark({ size }: { size: "sm" | "md" }) {
  const rays = Array.from({ length: 12 }, (_, i) => i * 30);
  return (
    <span className="flex size-full items-center justify-center bg-[#d97757]">
      <svg viewBox="0 0 24 24" className={cn(size === "sm" ? "size-3.5" : "size-5")} aria-hidden="true">
        {rays.map((angle) => (
          <rect
            key={angle}
            x="11.1"
            y="3"
            width="1.8"
            height="9"
            rx="0.9"
            fill="white"
            transform={`rotate(${angle} 12 12)`}
          />
        ))}
      </svg>
    </span>
  );
}

function AgentGlyph({ agent, size }: { agent: AgentRow; size: "sm" | "md" }) {
  if (agent.id === "claude-code") return <ClaudeSpark size={size} />;
  if (agent.logo)
    return <Logo src={agent.logo} label={agent.label} className={cn(size === "sm" ? "size-3.5" : "size-5")} />;
  const monogram = MONOGRAMS[agent.id];
  if (monogram) {
    return (
      <span
        className={cn(
          "flex size-full items-center justify-center font-semibold tracking-tight",
          size === "sm" ? "text-[9px]" : "text-xs",
          monogram.className,
        )}
      >
        {monogram.text}
      </span>
    );
  }
  return (
    <span className={cn("font-semibold uppercase text-muted-foreground", size === "sm" ? "text-xs" : "text-sm")}>
      {agent.label.charAt(0)}
    </span>
  );
}

export function AgentMark({ agent, size = "md" }: { agent: AgentRow; size?: "sm" | "md" }) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center justify-center overflow-hidden border bg-background",
        size === "sm" ? "size-6 rounded-md" : "size-9 rounded-lg",
      )}
    >
      <AgentGlyph agent={agent} size={size} />
    </span>
  );
}

function AgentItem({
  agent,
  rank,
  share,
  color,
  onOpen,
}: {
  agent: AgentRow;
  rank: number;
  share: number;
  color: string | undefined;
  onOpen?: (agent: AgentRow) => void;
}) {
  return (
    <li className="group grid grid-cols-[1.5rem_auto_minmax(0,1fr)_auto] items-center gap-3 py-2.5">
      <span className="text-sm tabular-nums text-muted-foreground">{rank}.</span>
      <span className="relative">
        <AgentMark agent={agent} />
        {color && (
          <span
            aria-hidden="true"
            className="absolute -right-0.5 -bottom-0.5 size-2.5 rounded-full ring-2 ring-card"
            style={{ backgroundColor: color }}
          />
        )}
      </span>
      <span className="min-w-0">
        <span className="block truncate text-sm font-medium text-foreground">{agent.label}</span>
        {onOpen ? (
          <button
            type="button"
            onClick={() => onOpen(agent)}
            className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
          >
            View activity
            <ArrowRight aria-hidden="true" className="size-3 transition-transform group-hover:translate-x-0.5" />
          </button>
        ) : (
          <span className="block truncate text-xs text-muted-foreground">{agent.description}</span>
        )}
      </span>
      <span className="text-right">
        <span className="block text-sm tabular-nums text-foreground">{formatCompact(agent.tokens)} tokens</span>
        <span className="block text-xs tabular-nums text-muted-foreground">
          {formatCompactUsd(agent.spend)} · {share.toFixed(1)}%
        </span>
      </span>
    </li>
  );
}

function TopAgentsBody({ loading, empty, children }: { loading: boolean; empty: boolean; children: React.ReactNode }) {
  if (loading) return <ChartSkeleton className="mx-5 h-64 w-auto" />;
  if (empty) return <p className="py-10 text-center text-xs text-muted-foreground">No agent traffic in this range</p>;
  return <>{children}</>;
}

/**
 * The agents and clients driving traffic, read from the proxy's `User-Agent:` tags: a daily stacked
 * chart (like Top models) over a ranked list. Named agents lead the list; SDKs and scripts follow,
 * so agents are not buried under raw HTTP clients that can outspend them. Each row opens User Agent
 * Activity filtered to that agent's tags.
 */
export default function TopAgents({
  rows,
  daily,
  loading,
  totalTokens,
  onOpenAgent,
}: {
  rows: readonly TagSummaryRow[] | undefined;
  /** Tag daily activity; each day's tags are folded into agents for the chart. */
  daily: readonly DailyData[];
  loading: boolean;
  /** All tokens in the range, so the panel can say how much traffic carried no User-Agent to attribute. */
  totalTokens: number;
  onOpenAgent?: (agent: AgentRow) => void;
}) {
  const [metric, setMetric] = useState<UsageMetric>("tokens");
  const allAgents = useMemo(() => topAgents(rows ?? []), [rows]);
  const agents = useMemo(
    () =>
      [
        ...allAgents.filter((agent) => agent.kind === "agent"),
        ...allAgents.filter((agent) => agent.kind === "sdk"),
      ].slice(0, ROWS),
    [allAgents],
  );
  const series = useMemo(() => seriesBy(agentDailyData(daily), "models", metric, CHART_SERIES), [daily, metric]);
  const totalsByDay = useMemo(() => bucketTotals(series), [series]);
  const colorOf = (label: string) => {
    const index = series.labels.indexOf(label);
    return index === -1 ? undefined : series.colors[index];
  };

  const totalShare = agents.reduce((sum, agent) => sum + agent.tokens, 0);
  const attributedTokens = allAgents.reduce((sum, agent) => sum + agent.tokens, 0);
  const coverage = totalTokens > 0 ? Math.min(attributedTokens / totalTokens, 1) : 1;
  const subtitle =
    coverage < COVERAGE_NOTE_BELOW
      ? `Daily usage by agent. ${Math.round(coverage * 100)}% of tokens came with a User-Agent; the rest is unattributed`
      : "Daily usage by agent, from each request's User-Agent";
  const half = Math.ceil(agents.length / 2);

  return (
    <Panel
      icon={Bot}
      title="Top agents"
      subtitle={subtitle}
      testId="top-agents"
      action={<Segmented label="Agent metric" value={metric} options={METRICS} onChange={setMetric} />}
      bodyClassName="px-0 pt-4 pb-0"
    >
      <TopAgentsBody loading={loading} empty={agents.length === 0}>
        <div className="px-2 pb-2">
          <StackedUsageChart
            data={series.data}
            series={series.keys}
            labels={series.labels}
            colors={series.colors}
            xKey="date"
            xLabel={(date) => labelForDate(series, date)}
            format={(value) => formatMetricValue(value, metric)}
            totalFor={(date) => totalsByDay.get(date)}
            className="h-64"
          />
        </div>
        <div className={cn("grid gap-x-10 border-t py-2 md:grid-cols-2", PANEL_INSET_X)}>
          {[agents.slice(0, half), agents.slice(half)].map((column, columnIndex) => (
            <ol key={columnIndex} className="divide-y divide-border/60">
              {column.map((agent, index) => (
                <AgentItem
                  key={agent.id}
                  agent={agent}
                  rank={columnIndex * half + index + 1}
                  share={totalShare > 0 ? (agent.tokens / totalShare) * 100 : 0}
                  color={colorOf(agent.label)}
                  onOpen={onOpenAgent}
                />
              ))}
            </ol>
          ))}
        </div>
      </TopAgentsBody>
    </Panel>
  );
}
