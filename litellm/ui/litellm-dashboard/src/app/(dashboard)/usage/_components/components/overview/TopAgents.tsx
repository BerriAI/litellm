"use client";

import { useTerminology } from "@/contexts/TerminologyContext";
import React, { useMemo } from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { ArrowRight, Bot, Monitor, Terminal } from "lucide-react";
import { stackedUsageColor } from "@/components/shared/charts";
import { Logo } from "@/components/molecules/logo/Logo";
import type { DailyData } from "@/components/UsagePage/types";
import { cn } from "@/lib/cva.config";
import { ChartSkeleton, Panel, PANEL_INSET_X } from "./Primitives";
import { agentDailyData, agentDisplayLabel, topAgents, type AgentRow, type TagSummaryRow } from "./agentCatalog";
import TrendChart, { GRANULARITY_ADJECTIVE, METRIC_NOUN, PartialPeriodNote } from "./TrendChart";
import { COMPACT_FORMAT } from "./trendFormat";
import type { TrendSelection } from "./trendSelection";
import { usageTrend, type DayRange, type TrendMetric } from "./usageTrendData";

const ROWS = 10;
const ONE_DECIMAL = { minimumFractionDigits: 1, maximumFractionDigits: 1 } as const;
const VALUE_LABEL = {
  tokens: "usage.topAgents.value.tokens",
  spend: "usage.topAgents.value.spend",
  requests: "usage.topAgents.value.requests",
} as const satisfies Readonly<Record<TrendMetric, ParseKeys>>;

const agentValue = (agent: AgentRow, metric: TrendMetric): number => agent[metric];

const byKindThenMetric =
  (metric: TrendMetric) =>
  (a: AgentRow, b: AgentRow): number =>
    Number(a.kind !== "agent") - Number(b.kind !== "agent") || agentValue(b, metric) - agentValue(a, metric);
/** Below this share of tokens carrying a User-Agent, say so rather than imply the list is the whole picture. */
const COVERAGE_NOTE_BELOW = 0.5;

/** Brand-tinted monograms for runtimes that have no logo file, so a top spender never shows a blank tile. */
const MONOGRAMS: Readonly<Record<string, { text: string; className: string }>> = {
  python: { text: "Py", className: "bg-[#3776ab] text-[#ffd43b]" },
  node: { text: "JS", className: "bg-[#3c873a] text-white" },
  curl: { text: "cu", className: "bg-[#073551] text-white" },
};
const CLIENT_SURFACES: Readonly<Partial<Record<string, typeof Monitor>>> = {
  "claude-desktop": Monitor,
  "claude-code": Terminal,
};

function AgentGlyph({ agent }: { agent: Pick<AgentRow, "id" | "label" | "logo"> }) {
  const { t } = useTranslation();
  const label = agentDisplayLabel(agent.label, t);
  if (agent.logo) return <Logo src={agent.logo} label={label} className="size-6 object-contain" />;
  const monogram = MONOGRAMS[agent.id];
  if (monogram) {
    return (
      <span
        className={cn(
          "flex size-full items-center justify-center text-xs font-semibold tracking-tight",
          monogram.className,
        )}
      >
        {monogram.text}
      </span>
    );
  }
  return <span className="text-sm font-semibold uppercase text-muted-foreground">{label.charAt(0)}</span>;
}

export function AgentMark({ agent }: { agent: Pick<AgentRow, "id" | "label" | "logo"> }) {
  const Surface = CLIENT_SURFACES[agent.id];
  return (
    <span className="relative inline-flex size-9 shrink-0 items-center justify-center overflow-hidden rounded-lg border bg-background">
      <AgentGlyph agent={agent} />
      {Surface && (
        <span className="absolute top-0 right-0 rounded-bl border-b border-l bg-background p-0.5 text-foreground">
          <Surface aria-hidden="true" className="size-2.5" />
        </span>
      )}
    </span>
  );
}

function AgentItem({
  agent,
  rank,
  share,
  color,
  metric,
  onOpen,
}: {
  agent: AgentRow;
  rank: number;
  share: number;
  color: string | undefined;
  metric: TrendMetric;
  onOpen?: (agent: AgentRow) => void;
}) {
  const { t } = useTranslation();
  const term = useTerminology();
  const openRow = onOpen && agent.tags.length > 0 ? () => onOpen(agent) : undefined;
  const content = (
    <>
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
        <span className="block truncate text-sm font-medium text-foreground">{agentDisplayLabel(agent.label, t)}</span>
        {openRow ? (
          <span className="inline-flex items-center gap-1 text-xs text-muted-foreground group-hover:text-foreground">
            {t("usage.topAgents.viewActivity")}
            <ArrowRight aria-hidden="true" className="size-3 transition-transform group-hover:translate-x-0.5" />
          </span>
        ) : (
          <span className="block truncate text-xs text-muted-foreground">{term(agent.description)}</span>
        )}
      </span>
      <span className="text-right">
        <span className="block text-sm tabular-nums text-foreground">
          {t(VALUE_LABEL[metric], { value: COMPACT_FORMAT[metric](agentValue(agent, metric)) })}
        </span>
        <span className="block text-xs tabular-nums text-muted-foreground">
          {t("usage.topAgents.share", { percent: t("format.number", { val: share, ...ONE_DECIMAL }) })}
        </span>
      </span>
    </>
  );
  const rowClassName =
    "group grid w-full grid-cols-[1.5rem_auto_minmax(0,1fr)_auto] items-center gap-3 py-2.5 text-left";
  return (
    <li>
      {openRow ? (
        <button
          type="button"
          aria-label={t("usage.topAgents.viewAgentUsage", { agent: agentDisplayLabel(agent.label, t) })}
          onClick={openRow}
          className={cn(
            rowClassName,
            "cursor-pointer rounded-md transition-colors hover:bg-muted/50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",
          )}
        >
          {content}
        </button>
      ) : (
        <div className={rowClassName}>{content}</div>
      )}
    </li>
  );
}

function TopAgentsBody({ loading, empty, children }: { loading: boolean; empty: boolean; children: React.ReactNode }) {
  const { t } = useTranslation();
  if (loading) return <ChartSkeleton className="mx-5 h-64 w-auto" />;
  if (empty) return <p className="py-10 text-center text-xs text-muted-foreground">{t("usage.topAgents.empty")}</p>;
  return <>{children}</>;
}

interface TopAgentsProps {
  rows: readonly TagSummaryRow[] | undefined;
  daily: readonly DailyData[];
  loading: boolean;
  totalTokens: number;
  range: DayRange | null;
  selection: TrendSelection;
  onOpenAgent?: (agent: AgentRow) => void;
}

/**
 * Clients are recognised only from each request's `User-Agent:` tag by `agentCatalog`; traffic
 * without one stays unattributed. Each day's tags are folded into clients before re-bucketing,
 * so a bare product tag and its versioned tags are never counted twice in a week or month.
 */
export default function TopAgents({
  rows,
  daily,
  loading,
  totalTokens,
  range,
  selection,
  onOpenAgent,
}: TopAgentsProps) {
  const { t } = useTranslation();
  const { metric, granularity } = selection;
  const allAgents = useMemo(() => topAgents(rows ?? []), [rows]);
  const agents = useMemo(() => [...allAgents].sort(byKindThenMetric(metric)).slice(0, ROWS), [allAgents, metric]);
  const colorByAgent = useMemo(
    () => new Map(allAgents.map((agent, index) => [agent.label, stackedUsageColor(index)])),
    [allAgents],
  );
  const agentDays = useMemo(() => agentDailyData(daily), [daily]);
  const trend = useMemo(() => {
    if (range === null) return null;
    const input = {
      results: agentDays,
      range,
      granularity,
      colorByModel: colorByAgent,
      dimension: "models",
      rankBy: metric,
    } as const;
    return usageTrend(input);
  }, [agentDays, range, granularity, colorByAgent, metric]);
  const series = trend?.series[metric];
  const colorOf = (label: string) => {
    const index = series ? series.labels.indexOf(label) : -1;
    return index === -1 || series === undefined ? undefined : series.colors[index];
  };

  const listTotal = agents.reduce((sum, agent) => sum + agentValue(agent, metric), 0);
  const attributedTokens = allAgents.reduce((sum, agent) => sum + agent.tokens, 0);
  const coverage = totalTokens > 0 ? Math.min(attributedTokens / totalTokens, 1) : 1;
  const caption = t("usage.topAgents.caption", {
    granularity: t(GRANULARITY_ADJECTIVE[granularity]),
    metric: t(METRIC_NOUN[metric]),
  });
  const subtitle =
    coverage < COVERAGE_NOTE_BELOW
      ? t("usage.topAgents.subtitleLowCoverage", { caption, percent: Math.round(coverage * 100) })
      : t("usage.topAgents.subtitle", { caption });
  const half = Math.ceil(agents.length / 2);

  return (
    <Panel
      icon={Bot}
      title={t("usage.topAgents.title")}
      subtitle={subtitle}
      wrapSubtitle
      testId="top-agents"
      bodyClassName="px-0 pt-4 pb-0"
    >
      <TopAgentsBody loading={loading} empty={agents.length === 0}>
        {trend !== null && series !== undefined && (
          <div className="grid gap-2 px-5 pb-2">
            <TrendChart
              trend={trend}
              selection={selection}
              labels={series.labels.map((label) => agentDisplayLabel(label, t))}
              ariaLabel={caption}
              className="h-64"
            />
            <PartialPeriodNote trend={trend} granularity={granularity} />
          </div>
        )}
        <div className={cn("grid gap-x-10 border-t py-2 md:grid-cols-2", PANEL_INSET_X)}>
          {[agents.slice(0, half), agents.slice(half)].map((column, columnIndex) => (
            <ol key={columnIndex} className="divide-y divide-border/60">
              {column.map((agent, index) => (
                <AgentItem
                  key={agent.id}
                  agent={agent}
                  rank={columnIndex * half + index + 1}
                  share={listTotal > 0 ? (agentValue(agent, metric) / listTotal) * 100 : 0}
                  color={colorOf(agent.label)}
                  metric={metric}
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
