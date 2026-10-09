"use client";

import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";
import { Badge } from "@/components/ui/badge";
import { AgentMark } from "../overview/TopAgents";
import { agentRowFor } from "../overview/agentCatalog";
import { formatCompact, formatCompactUsd, formatUsd } from "../overview/overviewData";
import { Panel, Stat } from "../overview/Primitives";
import { ActivityHeatmap } from "./ActivityHeatmap";
import { dailySeries, type BuilderInsightBuilder } from "./builderInsightsData";
import { MarkdownFile } from "./MarkdownFile";
import { stackedUsageColor } from "@/components/shared/charts";

const perPr = (builder: BuilderInsightBuilder) =>
  builder.spendPerPr ?? (builder.prs > 0 ? builder.spend / builder.prs : null);

const initials = (name: string) =>
  name
    .split(/\s+/)
    .map((part) => part[0] ?? "")
    .join("")
    .slice(0, 2)
    .toUpperCase();

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

function DailySpend({ builder }: { builder: BuilderInsightBuilder }) {
  const data = dailySeries(builder);
  const config = { spend: { label: "Spend", color: "var(--chart-1)" } } satisfies ChartConfig;
  return (
    <Panel title="Daily spend">
      <ChartContainer config={config} className="aspect-auto h-40 w-full">
        <BarChart data={data} margin={{ left: 0, right: 0, top: 4, bottom: 0 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="date" tickLine={false} axisLine={false} minTickGap={24} tickFormatter={(value) => String(value).slice(5)} />
          <YAxis tickLine={false} axisLine={false} width={42} tickFormatter={(value) => formatCompactUsd(Number(value))} />
          <ChartTooltip
            content={
              <ChartTooltipContent
                labelFormatter={(value) => String(value)}
                formatter={(value) => <span className="font-medium tabular-nums">{formatUsd(Number(value))}</span>}
              />
            }
          />
          <Bar dataKey="spend" fill="var(--chart-1)" radius={[2, 2, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ChartContainer>
    </Panel>
  );
}

function AgentMix({ builder }: { builder: BuilderInsightBuilder }) {
  const agents = [...builder.agents].filter((agent) => agent.spend > 0).sort((left, right) => right.spend - left.spend);
  const total = agents.reduce((sum, agent) => sum + agent.spend, 0);
  return (
    <Panel title="Agent mix">
      <div className="flex h-2 overflow-hidden rounded-full bg-muted">
        {agents.map((agent, index) => (
          <span
            key={agent.id}
            className="h-full"
            style={{ width: `${total > 0 ? (agent.spend / total) * 100 : 0}%`, backgroundColor: stackedUsageColor(index) }}
          />
        ))}
      </div>
      <div className="mt-3 grid gap-2">
        {agents.map((agent, index) => (
          <div key={agent.id} className="flex items-center gap-2 text-xs">
            <AgentMark agent={agentRowFor(agent.id)} size="sm" />
            <span className="min-w-0 flex-1 truncate">{agentRowFor(agent.id).label}</span>
            <span className="tabular-nums text-muted-foreground">
              {percent(total > 0 ? agent.spend / total : 0)}
            </span>
            <span className="size-2 rounded-sm" style={{ backgroundColor: stackedUsageColor(index) }} />
          </div>
        ))}
      </div>
    </Panel>
  );
}

function ModelMix({ builder }: { builder: BuilderInsightBuilder }) {
  const models = [...builder.models].sort((left, right) => right.spend - left.spend).slice(0, 5);
  return (
    <Panel title="Model mix">
      <div className="grid gap-3">
        {models.map((model) => (
          <div key={model.model} className="grid gap-1">
            <div className="flex items-center justify-between gap-3 text-xs">
              <span className="min-w-0 truncate font-mono">{model.model}</span>
              <span className="shrink-0 tabular-nums text-muted-foreground">{formatCompactUsd(model.spend)}</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-muted">
              <span
                className="block h-full rounded-full bg-[var(--chart-1)]"
                style={{ width: `${builder.spend > 0 ? (model.spend / builder.spend) * 100 : 0}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function PricestSessions({ builder }: { builder: BuilderInsightBuilder }) {
  const sessions = [...builder.topSessions].sort((left, right) => right.spend - left.spend).slice(0, 3);
  return (
    <Panel title="Priciest sessions">
      <div className="divide-y divide-border/60">
        {sessions.map((session, index) => (
          <div key={`${session.agent}-${session.spend}-${index}`} className="flex items-center gap-2 py-2 first:pt-0 last:pb-0">
            <AgentMark agent={agentRowFor(session.agent)} size="sm" />
            <span className="min-w-0 flex-1 truncate text-xs">{agentRowFor(session.agent).label}</span>
            <span className="text-right">
              <span className="block text-xs font-medium tabular-nums">{formatCompactUsd(session.spend)}</span>
              <span className="block text-[11px] tabular-nums text-muted-foreground">
                {session.requests.toLocaleString()} requests · {(session.durationMs / 3_600_000).toFixed(1)}h
              </span>
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
}

export function BuilderDetail({
  builder,
  sampleStart,
  sampleEnd,
  detailId,
}: {
  builder: BuilderInsightBuilder;
  sampleStart: string;
  sampleEnd: string;
  detailId: string;
}) {
  const costPerPr = perPr(builder);
  return (
    <section id={detailId} className="scroll-mt-4 space-y-3">
      <div className="flex flex-wrap items-center gap-4 rounded-xl border bg-card p-5">
        <span className="flex size-14 shrink-0 items-center justify-center rounded-full bg-muted text-base font-semibold text-muted-foreground">
          {initials(builder.name)}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-semibold tracking-tight">{builder.name}</h2>
            <Badge variant="outline" className="text-xs">
              {builder.archetype}
            </Badge>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">{builder.tagline}</p>
          <p className="mt-1 text-xs text-muted-foreground">{builder.email}</p>
        </div>
      </div>
      <div className="grid grid-cols-2 overflow-hidden rounded-xl border bg-card sm:grid-cols-4 lg:grid-cols-7 lg:divide-x">
        {[
          <Stat key="spend" label="Spend" value={formatCompactUsd(builder.spend)} />,
          <Stat key="requests" label="Requests" value={builder.requests.toLocaleString()} />,
          <Stat key="days" label="Active days" value={builder.activeDays.toLocaleString()} />,
          <Stat key="prs" label="Merged PRs" value={builder.prs.toLocaleString()} hint={`${builder.prsDevin} with Devin`} />,
          <Stat key="cost" label="Cost / PR" value={costPerPr === null ? "—" : formatCompactUsd(costPerPr)} />,
          <Stat key="prompt" label="Median prompt" value={formatCompact(builder.medianPromptTokens)} />,
          <Stat key="cache" label="Cache hit" value={percent(builder.cacheHitRate)} />,
        ].map((stat) => (
          <div key={stat.key} className="border-b px-4 py-4 last:border-b-0 sm:border-b-0 lg:border-b-0">
            {stat}
          </div>
        ))}
      </div>
      <div className="grid gap-3 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <MarkdownFile filename={`${builder.id}-profile.md`} markdown={builder.markdown} />
        </div>
        <div className="grid gap-3 lg:col-span-2">
          <DailySpend builder={builder} />
          <AgentMix builder={builder} />
          <ActivityHeatmap heat={builder.heatPdt} sampleStart={sampleStart} sampleEnd={sampleEnd} />
          <ModelMix builder={builder} />
          <PricestSessions builder={builder} />
        </div>
      </div>
    </section>
  );
}
