"use client";

import { Cell, Pie, PieChart } from "recharts";
import { ChartContainer } from "@/components/ui/chart";
import { AgentMark } from "../overview/TopAgents";
import { agentRowFor } from "../overview/agentCatalog";
import { formatCompactUsd } from "../overview/overviewData";
import { Panel } from "../overview/Primitives";
import { agentSpendShares, type BuilderAgentEntry } from "./builderInsightsData";

const COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"];

export function BuilderAgentSpendPanel({ agents }: { agents: readonly BuilderAgentEntry[] }) {
  const shares = agentSpendShares(agents).map((agent, index) => ({
    ...agent,
    label: agentRowFor(agent.id).label,
    color: COLORS[index % COLORS.length],
  }));
  const chartConfig = Object.fromEntries(shares.map((agent) => [agent.id, { label: agent.label, color: agent.color }]));

  return (
    <Panel title="Agent split" subtitle="Share of this builder's spend by agent · 7d sample">
      {shares.length === 0 ? (
        <p className="text-xs text-muted-foreground">No agent spend in this sample</p>
      ) : (
        <div className="grid items-center gap-4 sm:grid-cols-[10rem_minmax(0,1fr)]">
          <div className="relative mx-auto size-40">
            <ChartContainer config={chartConfig} className="aspect-auto size-40">
              <PieChart>
                <Pie
                  data={shares}
                  dataKey="spend"
                  nameKey="id"
                  innerRadius="60%"
                  outerRadius="86%"
                  paddingAngle={2}
                  stroke="var(--card)"
                  isAnimationActive={false}
                >
                  {shares.map((agent) => (
                    <Cell key={agent.id} fill={agent.color} />
                  ))}
                </Pie>
              </PieChart>
            </ChartContainer>
            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center text-center">
              <span className="text-lg font-semibold tabular-nums">
                {shares[0].share.toLocaleString(undefined, { style: "percent", maximumFractionDigits: 1 })}
              </span>
              <span className="max-w-24 truncate text-xs text-muted-foreground">{shares[0].label}</span>
            </div>
          </div>
          <div className="grid min-w-0 gap-3">
            {shares.map((agent) => (
              <div
                key={agent.id}
                className="grid min-w-0 grid-cols-[minmax(0,1fr)_minmax(3rem,0.7fr)_3.5rem_4.25rem] items-center gap-2 text-xs"
              >
                <div className="flex min-w-0 items-center gap-2">
                  <AgentMark agent={agentRowFor(agent.id)} size="sm" />
                  <span className="truncate">{agent.label}</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                  <span
                    className="block h-full rounded-full"
                    style={{ width: `${agent.share * 100}%`, backgroundColor: agent.color }}
                  />
                </div>
                <span className="text-right tabular-nums text-muted-foreground">
                  {agent.share.toLocaleString(undefined, { style: "percent", maximumFractionDigits: 1 })}
                </span>
                <span className="text-right tabular-nums text-muted-foreground">{formatCompactUsd(agent.spend)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </Panel>
  );
}
