"use client";

import { Cell, Pie, PieChart, Tooltip } from "recharts";
import { ChartContainer, ChartTooltipContent } from "@/components/ui/chart";
import { AgentMark } from "../overview/TopAgents";
import { agentRowFor } from "../overview/agentCatalog";
import { Panel } from "../overview/Primitives";
import { agentSpendShares, type BuilderAgentEntry } from "./builderInsightsData";

const COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"];

export function BuilderAgentSpendPanel({ agents, title }: { agents: readonly BuilderAgentEntry[]; title: string }) {
  const shares = agentSpendShares(agents);
  const chartConfig = Object.fromEntries(
    shares.map((agent, index) => [
      agent.id,
      { label: agentRowFor(agent.id).label, color: COLORS[index % COLORS.length] },
    ]),
  );

  return (
    <Panel title={title} subtitle="Share of spend by agent · 7d sample">
      {shares.length === 0 ? (
        <p className="text-xs text-muted-foreground">No agent spend in this sample</p>
      ) : (
        <>
          <ChartContainer config={chartConfig} className="mx-auto aspect-auto h-36 w-full max-w-56">
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
                {shares.map((agent, index) => (
                  <Cell key={agent.id} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip content={<ChartTooltipContent hideLabel />} />
            </PieChart>
          </ChartContainer>
          <div className="mt-3 grid gap-2">
            {shares.map((agent, index) => (
              <div key={agent.id} className="flex items-center gap-2 text-xs">
                <AgentMark agent={agentRowFor(agent.id)} size="sm" />
                <span className="min-w-0 flex-1 truncate">{agentRowFor(agent.id).label}</span>
                <span className="tabular-nums text-muted-foreground">
                  {agent.share.toLocaleString(undefined, { style: "percent", maximumFractionDigits: 1 })}
                </span>
                <span
                  aria-hidden="true"
                  className="size-2 rounded-full"
                  style={{ backgroundColor: COLORS[index % COLORS.length] }}
                />
              </div>
            ))}
          </div>
        </>
      )}
    </Panel>
  );
}
