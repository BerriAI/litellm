"use client";

import {
  CartesianGrid,
  ReferenceLine,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
  type ScatterShapeProps,
  type TooltipContentProps,
  type TooltipValueType,
} from "recharts";
import { ChartContainer, ChartTooltip, type ChartConfig } from "@/components/ui/chart";
import { formatCompactUsd, formatUsd } from "../overview/overviewData";
import { Panel } from "../overview/Primitives";
import type { BuilderInsightBuilder } from "./builderInsightsData";

interface SpendOutputPoint {
  id: string;
  name: string;
  firstName: string;
  spend: number;
  prs: number;
  prsDevin: number;
  spendPerPr: number | null;
  selected: boolean;
  mostlyDevin: boolean;
  labelOffsetX: number;
  labelOffsetY: number;
  onSelect: (id: string) => void;
}

interface LabelBounds {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

const chartConfig = {
  prs: { label: "Merged PRs", color: "var(--chart-1)" },
} satisfies ChartConfig;

const labelBoundsFor = (point: SpendOutputPoint, xMax: number, yMax: number, labelOffsetY: number): LabelBounds => {
  const x = (Math.log(Math.max(point.spend, 500) / 500) / Math.log(xMax / 500)) * 280;
  const y = 360 - (point.prs / yMax) * 360;
  const width = point.firstName.length * 5.5;
  const left = point.labelOffsetX < 0 ? x + point.labelOffsetX - width : x + point.labelOffsetX;
  const top = y + labelOffsetY - 7;
  return { left, right: left + width, top, bottom: top + 12 };
};

const placeLabels = (points: SpendOutputPoint[], xMax: number, yMax: number): SpendOutputPoint[] =>
  points.reduce<{ points: SpendOutputPoint[]; bounds: LabelBounds[] }>(
    (placed, point) => {
      const direction = point.prs <= yMax / 2 ? -1 : 1;
      const labelOffsetY = (step: number): number => {
        const offset = step * direction * 14;
        const bounds = labelBoundsFor(point, xMax, yMax, offset);
        const overlaps = placed.bounds.some((previous) => {
          const overlapsHorizontally = bounds.left < previous.right + 4 && bounds.right + 4 > previous.left;
          const overlapsVertically = bounds.top < previous.bottom + 2 && bounds.bottom + 2 > previous.top;
          return overlapsHorizontally && overlapsVertically;
        });
        return overlaps ? labelOffsetY(step + 1) : offset;
      };
      const offsetY = labelOffsetY(0);
      return {
        points: [...placed.points, { ...point, labelOffsetY: offsetY }],
        bounds: [...placed.bounds, labelBoundsFor(point, xMax, yMax, offsetY)],
      };
    },
    { points: [], bounds: [] },
  ).points;

function SpendOutputDot({ cx, cy, payload }: ScatterShapeProps) {
  const point = payload as SpendOutputPoint | undefined;
  if (!point || cx === undefined || cy === undefined) return null;

  return (
    <g
      role="button"
      tabIndex={0}
      aria-label={`Select ${point.name}`}
      className="cursor-pointer"
      onClick={() => point.onSelect(point.id)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          point.onSelect(point.id);
        }
      }}
    >
      {point.selected && (
        <circle cx={cx} cy={cy} r={7} fill="var(--chart-1)" fillOpacity={0.18} stroke="var(--chart-1)" strokeWidth={1.5} />
      )}
      <circle
        cx={cx}
        cy={cy}
        r={point.selected ? 4.5 : 3.5}
        fill={point.mostlyDevin ? "var(--card)" : "var(--chart-1)"}
        stroke={point.mostlyDevin ? "var(--chart-1)" : undefined}
        strokeWidth={point.mostlyDevin ? 1.75 : undefined}
      />
      <text
        x={cx + point.labelOffsetX}
        y={cy + 3 + point.labelOffsetY}
        textAnchor={point.labelOffsetX < 0 ? "end" : "start"}
        fill="var(--muted-foreground)"
        fontSize={10}
      >
        {point.firstName}
      </text>
    </g>
  );
}

function SpendOutputTooltip({
  active,
  payload,
}: Pick<TooltipContentProps<TooltipValueType, string | number>, "active" | "payload">) {
  const point = payload?.[0]?.payload as SpendOutputPoint | undefined;
  if (!active || !point) return null;
  return (
    <div className="rounded-md border border-border bg-popover p-2.5 text-xs">
      <p className="mb-2 font-medium">{point.name}</p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        <span className="text-muted-foreground">Spend</span>
        <span className="text-right font-medium tabular-nums">{formatUsd(point.spend)}</span>
        <span className="text-muted-foreground">Merged PRs</span>
        <span className="text-right font-medium tabular-nums">{point.prs.toLocaleString()}</span>
        <span className="text-muted-foreground">$/PR</span>
        <span className="text-right font-medium tabular-nums">
          {point.spendPerPr === null ? "—" : formatUsd(point.spendPerPr)}
        </span>
      </div>
    </div>
  );
}

export function SpendVsOutput({
  builders,
  selectedId,
  medianSpendPerPr,
  onSelect,
}: {
  builders: readonly BuilderInsightBuilder[];
  selectedId: string | null;
  medianSpendPerPr: number | null;
  onSelect: (id: string) => void;
}) {
  const rawPoints = builders.map((builder) => ({
      id: builder.id,
      name: builder.name,
      firstName: builder.name.split(/\s+/)[0] ?? builder.name,
      spend: builder.spend,
      prs: builder.prs,
      prsDevin: builder.prsDevin,
      spendPerPr: builder.spendPerPr ?? (builder.prs > 0 ? builder.spend / builder.prs : null),
      selected: builder.id === selectedId,
      mostlyDevin: builder.prs > 0 && builder.prsDevin / builder.prs > 0.5,
      labelOffsetX: builder.prs === 0 || builder.spend <= 25_000 ? 7 : -7,
      labelOffsetY: 0,
      onSelect,
    }));
  const maxSpend = Math.max(30_000, ...rawPoints.map((point) => point.spend));
  const xMax = maxSpend <= 30_000 ? 30_000 : Math.ceil(maxSpend / 10_000) * 10_000;
  const maxPrs = Math.max(0, ...rawPoints.map((point) => point.prs));
  const yMax = Math.max(1, maxPrs);
  const points = placeLabels(rawPoints, xMax, yMax);
  const referenceMedian = medianSpendPerPr && medianSpendPerPr > 0 ? medianSpendPerPr : undefined;
  const referenceEndSpend =
    referenceMedian === undefined ? undefined : Math.min(xMax, yMax * referenceMedian);
  const canRenderReference = referenceEndSpend !== undefined && referenceEndSpend >= 500;
  const referenceSegment =
    referenceMedian !== undefined && canRenderReference
      ? ([
          { x: 500, y: 500 / referenceMedian },
          { x: referenceEndSpend, y: referenceEndSpend / referenceMedian },
        ] as const)
      : undefined;

  return (
    <Panel
      className="h-full"
      title="Spend vs output"
      subtitle={
        <span className="block whitespace-normal">
          Merged PRs vs gateway spend, 30 days. Up and left is more output per dollar
        </span>
      }
    >
      <ChartContainer config={chartConfig} className="aspect-auto min-h-[360px] w-full flex-1">
        <ScatterChart margin={{ left: 0, right: 22, top: 12, bottom: 0 }}>
          <CartesianGrid vertical stroke="var(--border)" strokeOpacity={0.6} />
          <XAxis
            type="number"
            dataKey="spend"
            scale="log"
            domain={[500, xMax]}
            ticks={[500, 1_000, 5_000, 10_000, 30_000]}
            tickFormatter={(value) => formatCompactUsd(Number(value))}
            tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            allowDataOverflow
          />
          <YAxis
            type="number"
            dataKey="prs"
            domain={[0, yMax]}
            tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            allowDecimals={false}
          />
          {referenceSegment && (
            <ReferenceLine
              segment={referenceSegment}
              stroke="var(--muted-foreground)"
              strokeOpacity={0.4}
              strokeDasharray="4 4"
              ifOverflow="hidden"
              label={{
                value: "median $/PR",
                position: "insideTopRight",
                fill: "var(--muted-foreground)",
                fontSize: 10,
              }}
            />
          )}
          <ChartTooltip content={(props) => <SpendOutputTooltip {...props} />} />
          <Scatter
            data={points}
            dataKey="prs"
            fill="var(--chart-1)"
            shape={SpendOutputDot}
            isAnimationActive={false}
          />
        </ScatterChart>
      </ChartContainer>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 pt-2 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1.5">
          <span className="size-2 rounded-full" style={{ backgroundColor: "var(--chart-1)" }} />
          Own PRs
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            className="size-2 rounded-full border"
            style={{ backgroundColor: "var(--card)", borderColor: "var(--chart-1)" }}
          />
          Mostly Devin PRs (Devin compute billed elsewhere)
        </span>
      </div>
    </Panel>
  );
}
