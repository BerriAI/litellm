"use client";

import React from "react";
import { useTranslation } from "react-i18next";
import { Cell, LabelList, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";

import { ChartContainer } from "@/components/ui/chart";
import { UNASSIGNED_COLOR, formatCount, mapBubbles, type TaskMapPoint, type TaskSummary } from "./taskInsightsData";

const EMPTY_CONFIG = {};
const DOMAIN: [number, number] = [0, 100];

interface BubbleTooltipProps {
  active?: boolean;
  payload?: ReadonlyArray<{ payload?: unknown }>;
}

const bubbleOf = (value: unknown): { name: string; countLabel: string } | null => {
  if (typeof value !== "object" || value === null) return null;
  if (!("name" in value) || !("countLabel" in value)) return null;
  return { name: String(value.name), countLabel: String(value.countLabel) };
};

function BubbleTooltip({ active, payload }: BubbleTooltipProps) {
  const { t } = useTranslation();
  const bubble = active ? bubbleOf(payload?.[0]?.payload) : null;
  if (!bubble) return null;
  return (
    <div className="rounded-md border bg-popover px-2 py-1 text-xs text-popover-foreground shadow-sm">
      <p className="font-medium">{bubble.name}</p>
      <p className="text-muted-foreground">
        {t("usage.taskInsights.detail.requests", { requests: bubble.countLabel })}
      </p>
    </div>
  );
}

interface TaskClusterMapProps {
  tasks: readonly TaskSummary[];
  points: readonly TaskMapPoint[];
  colors: ReadonlyMap<string, string>;
  selectedTaskId: string | null;
  onSelect: (taskId: string) => void;
}

export default function TaskClusterMap({ tasks, points, colors, selectedTaskId, onSelect }: TaskClusterMapProps) {
  const { t } = useTranslation();
  const bubbles = React.useMemo(
    () => mapBubbles(tasks).map((bubble) => ({ ...bubble, countLabel: formatCount(bubble.z) })),
    [tasks],
  );
  const plotted = React.useMemo(() => points.filter((point) => colors.has(point.task_id)), [points, colors]);
  return (
    <figure className="flex flex-col gap-2">
      <ChartContainer
        config={EMPTY_CONFIG}
        role="img"
        aria-label={t("usage.taskInsights.map.ariaLabel", { tasks: bubbles.length })}
        className="aspect-auto h-80 w-full"
      >
        <ScatterChart margin={{ top: 24, right: 24, bottom: 12, left: 24 }} accessibilityLayer={false}>
          <Tooltip cursor={false} content={<BubbleTooltip />} />
          <XAxis type="number" dataKey="x" domain={DOMAIN} hide />
          <YAxis type="number" dataKey="y" domain={DOMAIN} hide reversed />
          <ZAxis zAxisId="points" type="number" dataKey="z" range={[14, 14]} />
          <ZAxis zAxisId="bubbles" type="number" dataKey="z" range={[220, 2600]} />
          <Scatter
            data={plotted.map((point) => ({ ...point, z: 1 }))}
            zAxisId="points"
            isAnimationActive={false}
            legendType="none"
          >
            {plotted.map((point) => {
              const color = colors.get(point.task_id) ?? UNASSIGNED_COLOR;
              return (
                <Cell
                  key={point.request_id}
                  fill={color}
                  fillOpacity={point.placement === "embedding" ? 0.55 : 0}
                  stroke={color}
                  strokeOpacity={0.6}
                />
              );
            })}
          </Scatter>
          <Scatter data={bubbles} zAxisId="bubbles" isAnimationActive={false} legendType="none">
            {bubbles.map((bubble) => (
              <Cell
                key={bubble.task_id}
                fill={colors.get(bubble.task_id) ?? UNASSIGNED_COLOR}
                fillOpacity={selectedTaskId === null || selectedTaskId === bubble.task_id ? 0.32 : 0.12}
                stroke={colors.get(bubble.task_id) ?? UNASSIGNED_COLOR}
                strokeWidth={selectedTaskId === bubble.task_id ? 3 : 1.5}
                className="cursor-pointer"
                onClick={() => onSelect(bubble.task_id)}
              />
            ))}
            <LabelList
              dataKey="countLabel"
              position="center"
              className="pointer-events-none fill-foreground text-xs font-medium"
            />
          </Scatter>
        </ScatterChart>
      </ChartContainer>
      <figcaption className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span>{t("usage.taskInsights.map.caption")}</span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden="true" className="size-2 rounded-full bg-muted-foreground/60" />
          {t("usage.taskInsights.map.embeddingPoints")}
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden="true" className="size-2 rounded-full border border-muted-foreground" />
          {t("usage.taskInsights.map.categoricalPoints")}
        </span>
        <span>{t("usage.taskInsights.map.sampled", { points: formatCount(plotted.length) })}</span>
      </figcaption>
    </figure>
  );
}
