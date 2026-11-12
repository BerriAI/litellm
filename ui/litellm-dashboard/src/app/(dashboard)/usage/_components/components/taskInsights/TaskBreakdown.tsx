"use client";

import React from "react";
import { useTranslation } from "react-i18next";

import {
  NUMERIC_CELL_CLASS,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/cva.config";
import {
  UNASSIGNED_COLOR,
  formatCount,
  formatShare,
  formatUsd,
  formatUsdTotal,
  type TaskSummary,
} from "./taskInsightsData";

const RANKS = ["requests", "spend"] as const;
type Rank = (typeof RANKS)[number];

interface TaskBreakdownProps {
  tasks: readonly TaskSummary[];
  colors: ReadonlyMap<string, string>;
  selectedTaskId: string | null;
  onSelect: (taskId: string) => void;
}

const ranked = (tasks: readonly TaskSummary[], rank: Rank): TaskSummary[] =>
  [...tasks].sort((a, b) =>
    rank === "requests" ? b.request_count - a.request_count : (b.spend ?? -1) - (a.spend ?? -1),
  );

export default function TaskBreakdown({ tasks, colors, selectedTaskId, onSelect }: TaskBreakdownProps) {
  const { t } = useTranslation();
  const [rank, setRank] = React.useState<Rank>("requests");
  const rows = React.useMemo(() => ranked(tasks, rank), [tasks, rank]);
  const maxShare = Math.max(0, ...tasks.map((task) => task.share));
  return (
    <div className="flex flex-col gap-2">
      <div
        role="radiogroup"
        aria-label={t("usage.taskInsights.breakdown.rankBy")}
        className="flex w-fit rounded-lg bg-muted p-1"
      >
        {RANKS.map((option) => (
          <button
            key={option}
            type="button"
            role="radio"
            aria-checked={rank === option}
            className={cn(
              "rounded-md px-3 py-1 text-xs transition-colors",
              rank === option ? "bg-card text-foreground shadow-xs" : "text-muted-foreground hover:text-foreground",
            )}
            onClick={() => setRank(option)}
          >
            {t(
              option === "requests"
                ? "usage.taskInsights.breakdown.byRequests"
                : "usage.taskInsights.breakdown.bySpend",
            )}
          </button>
        ))}
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>{t("usage.taskInsights.breakdown.task")}</TableHead>
            <TableHead className={NUMERIC_CELL_CLASS}>{t("usage.taskInsights.breakdown.requests")}</TableHead>
            <TableHead className={NUMERIC_CELL_CLASS}>{t("usage.taskInsights.breakdown.sessions")}</TableHead>
            <TableHead className={NUMERIC_CELL_CLASS}>{t("usage.taskInsights.breakdown.spend")}</TableHead>
            <TableHead className={NUMERIC_CELL_CLASS}>{t("usage.taskInsights.breakdown.avgCost")}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((task) => (
            <TableRow key={task.task_id} data-state={selectedTaskId === task.task_id ? "selected" : undefined}>
              <TableCell>
                <button
                  type="button"
                  className="flex items-center gap-2 text-left font-medium hover:underline"
                  onClick={() => onSelect(task.task_id)}
                >
                  <span
                    aria-hidden="true"
                    className="size-2.5 shrink-0 rounded-full"
                    style={{ backgroundColor: colors.get(task.task_id) ?? UNASSIGNED_COLOR }}
                  />
                  {task.name}
                </button>
              </TableCell>
              <TableCell className={NUMERIC_CELL_CLASS}>
                <div className="flex items-center justify-end gap-2">
                  <span
                    aria-hidden="true"
                    className="h-1.5 rounded-full bg-primary/60"
                    style={{ width: `${maxShare > 0 ? (task.share / maxShare) * 48 : 0}px` }}
                  />
                  <span>{formatCount(task.request_count)}</span>
                  <span className="w-12 text-muted-foreground">{formatShare(task.share)}</span>
                </div>
              </TableCell>
              <TableCell className={NUMERIC_CELL_CLASS}>{formatCount(task.session_count)}</TableCell>
              <TableCell className={NUMERIC_CELL_CLASS}>
                <span>{formatUsdTotal(task.spend) ?? t("usage.taskInsights.noData")}</span>
                {task.cost_observations < task.request_count && (
                  <span className="block text-xs text-muted-foreground">
                    {t("usage.taskInsights.breakdown.costKnownFor", {
                      known: formatCount(task.cost_observations),
                      total: formatCount(task.request_count),
                    })}
                  </span>
                )}
              </TableCell>
              <TableCell className={NUMERIC_CELL_CLASS}>
                {formatUsd(task.avg_cost_usd) ?? t("usage.taskInsights.noData")}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
