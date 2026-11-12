"use client";

import { TriangleAlert } from "lucide-react";
import type { ParseKeys } from "i18next";
import React from "react";
import { useTranslation } from "react-i18next";

import { cn } from "@/lib/cva.config";
import { TASK_INSIGHTS_METRICS, type TaskInsightsMetric } from "../../usageUrlState";
import {
  cellReading,
  compatibilityFor,
  entryLookup,
  formatReading,
  formatShare,
  heatIntensity,
  modelLabel,
  shareOfTask,
  type TaskMatrix,
} from "./taskInsightsData";

const METRIC_LABELS: Readonly<Record<TaskInsightsMetric, ParseKeys>> = {
  share: "usage.taskInsights.heatmap.metrics.share",
  cost: "usage.taskInsights.heatmap.metrics.cost",
  latency: "usage.taskInsights.heatmap.metrics.latency",
};

interface TaskHeatmapProps {
  matrix: TaskMatrix;
  metric: TaskInsightsMetric;
  onMetricChange: (metric: TaskInsightsMetric) => void;
  selectedTaskId: string | null;
  selectedModelId: string | null;
  onSelectCell: (taskId: string, modelConfigId: string) => void;
}

export default function TaskHeatmap({
  matrix,
  metric,
  onMetricChange,
  selectedTaskId,
  selectedModelId,
  onSelectCell,
}: TaskHeatmapProps) {
  const { t } = useTranslation();
  const lookup = React.useMemo(() => entryLookup(matrix.entries), [matrix.entries]);
  const values = React.useMemo(
    () => matrix.entries.map((entry) => cellReading(entry, metric).value),
    [matrix.entries, metric],
  );
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div
          role="radiogroup"
          aria-label={t("usage.taskInsights.heatmap.metricLabel")}
          className="flex w-fit rounded-lg bg-muted p-1"
        >
          {TASK_INSIGHTS_METRICS.map((option) => (
            <button
              key={option}
              type="button"
              role="radio"
              aria-checked={metric === option}
              className={cn(
                "rounded-md px-3 py-1 text-xs transition-colors",
                metric === option ? "bg-card text-foreground shadow-xs" : "text-muted-foreground hover:text-foreground",
              )}
              onClick={() => onMetricChange(option)}
            >
              {t(METRIC_LABELS[option])}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">{t("usage.taskInsights.heatmap.legend")}</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full border-separate border-spacing-1 text-xs">
          <caption className="sr-only">{t("usage.taskInsights.heatmap.caption", { metric: t(METRIC_LABELS[metric]) })}</caption>
          <thead>
            <tr>
              <th scope="col" className="w-48 text-left font-medium text-muted-foreground">
                {t("usage.taskInsights.heatmap.taskColumn")}
              </th>
              {matrix.model_configs.map((config) => (
                <th key={config.model_config_id} scope="col" className="min-w-28 text-left font-medium">
                  <span className="block truncate text-foreground">{modelLabel(config)}</span>
                  <span className="block truncate font-normal text-muted-foreground">
                    {config.configured ? config.provider_model : t("usage.taskInsights.heatmap.notConfigured")}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.tasks.map((task) => (
              <tr key={task.task_id}>
                <th scope="row" className="truncate text-left font-medium text-foreground">
                  {task.name}
                </th>
                {matrix.model_configs.map((config) => {
                  const entry = lookup(config.model_config_id, task.task_id);
                  const reading = cellReading(entry, metric);
                  const intensity = heatIntensity(reading.value, values);
                  const shown = formatReading(reading, metric);
                  const share = metric === "share" ? shareOfTask(matrix.entries, config.model_config_id, task.task_id) : null;
                  const compatibility = compatibilityFor(matrix.compatibility, config.model_config_id, task.task_id);
                  const selected = selectedTaskId === task.task_id && selectedModelId === config.model_config_id;
                  const label = t("usage.taskInsights.heatmap.cellLabel", {
                    task: task.name,
                    model: modelLabel(config),
                    value: shown ?? t("usage.taskInsights.heatmap.noEvidence"),
                  });
                  return (
                    <td key={config.model_config_id} className="p-0">
                      <button
                        type="button"
                        aria-label={label}
                        aria-pressed={selected}
                        onClick={() => onSelectCell(task.task_id, config.model_config_id)}
                        className={cn(
                          "flex h-12 w-full flex-col items-start justify-center rounded-md px-2 text-left transition-shadow",
                          intensity === null && "bg-muted bg-[repeating-linear-gradient(45deg,transparent,transparent_4px,rgba(0,0,0,0.04)_4px,rgba(0,0,0,0.04)_8px)] text-muted-foreground",
                          selected && "ring-2 ring-ring",
                        )}
                        style={
                          intensity === null
                            ? undefined
                            : { backgroundColor: `rgba(43, 63, 214, ${intensity})`, color: intensity > 0.55 ? "white" : undefined }
                        }
                      >
                        <span className="flex items-center gap-1 font-medium">
                          {shown ?? t("usage.taskInsights.heatmap.noEvidence")}
                          {compatibility?.status === "incompatible" && (
                            <TriangleAlert
                              aria-label={t("usage.taskInsights.heatmap.incompatible", {
                                missing: compatibility.missing.join(", "),
                              })}
                              className="size-3"
                            />
                          )}
                        </span>
                        {reading.value !== null && (
                          <span className="opacity-80">
                            {share !== null
                              ? t("usage.taskInsights.heatmap.ofTask", { share: formatShare(share) })
                              : t("usage.taskInsights.heatmap.observations", { observations: reading.observations })}
                          </span>
                        )}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {matrix.unattributed_request_count > 0 && (
        <p className="text-xs text-muted-foreground">
          {t("usage.taskInsights.heatmap.unattributed", { requests: matrix.unattributed_request_count })}
        </p>
      )}
    </div>
  );
}
