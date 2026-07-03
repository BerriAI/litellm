import { Progress, Typography } from "antd";
import React from "react";

import { formatNumberWithCommas } from "@/utils/dataUtils";
import { formatModelMaxBudgetTimePeriod } from "./ModelMaxBudgetEditor";

export type ModelMaxBudgetUsageEntry = {
  current_spend: number;
  budget_limit: number;
  time_period: string;
  scope?: "key" | "team_member" | "team";
  percent_used?: number;
};

export type ModelMaxBudgetUsageMap = Record<string, ModelMaxBudgetUsageEntry>;

function scopeLabel(scope: ModelMaxBudgetUsageEntry["scope"]): string {
  if (scope === "team" || scope === "team_member") {
    return "Shared across your keys on this team";
  }
  return "This key only";
}

function progressStatus(percent: number): "success" | "normal" | "exception" {
  if (percent >= 100) {
    return "exception";
  }
  if (percent >= 80) {
    return "normal";
  }
  return "success";
}

function progressStrokeColor(percent: number): string {
  if (percent >= 100) {
    return "#ff4d4f";
  }
  if (percent >= 80) {
    return "#faad14";
  }
  return "#52c41a";
}

export function ModelMaxBudgetUsageOverview({
  usage,
}: {
  usage: ModelMaxBudgetUsageMap | null | undefined;
}) {
  const entries = Object.entries(usage ?? {});
  if (entries.length === 0) {
    return <Typography.Text type="secondary">No per-model budget usage to show</Typography.Text>;
  }

  return (
    <div className="space-y-4">
      {entries.map(([model, entry]) => {
        const limit = entry.budget_limit ?? 0;
        const spend = entry.current_spend ?? 0;
        const percent =
          entry.percent_used ??
          (limit > 0 ? Math.min(Math.round((spend / limit) * 1000) / 10, 999.9) : 0);
        const period = formatModelMaxBudgetTimePeriod(entry.time_period);

        return (
          <div key={model}>
            <div className="flex items-baseline justify-between gap-2">
              <Typography.Text className="font-medium">{model}</Typography.Text>
              <Typography.Text className="text-xs text-gray-500">
                ${formatNumberWithCommas(spend, 2)} / ${formatNumberWithCommas(limit, 2)} / {period}
              </Typography.Text>
            </div>
            <Progress
              percent={Math.min(percent, 100)}
              success={{ percent: Math.min(percent, 100) }}
              status={progressStatus(percent)}
              strokeColor={progressStrokeColor(percent)}
              format={() => `${percent}%`}
            />
            <Typography.Text type="secondary" className="text-xs">
              {scopeLabel(entry.scope)}
            </Typography.Text>
          </div>
        );
      })}
    </div>
  );
}
