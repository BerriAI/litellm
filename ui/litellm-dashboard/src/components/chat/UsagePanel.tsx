"use client";

import React, { useState } from "react";
import { BarChart3 } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { userDailyActivityAggregatedCall } from "../networking";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

const USAGE_QUERY_KEY = "chat-user-usage";

interface Props {
  accessToken: string;
  userId: string;
}

type TimeRange = "7d" | "30d" | "90d";

function getDateRange(range: TimeRange): { start: Date; end: Date } {
  const end = new Date();
  const start = new Date();
  const days = range === "7d" ? 7 : range === "30d" ? 30 : 90;
  start.setDate(end.getDate() - days);
  return { start, end };
}

interface DailyMetrics {
  spend: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  api_requests: number;
  successful_requests: number;
  failed_requests: number;
}

interface DailySpendEntry {
  date: string;
  metrics: DailyMetrics;
}

interface UsageMetadata {
  total_spend: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_tokens: number;
  total_api_requests: number;
  total_successful_requests: number;
  total_failed_requests: number;
}

interface UsageResponse {
  results: DailySpendEntry[];
  metadata: UsageMetadata;
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString();
}

function SparklineBar({ data, maxVal }: { data: number[]; maxVal: number }) {
  const barWidth = Math.max(2, Math.floor(200 / Math.max(data.length, 1)));
  const height = 48;
  return (
    <div className="flex items-end gap-px" style={{ height }}>
      {data.map((val, i) => {
        const h = maxVal > 0 ? Math.max(2, (val / maxVal) * height) : 2;
        return (
          <div
            key={i}
            className="bg-primary rounded-[1px]"
            style={{
              width: barWidth,
              height: h,
              opacity: 0.7 + 0.3 * (val / Math.max(maxVal, 1)),
            }}
          />
        );
      })}
    </div>
  );
}

const TIME_RANGE_OPTIONS: { value: TimeRange; label: string }[] = [
  { value: "7d", label: "7d" },
  { value: "30d", label: "30d" },
  { value: "90d", label: "90d" },
];

const UsagePanel: React.FC<Props> = ({ accessToken, userId }) => {
  const [timeRange, setTimeRange] = useState<TimeRange>("30d");
  const { start, end } = getDateRange(timeRange);

  const { data, isLoading } = useQuery({
    queryKey: [USAGE_QUERY_KEY, accessToken, userId, timeRange],
    queryFn: () => userDailyActivityAggregatedCall(accessToken, start, end, userId),
    enabled: !!accessToken,
  });

  const usage = data as UsageResponse | undefined;
  const meta = usage?.metadata;
  const dailyData = usage?.results ?? [];
  const dailySpend = dailyData.map((d) => d.metrics.spend);
  const dailyRequests = dailyData.map((d) => d.metrics.api_requests);
  const maxSpend = Math.max(...dailySpend, 0);
  const maxRequests = Math.max(...dailyRequests, 0);

  const statCards: Array<{ label: string; value: string; sub?: string; subVariant?: "error" }> = meta
    ? [
        { label: "Total Spend", value: `$${meta.total_spend.toFixed(2)}` },
        { label: "API Requests", value: formatNumber(meta.total_api_requests) },
        {
          label: "Tokens Used",
          value: formatNumber(meta.total_tokens),
          sub: `${formatNumber(meta.total_prompt_tokens)} in / ${formatNumber(meta.total_completion_tokens)} out`,
        },
        {
          label: "Success Rate",
          value:
            meta.total_api_requests > 0
              ? `${((meta.total_successful_requests / meta.total_api_requests) * 100).toFixed(1)}%`
              : "N/A",
          sub: meta.total_failed_requests > 0 ? `${meta.total_failed_requests} failed` : undefined,
          subVariant: meta.total_failed_requests > 0 ? "error" : undefined,
        },
      ]
    : [];

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-base font-semibold text-foreground mb-0.5">Your Usage</h2>
          <p className="text-sm text-muted-foreground m-0">Spend and request activity</p>
        </div>
        <div className="flex gap-1">
          {TIME_RANGE_OPTIONS.map((opt) => (
            <Button
              key={opt.value}
              variant={timeRange === opt.value ? "default" : "outline"}
              size="sm"
              onClick={() => setTimeRange(opt.value)}
            >
              {opt.label}
            </Button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-2 gap-3">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="border rounded-lg p-4 bg-card flex flex-col gap-2">
              <Skeleton className="h-3 w-1/2" />
              <Skeleton className="h-5 w-2/3" />
            </div>
          ))}
        </div>
      ) : !meta || meta.total_api_requests === 0 ? (
        <div className="text-center text-muted-foreground text-sm py-12 border border-dashed rounded-lg">
          <BarChart3 className="h-6 w-6 mb-3 mx-auto text-muted-foreground/50" />
          No usage data for this period
        </div>
      ) : (
        <div>
          <div className="grid grid-cols-2 gap-3 mb-5">
            {statCards.map((card) => (
              <div key={card.label} className="border rounded-lg p-4 bg-card">
                <div className="text-xs text-muted-foreground mb-1">{card.label}</div>
                <div className="text-xl font-semibold text-foreground">{card.value}</div>
                {card.sub && (
                  <div
                    className={`text-xs mt-0.5 ${
                      card.subVariant === "error" ? "text-red-600 dark:text-red-400" : "text-muted-foreground"
                    }`}
                  >
                    {card.sub}
                  </div>
                )}
              </div>
            ))}
          </div>

          {dailyData.length > 1 && (
            <div className="grid grid-cols-2 gap-3">
              <div className="border rounded-lg p-4 bg-card">
                <div className="text-xs text-muted-foreground mb-2">Daily Spend</div>
                <SparklineBar data={dailySpend} maxVal={maxSpend} />
              </div>
              <div className="border rounded-lg p-4 bg-card">
                <div className="text-xs text-muted-foreground mb-2">Daily Requests</div>
                <SparklineBar data={dailyRequests} maxVal={maxRequests} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default UsagePanel;
