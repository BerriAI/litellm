"use client";

import { useQuery } from "@tanstack/react-query";
import { Alert, Card, Segmented, Table, Tag, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import { ComponentProps, useMemo, useState } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { MetricCard } from "@/components/GuardrailsMonitor/MetricCard";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import ChartLoader from "@/components/shared/chart_loader";
import { AreaChart, CustomLegend, DonutChart, type ChartColor } from "@/components/shared/charts";
import {
  CostOptimizationType,
  CostSavingsMetrics,
  costSavingsActivityCall,
  costSavingsRecentRequestsCall,
  OptimizedRequestSummary,
  RecentOptimizedRequestsResponse,
} from "@/components/networking";

type DateRangeValue = ComponentProps<typeof AdvancedDatePicker>["value"];

const SERIES = [
  { key: "caching", name: "Caching", color: "emerald" },
  { key: "compression", name: "Compression", color: "blue" },
] as const;

const SERIES_CATEGORIES = SERIES.map((series) => series.name);
const SERIES_COLORS: readonly ChartColor[] = SERIES.map((series) => series.color);

const OPTIMIZATION_TAG_COLOR: Record<CostOptimizationType, string> = {
  caching: "green",
  compression: "blue",
};

type OptimizationFilter = "all" | CostOptimizationType;

const OPTIMIZATION_FILTER_OPTIONS: readonly { label: string; value: OptimizationFilter }[] = [
  { label: "All", value: "all" },
  { label: "Caching", value: "caching" },
  { label: "Compression", value: "compression" },
];

export function formatUsd(value: number): string {
  if (value === 0) return "$0";
  const sign = value < 0 ? "-" : "";
  const abs = Math.abs(value);
  if (abs >= 0.01) {
    return `${sign}$${abs.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  return `${sign}$${abs.toFixed(6).replace(/0+$/, "").replace(/\.$/, "")}`;
}

function defaultDateRange(): DateRangeValue {
  return {
    from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    to: new Date(),
  };
}

const RECENT_REQUEST_COLUMNS: ColumnsType<OptimizedRequestSummary> = [
  {
    title: "Request ID",
    dataIndex: "request_id",
    key: "request_id",
    render: (value: string) => <span className="font-mono text-xs">{value}</span>,
  },
  { title: "Model", dataIndex: "model", key: "model" },
  {
    title: "Tokens",
    dataIndex: "total_tokens",
    key: "total_tokens",
    render: (value: number) => value.toLocaleString(),
  },
  {
    title: "Type",
    dataIndex: "optimizations",
    key: "optimizations",
    render: (optimizations: CostOptimizationType[]) => (
      <span>
        {optimizations.map((optimization) => (
          <Tag key={optimization} color={OPTIMIZATION_TAG_COLOR[optimization]}>
            {optimization}
          </Tag>
        ))}
      </span>
    ),
  },
  {
    title: "Original Cost",
    dataIndex: "original_cost",
    key: "original_cost",
    align: "right",
    render: (value: number) => <span className="text-gray-500 line-through">{formatUsd(value)}</span>,
  },
  {
    title: "Optimized Cost",
    dataIndex: "optimized_cost",
    key: "optimized_cost",
    align: "right",
    render: (value: number) => formatUsd(value),
  },
  {
    title: "Savings",
    dataIndex: "savings",
    key: "savings",
    align: "right",
    render: (value: number) => <span className="text-green-600 font-medium">{formatUsd(value)}</span>,
  },
];

interface SavingsKpiGridProps {
  totals: CostSavingsMetrics | undefined;
  filter: OptimizationFilter;
}

const FILTERED_KPI_CONTENT = {
  caching: {
    savingsLabel: "Caching Savings",
    savingsColor: "text-emerald-600",
    tokensLabel: "Cached Tokens Read",
    savingsOf: (totals: CostSavingsMetrics | undefined) => totals?.cache_savings ?? 0,
    tokensOf: (totals: CostSavingsMetrics | undefined) => totals?.cache_read_input_tokens ?? 0,
  },
  compression: {
    savingsLabel: "Compression Savings",
    savingsColor: "text-blue-600",
    tokensLabel: "Tokens Compressed Away",
    savingsOf: (totals: CostSavingsMetrics | undefined) => totals?.compression_savings ?? 0,
    tokensOf: (totals: CostSavingsMetrics | undefined) => totals?.compression_saved_tokens ?? 0,
  },
} as const;

function SavingsKpiGrid({ totals, filter }: SavingsKpiGridProps) {
  if (filter !== "all") {
    const content = FILTERED_KPI_CONTENT[filter];
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        <MetricCard
          label={content.savingsLabel}
          value={formatUsd(content.savingsOf(totals))}
          valueColor={content.savingsColor}
        />
        <MetricCard label={content.tokensLabel} value={content.tokensOf(totals).toLocaleString()} />
        <MetricCard label="Total Spend" value={formatUsd(totals?.spend ?? 0)} />
      </div>
    );
  }
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <MetricCard label="Total Savings" value={formatUsd(totals?.total_savings ?? 0)} valueColor="text-green-600" />
      <MetricCard
        label="Caching Savings"
        value={formatUsd(totals?.cache_savings ?? 0)}
        valueColor="text-emerald-600"
        subtitle={`${(totals?.cache_read_input_tokens ?? 0).toLocaleString()} cached tokens read`}
      />
      <MetricCard
        label="Compression Savings"
        value={formatUsd(totals?.compression_savings ?? 0)}
        valueColor="text-blue-600"
        subtitle={`${(totals?.compression_saved_tokens ?? 0).toLocaleString()} tokens compressed away`}
      />
      <MetricCard label="Total Spend" value={formatUsd(totals?.spend ?? 0)} />
    </div>
  );
}

interface RecentOptimizedRequestsCardProps {
  data: RecentOptimizedRequestsResponse | undefined;
  isLoading: boolean;
  optimizationFilter: OptimizationFilter;
}

function RecentOptimizedRequestsCard({ data, isLoading, optimizationFilter }: RecentOptimizedRequestsCardProps) {
  const requests = data?.requests;
  const filteredRequests = useMemo(() => {
    if (!requests) return [];
    if (optimizationFilter === "all") return requests;
    return requests.filter((request) => request.optimizations.includes(optimizationFilter));
  }, [requests, optimizationFilter]);

  return (
    <Card className="border border-gray-200 rounded-lg" styles={{ body: { padding: 0 } }}>
      <div className="p-6 pb-4">
        <Typography.Title level={5} className="mb-0!">
          Recent Optimized Requests
        </Typography.Title>
        <Typography.Text type="secondary">
          Latest requests that benefited from caching or compression
          {data ? ` (scanned last ${data.scanned_requests} requests in range)` : ""}
        </Typography.Text>
      </div>
      {isLoading ? (
        <div className="p-6">
          <ChartLoader />
        </div>
      ) : (
        <Table
          columns={RECENT_REQUEST_COLUMNS}
          dataSource={filteredRequests}
          rowKey="request_id"
          pagination={false}
          size="middle"
          locale={{
            emptyText:
              optimizationFilter === "all"
                ? "No optimized requests in this window. Savings appear here once prompt caching or prompt compression kicks in."
                : `No ${optimizationFilter} requests in this window.`,
          }}
        />
      )}
    </Card>
  );
}

export default function CostSavingsView() {
  const { accessToken } = useAuthorized();
  const [dateValue, setDateValue] = useState<DateRangeValue>(defaultDateRange);
  const [optimizationFilter, setOptimizationFilter] = useState<OptimizationFilter>("all");

  const startTime = dateValue.from;
  const endTime = dateValue.to;
  const rangeReady = Boolean(accessToken && startTime && endTime);

  const activityQuery = useQuery({
    queryKey: ["costSavingsActivity", accessToken, startTime?.toDateString(), endTime?.toDateString()],
    queryFn: () => costSavingsActivityCall(accessToken!, startTime!, endTime!),
    enabled: rangeReady,
  });

  const recentQuery = useQuery({
    queryKey: ["costSavingsRecentRequests", accessToken, startTime?.toDateString(), endTime?.toDateString()],
    queryFn: () => costSavingsRecentRequestsCall(accessToken!, startTime!, endTime!),
    enabled: rangeReady,
  });

  const totals = activityQuery.data?.totals;
  const unpricedModels = activityQuery.data?.unpriced_models ?? [];
  const visibleSeries = SERIES.filter((series) => optimizationFilter === "all" || series.key === optimizationFilter);
  const visibleCategories = visibleSeries.map((series) => series.name);
  const visibleColors: readonly ChartColor[] = visibleSeries.map((series) => series.color);
  const chartData =
    activityQuery.data?.results.map((day) => ({
      date: day.date,
      Caching: day.metrics.cache_savings,
      Compression: day.metrics.compression_savings,
    })) ?? [];
  const donutData = totals
    ? [
        { name: "Caching", value: totals.cache_savings },
        { name: "Compression", value: totals.compression_savings },
      ]
    : [];

  return (
    <div className="w-full p-8">
      <div className="flex items-end justify-between gap-6 mb-6">
        <div>
          <Typography.Title level={3} className="mb-0!">
            Cost Savings
          </Typography.Title>
          <Typography.Text type="secondary">Savings from prompt caching and prompt compression</Typography.Text>
        </div>
        <div className="flex items-end gap-4">
          <Segmented<OptimizationFilter>
            options={[...OPTIMIZATION_FILTER_OPTIONS]}
            value={optimizationFilter}
            onChange={setOptimizationFilter}
          />
          <AdvancedDatePicker value={dateValue} onValueChange={setDateValue} label="" showTimeRange={false} />
        </div>
      </div>

      {unpricedModels.length > 0 && (
        <Alert
          className="mb-6"
          type="warning"
          showIcon
          message="Some models are missing prices"
          description={`Savings could not be computed for: ${unpricedModels.join(", ")}. Their savings are shown as $0.`}
        />
      )}

      <SavingsKpiGrid totals={totals} filter={optimizationFilter} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        <Card
          className={
            optimizationFilter === "all"
              ? "lg:col-span-2 border border-gray-200 rounded-lg"
              : "lg:col-span-3 border border-gray-200 rounded-lg"
          }
        >
          <Typography.Title level={5} className="mb-0!">
            Savings Over Time
          </Typography.Title>
          <Typography.Text type="secondary">Daily savings by optimization type</Typography.Text>
          {activityQuery.isLoading ? (
            <ChartLoader />
          ) : (
            <AreaChart
              className="mt-4"
              data={chartData}
              index="date"
              categories={visibleCategories}
              colors={visibleColors}
              valueFormatter={formatUsd}
              yAxisWidth={80}
            />
          )}
        </Card>
        {optimizationFilter === "all" && (
          <Card className="border border-gray-200 rounded-lg">
            <Typography.Title level={5} className="mb-0!">
              Savings Distribution
            </Typography.Title>
            <Typography.Text type="secondary">By optimization type</Typography.Text>
            {activityQuery.isLoading ? (
              <ChartLoader />
            ) : (
              <>
                <DonutChart
                  className="mt-4 h-60"
                  data={donutData}
                  index="name"
                  category="value"
                  colors={SERIES_COLORS}
                  valueFormatter={formatUsd}
                  showLabel
                />
                <CustomLegend categories={SERIES_CATEGORIES} colors={SERIES_COLORS} />
              </>
            )}
          </Card>
        )}
      </div>

      <RecentOptimizedRequestsCard
        data={recentQuery.data}
        isLoading={recentQuery.isLoading}
        optimizationFilter={optimizationFilter}
      />
    </div>
  );
}
