"use client";

import { useQuery } from "@tanstack/react-query";
import { Alert, Card, Table, Tag, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import { ComponentProps, useState } from "react";
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
} from "@/components/networking";

type DateRangeValue = ComponentProps<typeof AdvancedDatePicker>["value"];

const SERIES_CATEGORIES = ["Caching", "Compression"] as const;
const SERIES_COLORS: readonly ChartColor[] = ["emerald", "blue"];

const OPTIMIZATION_TAG_COLOR: Record<CostOptimizationType, string> = {
  caching: "green",
  compression: "blue",
};

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
}

function SavingsKpiGrid({ totals }: SavingsKpiGridProps) {
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

export default function CostSavingsView() {
  const { accessToken } = useAuthorized();
  const [dateValue, setDateValue] = useState<DateRangeValue>(defaultDateRange);

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
  const recentRequests = recentQuery.data?.requests ?? [];

  return (
    <div className="w-full p-8">
      <div className="flex items-end justify-between gap-6 mb-6">
        <div>
          <Typography.Title level={3} className="mb-0!">
            Cost Savings
          </Typography.Title>
          <Typography.Text type="secondary">Savings from prompt caching and prompt compression</Typography.Text>
        </div>
        <AdvancedDatePicker value={dateValue} onValueChange={setDateValue} label="" showTimeRange={false} />
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

      <SavingsKpiGrid totals={totals} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        <Card className="lg:col-span-2 border border-gray-200 rounded-lg">
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
              categories={SERIES_CATEGORIES}
              colors={SERIES_COLORS}
              valueFormatter={formatUsd}
              yAxisWidth={80}
            />
          )}
        </Card>
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
      </div>

      <Card className="border border-gray-200 rounded-lg" styles={{ body: { padding: 0 } }}>
        <div className="p-6 pb-4">
          <Typography.Title level={5} className="mb-0!">
            Recent Optimized Requests
          </Typography.Title>
          <Typography.Text type="secondary">
            Latest requests that benefited from caching or compression
            {recentQuery.data ? ` (scanned last ${recentQuery.data.scanned_requests} requests in range)` : ""}
          </Typography.Text>
        </div>
        {recentQuery.isLoading ? (
          <div className="p-6">
            <ChartLoader />
          </div>
        ) : (
          <Table
            columns={RECENT_REQUEST_COLUMNS}
            dataSource={recentRequests}
            rowKey="request_id"
            pagination={false}
            size="middle"
            locale={{
              emptyText:
                "No optimized requests in this window. Savings appear here once prompt caching or prompt compression kicks in.",
            }}
          />
        )}
      </Card>
    </div>
  );
}
