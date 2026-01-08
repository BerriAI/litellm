import React, { useMemo, useState } from "react";
import { Card, LineChart, Title } from "@tremor/react";
import { Segmented } from "antd";
import { BreakdownMetrics, DailyData, MetricWithMetadata } from "../../../types";
import { mockEndpointData } from "../mockEndpointData";

interface EndpointUsageLineChartProps {
  dailyData?: { results: DailyData[] };
  endpointData?: Record<string, MetricWithMetadata>;
}

type TimePeriod = "7" | "30" | "90";

// Create mock daily data from aggregated endpoint data
function createMockDailyData(endpointData: Record<string, MetricWithMetadata>, days: number): DailyData[] {
  const today = new Date();
  const dailyResults: DailyData[] = [];

  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    const dateStr = date.toISOString().split("T")[0];

    // Distribute aggregated totals across days with some variation
    const breakdown: { [key: string]: MetricWithMetadata } = {};
    Object.entries(endpointData).forEach(([endpointName, endpoint]) => {
      // Calculate average daily values
      const avgDaily = {
        spend: endpoint.metrics.spend / days,
        prompt_tokens: endpoint.metrics.prompt_tokens / days,
        completion_tokens: endpoint.metrics.completion_tokens / days,
        total_tokens: endpoint.metrics.total_tokens / days,
        api_requests: endpoint.metrics.api_requests / days,
        successful_requests: endpoint.metrics.successful_requests / days,
        failed_requests: endpoint.metrics.failed_requests / days,
        cache_read_input_tokens: endpoint.metrics.cache_read_input_tokens / days,
        cache_creation_input_tokens: endpoint.metrics.cache_creation_input_tokens / days,
      };

      // Add simple variation (±20%) to make it look more realistic
      const variation = 0.8 + Math.random() * 0.4; // Random factor between 0.8 and 1.2

      breakdown[endpointName] = {
        metrics: {
          spend: avgDaily.spend * variation,
          prompt_tokens: Math.round(avgDaily.prompt_tokens * variation),
          completion_tokens: Math.round(avgDaily.completion_tokens * variation),
          total_tokens: Math.round(avgDaily.total_tokens * variation),
          api_requests: Math.round(avgDaily.api_requests * variation),
          successful_requests: Math.round(avgDaily.successful_requests * variation),
          failed_requests: Math.round(avgDaily.failed_requests * variation),
          cache_read_input_tokens: Math.round(avgDaily.cache_read_input_tokens * variation),
          cache_creation_input_tokens: Math.round(avgDaily.cache_creation_input_tokens * variation),
        },
        metadata: endpoint.metadata,
        api_key_breakdown: endpoint.api_key_breakdown,
      };
    });

    const breakdownMetrics: BreakdownMetrics = {
      models: {},
      model_groups: {},
      mcp_servers: {},
      providers: {},
      api_keys: {},
      entities: {},
      endpoints: breakdown,
    };

    dailyResults.push({
      date: dateStr,
      metrics: {
        spend: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        api_requests: 0,
        successful_requests: 0,
        failed_requests: 0,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      breakdown: breakdownMetrics,
    });
  }

  return dailyResults;
}

// Transform daily data into chart format
function transformDailyDataToChart(dailyData: DailyData[], days: number): Array<Record<string, string | number>> {
  const chartData: Array<Record<string, string | number>> = [];

  // Get all unique endpoint names
  const endpointNames = new Set<string>();
  dailyData.forEach((day) => {
    if (day.breakdown.endpoints) {
      Object.keys(day.breakdown.endpoints).forEach((name) => endpointNames.add(name));
    }
  });

  // Filter to the last N days based on time period
  const filteredData = dailyData.slice(-days);

  filteredData.forEach((day) => {
    const date = new Date(day.date);
    const dateStr = date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    });

    const dataPoint: Record<string, string | number> = {
      date: dateStr,
    };

    endpointNames.forEach((endpointName) => {
      const endpoint = day.breakdown.endpoints?.[endpointName];
      dataPoint[endpointName] = endpoint?.metrics.api_requests || 0;
    });

    chartData.push(dataPoint);
  });

  return chartData;
}

export function EndpointUsageLineChart({ dailyData, endpointData }: EndpointUsageLineChartProps) {
  const [timePeriod, setTimePeriod] = useState<TimePeriod>("7");
  const days = parseInt(timePeriod);

  // Use daily data if provided and has endpoints, otherwise create mock daily data from aggregated endpoint data
  const chartData = useMemo(() => {
    // Check if dailyData has endpoints breakdown
    const hasEndpointsInDailyData =
      dailyData?.results &&
      dailyData.results.some((day) => day.breakdown.endpoints && Object.keys(day.breakdown.endpoints).length > 0);

    if (hasEndpointsInDailyData) {
      // Check if we have a reasonable number of endpoints (at least 2 for proof of concept)
      const endpointCount = new Set<string>();
      dailyData.results.forEach((day) => {
        if (day.breakdown.endpoints) {
          Object.keys(day.breakdown.endpoints).forEach((name) => endpointCount.add(name));
        }
      });

      // If we have multiple endpoints in real data, use it; otherwise use mock data for proof of concept
      if (endpointCount.size >= 2) {
        return transformDailyDataToChart(dailyData.results, days);
      }
    }

    // Use mock data for proof of concept
    const dataToUse = endpointData || mockEndpointData;
    const mockDaily = createMockDailyData(dataToUse, days);
    return transformDailyDataToChart(mockDaily, days);
  }, [dailyData, endpointData, days]);

  // Get endpoint names from chart data
  const categories = useMemo(() => {
    if (chartData.length === 0) return [];
    const keys = Object.keys(chartData[0]).filter((key) => key !== "date");
    return keys;
  }, [chartData]);

  // Tremor color palette for multiple lines
  const colors = ["blue", "cyan", "indigo", "violet", "purple", "fuchsia", "pink", "rose", "red", "orange"];

  return (
    <Card className="mb-6">
      <div className="flex items-center justify-between mb-4">
        <Title>Endpoint Usage Trends</Title>
        <Segmented
          options={[
            {
              label: "7 Days",
              value: "7",
            },
            {
              label: "30 Days",
              value: "30",
            },
            {
              label: "90 Days",
              value: "90",
            },
          ]}
          value={timePeriod}
          onChange={(value) => setTimePeriod(value as TimePeriod)}
          style={{
            backgroundColor: "#f3f4f6",
          }}
        />
      </div>
      <LineChart
        className="h-80"
        data={chartData}
        index="date"
        categories={categories}
        colors={colors.slice(0, categories.length)}
        valueFormatter={(value) => value.toLocaleString()}
        showLegend={true}
        showGridLines={true}
        yAxisWidth={60}
        connectNulls={true}
        curveType="natural"
      />
    </Card>
  );
}

export default EndpointUsageLineChart;
