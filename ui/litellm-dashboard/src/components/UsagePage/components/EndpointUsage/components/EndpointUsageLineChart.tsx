import React, { useMemo, useState } from "react";
import { Card, LineChart, Title } from "@tremor/react";
import { Segmented } from "antd";
import { DailyData } from "../../../types";

interface EndpointUsageLineChartProps {
  dailyData?: { results: DailyData[] };
  endpointData?: Record<string, any>;
}

type TimePeriod = "7" | "30" | "90";


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

  const chartData = useMemo(() => {
    if (!dailyData?.results || dailyData.results.length === 0) {
      return [];
    }

    return transformDailyDataToChart(dailyData.results, days);
  }, [dailyData, days]);

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
