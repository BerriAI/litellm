import { useMemo } from "react";
import { LineChart, type ChartColor } from "@/components/shared/charts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DailyData } from "@/components/UsagePage/types";

interface EndpointUsageLineChartProps {
  dailyData?: { results: DailyData[] };
}

// Transform daily data into chart format
function transformDailyDataToChart(dailyData: DailyData[]): Array<Record<string, string | number>> {
  const chartData: Array<Record<string, string | number>> = [];

  // Get all unique endpoint names
  const endpointNames = new Set<string>();
  dailyData.forEach((day) => {
    if (day.breakdown.endpoints) {
      Object.keys(day.breakdown.endpoints).forEach((name) => endpointNames.add(name));
    }
  });

  dailyData.forEach((day) => {
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

  // Reverse the array so most recent dates appear on the right
  return chartData.reverse();
}

export function EndpointUsageLineChart({ dailyData }: EndpointUsageLineChartProps) {
  const chartData = useMemo(() => {
    if (!dailyData?.results || dailyData.results.length === 0) {
      return [];
    }

    return transformDailyDataToChart(dailyData.results);
  }, [dailyData]);

  // Get endpoint names from chart data
  const categories = useMemo(() => {
    if (chartData.length === 0) return [];
    const keys = Object.keys(chartData[0]).filter((key) => key !== "date");
    return keys;
  }, [chartData]);

  // Tremor color palette for multiple lines
  const colors: readonly ChartColor[] = [
    "blue",
    "cyan",
    "indigo",
    "violet",
    "purple",
    "fuchsia",
    "pink",
    "rose",
    "red",
    "orange",
  ];

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle className="text-base font-semibold">Endpoint Usage Trends</CardTitle>
      </CardHeader>
      <CardContent>
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
      </CardContent>
    </Card>
  );
}

export default EndpointUsageLineChart;
