import { useMemo } from "react";
import { ChartLine } from "lucide-react";
import { LineChart, stackedUsageColor } from "@/components/shared/charts";
import { DailyData } from "@/components/UsagePage/types";
import { Panel } from "../../overview/Primitives";

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

  const colors = useMemo(() => categories.map((_, i) => stackedUsageColor(i)), [categories]);

  return (
    <Panel icon={ChartLine} title="Endpoint Usage Trends">
      <LineChart
        className="h-64"
        data={chartData}
        index="date"
        categories={categories}
        colors={colors}
        valueFormatter={(value) => value.toLocaleString()}
        showLegend={true}
        showGridLines={true}
        yAxisWidth={56}
        connectNulls={true}
        curveType="monotone"
      />
    </Panel>
  );
}

export default EndpointUsageLineChart;
