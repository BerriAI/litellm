// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import { BarChart, Card, Title } from "@tremor/react";
import React from "react";

/**
 * Overview chart: Request Outcomes Over Time (passed vs blocked).
 * Uses Tremor BarChart with stacked data. Data from usage/overview API (chart array).
 */
interface ScoreChartProps {
  data?: Array<{ date: string; passed: number; blocked: number }>;
}

export function ScoreChart({ data }: ScoreChartProps) {
  const chartData = data && data.length > 0 ? data : [];

  return (
    <Card className="bg-background border border-border">
      <Title className="text-base font-semibold text-foreground mb-4">
        Request Outcomes Over Time
      </Title>
      <div className="h-80 min-h-[280px]">
        {chartData.length > 0 ? (
          <BarChart
            data={chartData}
            index="date"
            categories={["passed", "blocked"]}
            colors={["green", "red"]}
            valueFormatter={(v) => v.toLocaleString()}
            yAxisWidth={48}
            showLegend={true}
            stack={true}
          />
        ) : (
          <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
            No chart data for this period
          </div>
        )}
      </div>
    </Card>
  );
}
