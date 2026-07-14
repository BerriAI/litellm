import React from "react";
import { BarChart } from "@/components/shared/charts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { GuardrailUsageChartPoint } from "@/app/(dashboard)/hooks/guardrailsMonitor/useGuardrailsUsageOverview";

/**
 * Overview chart: Request Outcomes Over Time (passed vs blocked).
 * Stacked bar chart. Data from usage/overview API (chart array).
 */
interface ScoreChartProps {
  data?: GuardrailUsageChartPoint[];
}

export function ScoreChart({ data }: ScoreChartProps) {
  const chartData = data && data.length > 0 ? data : [];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base font-semibold">Request Outcomes Over Time</CardTitle>
      </CardHeader>
      <CardContent>
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
              className="h-full"
            />
          ) : (
            <div className="flex items-center justify-center h-full text-sm text-gray-500">
              No chart data for this period
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
