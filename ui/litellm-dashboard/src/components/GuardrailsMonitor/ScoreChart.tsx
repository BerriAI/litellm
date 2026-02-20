import { BarChart, Card, Title } from "@tremor/react";
import React from "react";
import { overviewChartData } from "./mockData";

/**
 * Overview chart: Request Outcomes Over Time (passed vs blocked).
 * Uses Tremor BarChart with stacked data (same stack as UsagePageView patterns).
 */
export function ScoreChart() {
  return (
    <Card className="bg-white border border-gray-200">
      <Title className="text-base font-semibold text-gray-900 mb-4">
        Request Outcomes Over Time
      </Title>
      <div className="h-80 min-h-[280px]">
        <BarChart
          data={overviewChartData}
          index="date"
          categories={["passed", "blocked"]}
          colors={["green", "red"]}
          valueFormatter={(v) => v.toLocaleString()}
          yAxisWidth={48}
          showLegend={true}
          stack={true}
          maxValue={2400}
        />
      </div>
    </Card>
  );
}
