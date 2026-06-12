import { BarChart, Card, Title } from "@tremor/react";
import React from "react";
import { useTranslation } from "react-i18next";

/**
 * Overview chart: Request Outcomes Over Time (passed vs blocked).
 * Uses Tremor BarChart with stacked data. Data from usage/overview API (chart array).
 */
interface ScoreChartProps {
  data?: Array<{ date: string; passed: number; blocked: number }>;
}

export function ScoreChart({ data }: ScoreChartProps) {
  const { t } = useTranslation();
  const chartData = data && data.length > 0 ? data : [];

  return (
    <Card className="bg-white border border-gray-200">
      <Title className="text-base font-semibold text-gray-900 mb-4">{t("guardrailsMonitor.scoreChart.title")}</Title>
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
          <div className="flex items-center justify-center h-full text-sm text-gray-500">
            {t("guardrailsMonitor.scoreChart.noData")}
          </div>
        )}
      </div>
    </Card>
  );
}
