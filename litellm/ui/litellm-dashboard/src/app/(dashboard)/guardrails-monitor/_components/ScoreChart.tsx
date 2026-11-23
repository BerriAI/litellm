import React from "react";
import { useTranslation } from "react-i18next";
import { BarChart } from "@/components/shared/charts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/**
 * Overview chart: Request Outcomes Over Time (passed vs blocked).
 * Stacked bar chart. Data from usage/overview API (chart array).
 */
interface ScoreChartProps {
  data?: Array<{ date: string; passed: number; blocked: number }>;
}

export function ScoreChart({ data }: ScoreChartProps) {
  const { t } = useTranslation();
  const chartData = data && data.length > 0 ? data : [];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base font-semibold">{t("guardrailsMonitor.scoreChart.title")}</CardTitle>
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
            <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
              {t("guardrailsMonitor.scoreChart.empty")}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
