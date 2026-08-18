"use client";

import { useMemo, useState } from "react";
import { BarChart } from "@/components/shared/charts";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { ActiveRequest } from "./activeRequestsApi";
import { chartHeight, countBy, countByAge, GROUP_BY, magnitudeFills, type GroupBy } from "./activeRequestGrouping";
import { useIsDarkMode } from "./useIsDarkMode";

export interface ActiveRequestChartsProps {
  items: readonly ActiveRequest[];
  now: number;
}

export default function ActiveRequestCharts({ items, now }: ActiveRequestChartsProps) {
  const [groupBy, setGroupBy] = useState<GroupBy>("model");
  const dark = useIsDarkMode();

  const grouped = useMemo(() => countBy(items, groupBy), [items, groupBy]);
  const byAge = useMemo(() => countByAge(items, now), [items, now]);
  const groupLabel = GROUP_BY.find((option) => option.value === groupBy)?.label ?? "Model";
  const height = chartHeight(grouped.length);

  return (
    <div className="grid gap-4 lg:grid-cols-12">
      <Card className="lg:col-span-7">
        <CardHeader>
          <CardTitle>Running requests by {groupLabel.toLowerCase()}</CardTitle>
          <CardAction>
            <Tabs value={groupBy} onValueChange={(value) => setGroupBy(value as GroupBy)}>
              <TabsList>
                {GROUP_BY.map((option) => (
                  <TabsTrigger key={option.value} value={option.value}>
                    {option.label}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          </CardAction>
        </CardHeader>
        <CardContent>
          {grouped.length === 0 ? (
            <p className="text-sm text-muted-foreground">No running requests on this page.</p>
          ) : (
            <BarChart
              data={grouped}
              index="label"
              categories={["requests"]}
              colors={magnitudeFills(grouped.length, { dark, ascending: true })}
              colorByDatum
              layout="vertical"
              yAxisWidth={190}
              maxBarSize={22}
              showLegend={false}
              valueFormatter={(value) => `${value}`}
              style={{ height }}
            />
          )}
        </CardContent>
      </Card>
      <Card className="lg:col-span-5">
        <CardHeader>
          <CardTitle>How long they have been running</CardTitle>
        </CardHeader>
        <CardContent>
          <BarChart
            data={byAge}
            index="label"
            categories={["requests"]}
            colors={magnitudeFills(byAge.length, { dark, ascending: false })}
            colorByDatum
            maxBarSize={44}
            showLegend={false}
            valueFormatter={(value) => `${value}`}
            style={{ height }}
          />
        </CardContent>
      </Card>
    </div>
  );
}
