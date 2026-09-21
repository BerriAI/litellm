"use client";

import { useMemo, useState } from "react";

import { AreaChart, BarChart, CustomLegend } from "@/components/shared/charts";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import SavingsTiles from "@/components/shared/SavingsTiles";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  formatRangeLabel,
  localIsoDay,
  MAX_POINTS_WITH_DOTS,
  SAVINGS_COLORS,
  SAVINGS_SERIES,
  SavingsAccumulation,
  SavingsPoint,
  savingsSeriesOf,
  shortDate,
  toCumulative,
  usd,
  withStartAnchor,
} from "@/app/(dashboard)/cost-optimization/_components/costOptimizationUtils";
import {
  useScopedDailyActivityRange,
  type ActivityDateRange,
  type DailyActivityScope,
} from "@/app/(dashboard)/cost-optimization/_components/useDailyActivityRange";

interface ScopedSavingsTabProps {
  accessToken: string | null;
  scope: DailyActivityScope;
  activity: ActivityDateRange;
  entityType: "key" | "user";
  scopeNote?: string;
}

const ScopedSavingsTab = ({ accessToken, scope, activity, entityType, scopeNote }: ScopedSavingsTabProps) => {
  const { dateValue, onDateChange, results, loading, isFetchingMore, failed, cancelled } = useScopedDailyActivityRange(
    accessToken,
    scope,
    activity,
  );
  const startTime = dateValue.from;
  const endTime = dateValue.to;

  const [accumulation, setAccumulation] = useState<SavingsAccumulation>("cumulative");

  const perInterval = useMemo<SavingsPoint[]>(() => savingsSeriesOf(results), [results]);

  const overTime = useMemo(() => {
    if (accumulation !== "cumulative") return perInterval;
    const startLabel = startTime ? shortDate(localIsoDay(startTime)) : "";
    return withStartAnchor(toCumulative(perInterval), startLabel);
  }, [accumulation, perInterval, startTime]);

  const intervalLabel = "Per day";
  const rangeLabel = formatRangeLabel(startTime, endTime);
  const savingsSubtitle = [
    accumulation === "cumulative" ? "Running total saved" : `Saved ${intervalLabel.toLowerCase()}`,
    rangeLabel && `${rangeLabel} (UTC)`,
  ]
    .filter(Boolean)
    .join(" · ");

  const isLoading = loading || isFetchingMore;
  const unavailable = failed || cancelled;
  const showResults = !isLoading && !unavailable;
  const hasRows = results.length > 0;
  const showEmpty = !unavailable && (isLoading || !hasRows);
  const showChart = showResults && hasRows;
  const chartProps = {
    data: overTime,
    index: "date",
    categories: SAVINGS_SERIES,
    colors: SAVINGS_COLORS,
    valueFormatter: usd,
    showLegend: false,
  };

  return (
    <div className="w-full space-y-6">
      <div className="flex flex-wrap items-center justify-end gap-4">
        <span className="text-sm text-muted-foreground">Spend is bucketed by UTC day</span>
        <AdvancedDatePicker value={dateValue} onValueChange={onDateChange} />
      </div>

      {scopeNote && (
        <p className="text-sm text-muted-foreground" data-testid={`${entityType}-savings-scope-note`}>
          {scopeNote}
        </p>
      )}

      {unavailable && (
        <p role="alert" className="text-sm text-muted-foreground">
          Savings are unavailable for this range. Try another date range or reopen this tab.
        </p>
      )}
      {showResults && <SavingsTiles results={results} isLoading={false} />}

      <Card>
        <CardHeader>
          <CardTitle>Savings</CardTitle>
          <CardDescription>{savingsSubtitle}</CardDescription>
          <CardAction className="flex flex-wrap items-center justify-end gap-x-4 gap-y-2">
            <CustomLegend categories={SAVINGS_SERIES} colors={SAVINGS_COLORS} />
            <Tabs value={accumulation} onValueChange={(value) => setAccumulation(value as SavingsAccumulation)}>
              <TabsList>
                <TabsTrigger value="cumulative">Cumulative</TabsTrigger>
                <TabsTrigger value="per-interval">{intervalLabel}</TabsTrigger>
              </TabsList>
            </Tabs>
          </CardAction>
        </CardHeader>
        <CardContent>
          {showEmpty && (
            <p className="py-12 text-center text-sm text-muted-foreground" data-testid={`${entityType}-savings-empty`}>
              {isLoading ? "Loading savings..." : `No usage recorded for this ${entityType} in this range.`}
            </p>
          )}
          {showChart &&
            (accumulation === "cumulative" ? (
              <AreaChart {...chartProps} showDots={overTime.length <= MAX_POINTS_WITH_DOTS} />
            ) : (
              <BarChart {...chartProps} />
            ))}
        </CardContent>
      </Card>
    </div>
  );
};

export default ScopedSavingsTab;
