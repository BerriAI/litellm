"use client";

import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

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
  type ScopedActivityInput,
} from "@/app/(dashboard)/cost-optimization/_components/useDailyActivityRange";

interface ScopedSavingsTabProps {
  accessToken: string | null;
  scope: ScopedActivityInput;
  activity: ActivityDateRange;
  entityType: "key" | "user";
  scopeNote?: string;
}

const ScopedSavingsTab = ({ accessToken, scope, activity, entityType, scopeNote }: ScopedSavingsTabProps) => {
  const { t } = useTranslation();
  const { dateValue, onDateChange, results, loading, failed } = useScopedDailyActivityRange(
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

  const intervalLabel = t("shared.scopedSavingsTab.perDay");
  const rangeLabel = formatRangeLabel(startTime, endTime);
  const savingsSubtitle = [
    accumulation === "cumulative"
      ? t("shared.scopedSavingsTab.runningTotal")
      : t("shared.scopedSavingsTab.savedInterval", { interval: intervalLabel.toLowerCase() }),
    rangeLabel && t("shared.scopedSavingsTab.rangeUtc", { range: rangeLabel }),
  ]
    .filter(Boolean)
    .join(" · ");

  const isLoading = loading;
  const unavailable = failed;
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
        <span className="text-sm text-muted-foreground">{t("shared.scopedSavingsTab.bucketedByUtc")}</span>
        <AdvancedDatePicker value={dateValue} onValueChange={onDateChange} />
      </div>

      {scopeNote && (
        <p className="text-sm text-muted-foreground" data-testid={`${entityType}-savings-scope-note`}>
          {scopeNote}
        </p>
      )}

      {unavailable && (
        <p role="alert" className="text-sm text-muted-foreground">
          {t("shared.scopedSavingsTab.unavailable")}
        </p>
      )}
      {showResults && <SavingsTiles results={results} isLoading={false} />}

      <Card>
        <CardHeader>
          <CardTitle>{t("shared.scopedSavingsTab.title")}</CardTitle>
          <CardDescription>{savingsSubtitle}</CardDescription>
          <CardAction className="flex flex-wrap items-center justify-end gap-x-4 gap-y-2">
            <CustomLegend categories={SAVINGS_SERIES} colors={SAVINGS_COLORS} />
            <Tabs value={accumulation} onValueChange={(value) => setAccumulation(value as SavingsAccumulation)}>
              <TabsList>
                <TabsTrigger value="cumulative">{t("shared.scopedSavingsTab.cumulative")}</TabsTrigger>
                <TabsTrigger value="per-interval">{intervalLabel}</TabsTrigger>
              </TabsList>
            </Tabs>
          </CardAction>
        </CardHeader>
        <CardContent>
          {showEmpty && (
            <p className="py-12 text-center text-sm text-muted-foreground" data-testid={`${entityType}-savings-empty`}>
              {isLoading ? t("shared.scopedSavingsTab.loading") : t("shared.scopedSavingsTab.noUsage", { entityType })}
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
