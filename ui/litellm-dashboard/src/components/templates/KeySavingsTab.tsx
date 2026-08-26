"use client";

import React, { useMemo, useState } from "react";

import { AreaChart, BarChart, CustomLegend } from "@/components/shared/charts";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import SavingsTiles from "@/components/shared/SavingsTiles";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { hasProxyWideSpendView, spendScopeUserId } from "@/utils/roles";
import {
  autorouterOf,
  cachingOf,
  compressionOf,
  formatRangeLabel,
  localIsoDay,
  MAX_POINTS_WITH_DOTS,
  SAVINGS_COLORS,
  SAVINGS_SERIES,
  SavingsAccumulation,
  SavingsPoint,
  shortDate,
  toCumulative,
  usd,
  withStartAnchor,
} from "@/app/(dashboard)/cost-optimization/_components/costOptimizationUtils";
import { useScopedDailyActivityRange } from "@/app/(dashboard)/cost-optimization/_components/useDailyActivityRange";

interface KeySavingsTabProps {
  accessToken: string | null;
  /** The key's token hash — what spend rows are keyed by, not the one-time plaintext secret. */
  keyToken: string;
  userId: string | null;
  userRole: string;
}

const KeySavingsTab: React.FC<KeySavingsTabProps> = ({ accessToken, keyToken, userId, userRole }) => {
  // Proxy admins read the whole key. For anyone else the endpoint applies the caller's own user_id
  // alongside the key filter, so the figures cover only that viewer's requests on this key -- said
  // plainly in the scope note below rather than left to be misread as the key's total.
  const readsWholeKey = hasProxyWideSpendView(userRole);
  const activity = useScopedDailyActivityRange(accessToken, {
    userId: spendScopeUserId(userRole, userId),
    apiKey: keyToken,
  });

  const { dateValue, onDateChange, results, loading, isFetchingMore } = activity;
  const startTime = dateValue.from ?? null;
  const endTime = dateValue.to ?? null;

  const [accumulation, setAccumulation] = useState<SavingsAccumulation>("cumulative");

  // Sort on the raw ISO date before shortDate() drops the year: the rollup arrives newest
  // first, and the running total has to accumulate forward in time.
  const perInterval = useMemo<SavingsPoint[]>(
    () =>
      [...results]
        .sort((a, b) => a.date.localeCompare(b.date))
        .map((d) => ({
          date: shortDate(d.date),
          Compression: compressionOf(d.metrics),
          "Prompt caching": cachingOf(d.metrics),
          "Auto-router": autorouterOf(d.metrics),
        })),
    [results],
  );

  const overTime = useMemo(() => {
    if (accumulation !== "cumulative") return perInterval;
    const startLabel = startTime ? shortDate(localIsoDay(startTime)) : "";
    return withStartAnchor(toCumulative(perInterval), startLabel);
  }, [accumulation, perInterval, startTime]);

  const intervalLabel = "Per day";
  const rangeLabel = formatRangeLabel(startTime ?? undefined, endTime ?? undefined);
  const savingsSubtitle = [
    accumulation === "cumulative" ? "Running total saved" : `Saved ${intervalLabel.toLowerCase()}`,
    rangeLabel && `${rangeLabel} (UTC)`,
  ]
    .filter(Boolean)
    .join(" · ");

  const isLoading = loading || isFetchingMore;
  const hasRows = results.length > 0;
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

      {!readsWholeKey && (
        <p className="text-sm text-muted-foreground" data-testid="key-savings-scope-note">
          Showing your own requests on this key. A key shared across a team will have spend from other members that is
          not counted here.
        </p>
      )}

      <SavingsTiles results={results} isLoading={isLoading} />

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
          {/* Distinguishes "still fetching" from "this key genuinely had no traffic": an empty
              chart alone reads as a broken panel, and a $0.00 tile reads as a real zero. */}
          {!hasRows && (
            <p className="py-12 text-center text-sm text-muted-foreground" data-testid="key-savings-empty">
              {isLoading ? "Loading savings..." : "No usage recorded for this key in this range."}
            </p>
          )}
          {hasRows && accumulation === "cumulative" && (
            <AreaChart {...chartProps} showDots={overTime.length <= MAX_POINTS_WITH_DOTS} />
          )}
          {/* Not stacked: auto-router can go negative on a cold-cache write, and stacking would
              draw that below the axis while the rest of the bar still read as the total. */}
          {hasRows && accumulation !== "cumulative" && <BarChart {...chartProps} />}
        </CardContent>
      </Card>
    </div>
  );
};

export default KeySavingsTab;
