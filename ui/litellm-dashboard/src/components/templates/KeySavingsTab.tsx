"use client";

import React, { useMemo, useState } from "react";

import { AreaChart, BarChart, CustomLegend } from "@/components/shared/charts";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import SummaryCard from "@/components/shared/SummaryCard";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { hasProxyWideSpendView, spendScopeUserId } from "@/utils/roles";
import {
  autorouterOf,
  cacheHitRatio,
  cachingOf,
  compressionOf,
  formatRangeLabel,
  localIsoDay,
  MAX_POINTS_WITH_DOTS,
  pct,
  savedTokensOf,
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

  const compressionTotal = useMemo(() => results.reduce((sum, d) => sum + compressionOf(d.metrics), 0), [results]);
  const cachingTotal = useMemo(() => results.reduce((sum, d) => sum + cachingOf(d.metrics), 0), [results]);
  const autorouterTotal = useMemo(() => results.reduce((sum, d) => sum + autorouterOf(d.metrics), 0), [results]);
  const savedTokensTotal = useMemo(() => results.reduce((sum, d) => sum + savedTokensOf(d.metrics), 0), [results]);
  const totalSaved = compressionTotal + cachingTotal + autorouterTotal;

  // The rows are already filtered to this key server-side, so the ratio comes straight off the
  // day totals -- no breakdown walk, and the same formula the proxy-wide leakage table uses.
  const hitRatio = useMemo(() => {
    const totals = results.reduce(
      (agg, d) => ({
        cacheRead: agg.cacheRead + (d.metrics.cache_read_input_tokens ?? 0),
        prompt: agg.prompt + (d.metrics.prompt_tokens ?? 0),
      }),
      { cacheRead: 0, prompt: 0 },
    );
    return { ratio: cacheHitRatio(totals.cacheRead, totals.prompt), promptTokens: totals.prompt };
  }, [results]);

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

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-5">
        <SummaryCard
          label="Total saved"
          value={usd(totalSaved)}
          hint={isLoading ? "Loading..." : "Compression + prompt caching + auto-router"}
        />
        <SummaryCard
          label="Compression savings"
          value={usd(compressionTotal)}
          hint={`${formatNumberWithCommas(savedTokensTotal)} tokens compressed`}
          info="Tokens Headroom removed before the call, priced at the model's input rate."
        />
        <SummaryCard
          label="Prompt caching savings"
          value={usd(cachingTotal)}
          hint="Cache reads, net of write premium"
          info="What caching saved against paying the input rate for every token: the discount on tokens served from cache, less the premium providers charge to write a cache entry. Can be negative on traffic that writes more cache than it reuses."
        />
        <SummaryCard
          label="Auto-router savings"
          value={usd(autorouterTotal)}
          hint="vs. the priciest model it could pick"
          info="What this traffic would have cost had every request gone to the most expensive model the auto-router can route to, minus what it actually cost. Switching leaves the new model with a cold cache, so it pays to write the prompt again while the baseline is priced as already warm; a route that thrashes the cache can total below zero, and a genuine first turn, where neither side had anything cached, is undercounted."
        />
        <SummaryCard
          label="Cache hit rate"
          value={hitRatio.promptTokens > 0 ? pct(hitRatio.ratio) : "--"}
          hint={
            hitRatio.promptTokens > 0
              ? `${formatNumberWithCommas(hitRatio.promptTokens)} prompt tokens`
              : "No prompt tokens in range"
          }
          info="Share of this key's input tokens that were served from cache. Prompt tokens already include cached reads, so this is cache reads over prompt tokens."
        />
      </div>

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
