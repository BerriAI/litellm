"use client";

import React, { useState } from "react";

import type { AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import { useAutoRouters } from "@/app/(dashboard)/hooks/models/useModels";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { ApiError } from "@/lib/http/client";
import { formatNumberWithCommas } from "@/utils/dataUtils";

import {
  ALL_ROUTERS,
  bucketRows,
  bucketTurnsTotal,
  durationLabel,
  groupKey,
  expiredMissShare,
  groupLabel,
  pctLabel,
  viewFor,
  type AutoRouterBenchmarksResponse,
  type AutoRouterCacheStats,
  type BenchmarkView,
  type BucketRow,
} from "./autoRouterBenchmarks";
import { formatRangeLabel, usd } from "./costOptimizationUtils";
import ShadowEvalSection from "./ShadowEvalSection";
import TierTurnsChart from "./TierTurnsChart";
import { useAutoRouterBenchmarks } from "./useAutoRouterBenchmarks";
import { DailyActivityRange } from "./useDailyActivityRange";

const Message: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <p className="py-8 text-center text-sm text-muted-foreground">{children}</p>
);

const Metric: React.FC<{ label: string; value: string; hint?: string }> = ({ label, value, hint }) => (
  <Card size="sm">
    <CardHeader>
      <CardTitle className="text-sm font-normal text-muted-foreground">{label}</CardTitle>
    </CardHeader>
    <CardContent className="flex flex-wrap items-baseline gap-2">
      <p className="text-3xl font-semibold text-foreground">{value}</p>
      {hint && <p className="text-sm text-muted-foreground">{hint}</p>}
    </CardContent>
  </Card>
);

const SpendRow: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <dl className="flex items-baseline justify-between gap-6 py-3">
    <dt className="text-sm text-muted-foreground">{label}</dt>
    <dd className="text-base font-semibold tabular-nums text-foreground">{value}</dd>
  </dl>
);

const HeroCard: React.FC<{ view: BenchmarkView }> = ({ view }) => {
  const stats = view.stats;
  const cheaper = stats.saved_spend >= 0;
  return (
    <Card className="overflow-hidden py-0">
      <div className="grid md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="flex flex-col items-center justify-center gap-2 p-6">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Total estimated savings
          </p>
          <div className="flex flex-wrap items-center justify-center gap-3">
            <p className="text-6xl font-semibold tracking-tight text-foreground">{usd(stats.saved_spend)}</p>
            <Badge
              variant="secondary"
              className={`h-6 px-2.5 text-sm ${cheaper ? "bg-success/10 text-success" : "bg-destructive/10 text-destructive"}`}
            >
              {stats.saved_spend !== 0 && (cheaper ? "-" : "+")}
              {Math.abs(stats.saved_pct).toFixed(0)}%
            </Badge>
          </div>
        </div>

        <div className="flex flex-col justify-center border-t p-6 md:border-t-0 md:border-l">
          <SpendRow label="Actual auto-router spend" value={usd(stats.spend)} />
          <Separator />
          <SpendRow label="Estimated spend at highest-tier model" value={usd(stats.baseline_spend)} />
        </div>
      </div>
    </Card>
  );
};

const StackedTurnBar: React.FC<{ buckets: BucketRow[] }> = ({ buckets }) => {
  const segments = buckets.filter((b) => b.turns > 0);
  return (
    <div className="flex flex-col gap-1">
      <div
        className={`flex h-2.5 w-full gap-0.5 overflow-hidden rounded-sm ${segments.length === 0 ? "bg-muted" : ""}`}
        role="img"
        aria-label="Share of turns by bucket"
      >
        {segments.map((b) => (
          <div
            key={b.key}
            className={`${b.fill} first:rounded-l-sm last:rounded-r-sm`}
            style={{ width: `${b.sharePct}%` }}
            title={`${b.label}: ${b.turns.toLocaleString()} turns`}
          />
        ))}
      </div>
      <div className="flex w-full gap-0.5 text-[11px] text-muted-foreground">
        {segments.map((b) => (
          <span key={b.key} className="whitespace-nowrap" style={{ width: `${b.sharePct}%` }}>
            {b.sharePct}%
          </span>
        ))}
      </div>
    </div>
  );
};

const BucketTable: React.FC<{ buckets: BucketRow[] }> = ({ buckets }) => (
  <Table className="border-b">
    <TableHeader>
      <TableRow className="hover:bg-transparent">
        <TableHead className="text-[11px] uppercase tracking-wide">Bucket</TableHead>
        <TableHead className="text-right text-[11px] uppercase tracking-wide">Turns</TableHead>
        <TableHead className="w-1/2" />
        <TableHead className="text-right text-[11px] uppercase tracking-wide">Hit rate</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      {buckets.map((b) => (
        <TableRow key={b.key} className="hover:bg-transparent">
          <TableCell className="text-foreground">
            <span className="flex items-center gap-2">
              <span className={`inline-block size-2 shrink-0 rounded-sm ${b.fill}`} aria-hidden />
              <span>
                {b.label}
                <span className="block text-xs font-normal text-muted-foreground">{b.sublabel}</span>
              </span>
            </span>
          </TableCell>
          <TableCell className="text-right align-middle tabular-nums text-foreground">
            {b.turns.toLocaleString()}
          </TableCell>
          <TableCell className="align-middle">
            <div className="h-1.5 w-full rounded-full bg-muted">
              <div className="h-full rounded-full bg-foreground" style={{ width: `${b.hitRatePct}%` }} aria-hidden />
            </div>
          </TableCell>
          <TableCell className="text-right align-middle font-medium tabular-nums text-foreground">
            {pctLabel(b.hitRatePct)}
          </TableCell>
        </TableRow>
      ))}
    </TableBody>
  </Table>
);

const CachingCard: React.FC<{ cache: AutoRouterCacheStats }> = ({ cache }) => {
  const buckets = bucketRows(cache);
  const total = bucketTurnsTotal(cache);
  const expiredMissPct = expiredMissShare(cache);
  return (
    <Card className="overflow-hidden py-0">
      <div className="grid lg:grid-cols-[1fr_3fr]">
        <div className="flex flex-col border-b p-6 lg:border-b-0 lg:border-r">
          <div className="flex flex-1 flex-col justify-center gap-3">
            <p className="text-sm text-muted-foreground">Cache hit rate</p>
            <p className="text-5xl font-semibold tracking-tight text-foreground">{pctLabel(cache.hit_rate_pct)}</p>
          </div>
          {expiredMissPct === null ? null : (
            <TooltipProvider delay={200}>
              <Tooltip>
                <TooltipTrigger
                  render={
                    <button
                      type="button"
                      className="flex w-full cursor-default items-baseline justify-between gap-2 border-t pt-3 text-left"
                    />
                  }
                >
                  <span className="text-sm text-muted-foreground underline decoration-dotted underline-offset-2">
                    Expired-miss
                  </span>
                  <span className="font-medium tabular-nums text-foreground">{pctLabel(expiredMissPct)}</span>
                </TooltipTrigger>
                <TooltipContent className="max-w-64">
                  share of all measured turns that missed cache because a return to an earlier tier came after its TTL
                  lapsed
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>

        <div className="flex flex-col gap-3 p-6">
          <div className="flex items-baseline justify-between">
            <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Share of turns</p>
            <p className="text-xs text-muted-foreground">
              <span className="text-lg font-semibold tabular-nums text-foreground">{total.toLocaleString()}</span> turns
              measured
            </p>
          </div>
          <StackedTurnBar buckets={buckets} />
          <BucketTable buckets={buckets} />
          {cache.unordered_turns > 0 && (
            <p className="text-xs text-muted-foreground">
              {cache.unordered_turns.toLocaleString()} turns arrived out of order across pods and are not bucketed
            </p>
          )}
        </div>
      </div>
    </Card>
  );
};

interface BenchmarksBodyProps {
  isPending: boolean;
  error: unknown;
  data: AutoRouterBenchmarksResponse | undefined;
  selectedKey: string;
  autoRouters: readonly AutoRouterDeployment[];
}

const BenchmarksBody: React.FC<BenchmarksBodyProps> = ({ isPending, error, data, selectedKey, autoRouters }) => {
  if (isPending) return <Message>Loading auto-router usage...</Message>;
  if (error instanceof ApiError && error.status === 403) {
    return <Message>Auto-router usage is visible to proxy admin roles only</Message>;
  }
  if (error || !data) return <Message>Auto-router usage is unavailable right now</Message>;

  const view = viewFor(data, selectedKey);
  const stats = view.stats;
  return (
    <>
      <HeroCard view={view} />

      <TierTurnsChart view={view} autoRouters={autoRouters} />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          label="Avg saved per session"
          value={usd(stats.saved_per_session)}
          hint={`· ${stats.sessions.toLocaleString()} sessions`}
        />
        <Metric label="Avg turns per session" value={stats.avg_turns_per_session.toFixed(1)} />
        <Metric label="Avg session length" value={durationLabel(stats.avg_session_seconds)} />
        <Metric label="Avg tokens per session" value={formatNumberWithCommas(stats.avg_tokens_per_session, 1, true)} />
      </div>

      <p className="text-xs text-muted-foreground">
        Compares your actual routed spend with the estimated cost of using only the most expensive model configured in
        the auto-router. It accounts for both the cache savings from staying on one model and the added cache costs from
        switching models. The range counts whole sessions that overlap it, so totals can differ slightly from the
        Overall tab, which buckets savings by UTC day.
      </p>

      <div className="space-y-4">
        <div className="flex flex-wrap items-baseline gap-2">
          <h3 className="text-lg font-semibold text-foreground">Auto-router prompt caching</h3>
          <p className="text-xs text-muted-foreground">
            every turn falls in exactly one bucket, by what the router did
          </p>
        </div>
        <CachingCard cache={stats.cache} />
      </div>
    </>
  );
};

interface AutoRouterBenchmarksTabProps {
  accessToken: string | null;
  activity: DailyActivityRange;
}

const UsageView: React.FC<AutoRouterBenchmarksTabProps> = ({ accessToken, activity }) => {
  const { dateValue, onDateChange } = activity;
  const { data, isPending, error } = useAutoRouterBenchmarks(accessToken, dateValue);
  const [selectedKey, setSelectedKey] = useState<string>(ALL_ROUTERS);
  const { data: autoRouters } = useAutoRouters();

  const groups = data?.groups ?? [];
  const selectedLabel = data ? viewFor(data, selectedKey).label : "All auto-routers";
  const rangeLabel = formatRangeLabel(dateValue.from, dateValue.to);

  return (
    <div className="w-full space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-foreground">Auto-router usage</h2>
          {rangeLabel && <p className="mt-1 text-sm text-muted-foreground">{rangeLabel} (UTC)</p>}
        </div>
        <div className="flex w-full flex-col gap-3 sm:w-auto sm:flex-row sm:items-center">
          <AdvancedDatePicker value={dateValue} onValueChange={onDateChange} />
          <div className="w-full sm:w-64">
            <Select value={selectedKey} onValueChange={(value: string | null) => setSelectedKey(value ?? ALL_ROUTERS)}>
              <SelectTrigger className="w-full">
                <SelectValue>{selectedLabel}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL_ROUTERS}>All auto-routers</SelectItem>
                {groups.map((g) => (
                  <SelectItem key={groupKey(g)} value={groupKey(g)}>
                    {groupLabel(g, groups)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      <BenchmarksBody
        isPending={isPending}
        error={error}
        data={data}
        selectedKey={selectedKey}
        autoRouters={autoRouters ?? []}
      />
    </div>
  );
};

const AutoRouterBenchmarksTab: React.FC<AutoRouterBenchmarksTabProps> = ({ accessToken, activity }) => {
  const [visitedTabs, setVisitedTabs] = useState<readonly string[]>(["usage"]);

  const handleTabChange = (value: unknown) => {
    if (typeof value !== "string") {
      return;
    }

    setVisitedTabs((currentTabs) => (currentTabs.includes(value) ? currentTabs : [...currentTabs, value]));
  };

  return (
    <Tabs defaultValue="usage" onValueChange={handleTabChange} className="w-full gap-4">
      <TabsList>
        <TabsTrigger value="usage" className="px-3">
          Usage
        </TabsTrigger>
        <TabsTrigger value="shadow-evals" className="px-3">
          Shadow Evals
        </TabsTrigger>
      </TabsList>

      <TabsContent value="usage" keepMounted={visitedTabs.includes("usage")}>
        <UsageView accessToken={accessToken} activity={activity} />
      </TabsContent>
      <TabsContent value="shadow-evals" keepMounted={visitedTabs.includes("shadow-evals")}>
        <ShadowEvalSection />
      </TabsContent>
    </Tabs>
  );
};

export default AutoRouterBenchmarksTab;
