"use client";

import { useMemo, type ReactNode } from "react";
import type { TFunction } from "i18next";
import { Trans, useTranslation } from "react-i18next";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import type { DailyData } from "@/components/UsagePage/types";
import { cn } from "@/lib/cva.config";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { Skeleton } from "@/components/ui/skeleton";
import { dailyTotals, type OverviewTotals } from "./overviewData";
import { PANEL_INSET_X, Sparkline, Stat } from "./Primitives";
import type { Change, StatComparison } from "./usageComparison";
import { formatDayRange } from "./usageTrendData";

const BRAND = "#2b3fd6";
const ONE_DECIMAL = { minimumFractionDigits: 1, maximumFractionDigits: 1 } as const;
const FOUR_DECIMALS = { minimumFractionDigits: 4, maximumFractionDigits: 4 } as const;
const MILLISECONDS = { style: "unit", unit: "millisecond", unitDisplay: "narrow", maximumFractionDigits: 0 } as const;
const SECONDS = { ...ONE_DECIMAL, style: "unit", unit: "second", unitDisplay: "narrow" } as const;

interface UsageStatStripProps {
  results: readonly DailyData[];
  totals: OverviewTotals;
  loading: boolean;
  requestCountsPending: boolean;
  budget: number | null;
  showBudgetHint?: boolean;
  comparison?: StatComparison;
  failed?: boolean;
  requestsFailed?: boolean;
}

function ChangeLine({ change, period }: { change: Change; period: string }) {
  const { t } = useTranslation();
  const vs = t("usage.statStrip.comparison.vs", { period });
  if (change.kind === "pending") return <Skeleton className="mt-1 h-4 w-28" />;
  const rising = change.kind === "change" && change.percent >= 0;
  const text = (() => {
    if (change.kind === "unavailable") return t("usage.statStrip.comparison.unavailable");
    if (change.kind === "new") return t("usage.statStrip.comparison.new");
    if (change.kind === "same") return t("usage.statStrip.comparison.same");
    const magnitude = t("format.number", { val: Math.abs(change.percent), ...ONE_DECIMAL });
    return t(rising ? "usage.statStrip.comparison.up" : "usage.statStrip.comparison.down", { percent: magnitude });
  })();
  const Icon = rising ? ArrowUpRight : ArrowDownRight;
  return (
    <div
      className="mt-1 flex min-w-0 items-center gap-1 text-xs leading-4 text-muted-foreground"
      title={`${text} ${vs}`}
    >
      {change.kind === "change" && <Icon aria-hidden="true" className="size-3 shrink-0" />}
      <span className="font-medium tabular-nums text-foreground">{text}</span> <span className="truncate">{vs}</span>
    </div>
  );
}

interface CellProps {
  totals: OverviewTotals;
  loading: boolean;
  failed: boolean;
  daily: ReturnType<typeof dailyTotals>;
  change: ReactNode;
}

function SpendCell({ totals, loading, failed, daily, change, hint }: CellProps & { hint: string | undefined }) {
  const { t } = useTranslation();
  const label = t("usage.statStrip.spend");
  return (
    <StatCell label={label} trend={<Sparkline data={daily} dataKey="spend" color={BRAND} className="h-14" />}>
      <Stat
        label={label}
        value={failed ? t("usage.statStrip.unavailable") : t("format.usd", { val: totals.spend })}
        pending={loading}
        hint={hint}
      />
      {change}
    </StatCell>
  );
}

function RequestCounts({ totals }: { totals: OverviewTotals }) {
  const { t } = useTranslation();
  return (
    <div className="mt-0.5 flex gap-3 text-xs leading-4 text-muted-foreground">
      <span>
        <Trans
          i18nKey="usage.statStrip.successfulCount"
          components={{
            count: <span className="text-success tabular-nums">{t("format.number", { val: totals.successful })}</span>,
          }}
        />
      </span>
      <span>
        <Trans
          i18nKey="usage.statStrip.failedCount"
          components={{
            count: (
              <span className={totals.failed > 0 ? "text-destructive tabular-nums" : "tabular-nums"}>
                {t("format.number", { val: totals.failed })}
              </span>
            ),
          }}
        />
      </span>
    </div>
  );
}

function RequestsCell({ totals, loading, failed, daily, change }: CellProps) {
  const { t } = useTranslation();
  const label = t("usage.statStrip.totalRequests");
  return (
    <StatCell label={label} trend={<Sparkline data={daily} dataKey="requests" color={BRAND} className="h-14" />}>
      <Stat
        label={label}
        value={failed ? t("usage.statStrip.unavailable") : t("format.number", { val: totals.requests })}
        pending={loading}
      />
      {!loading && change}
      {!loading && !failed && <RequestCounts totals={totals} />}
    </StatCell>
  );
}

function TokenBreakdownPopover({ totals }: { totals: OverviewTotals }) {
  const { t } = useTranslation();
  return (
    <Popover>
      <PopoverTrigger className="mt-1 text-xs font-medium text-primary underline-offset-4 hover:underline">
        {t("usage.statStrip.viewTokenBreakdown")}
      </PopoverTrigger>
      <PopoverContent side="right" align="start" className="w-80 gap-3">
        <PopoverTitle>{t("usage.statStrip.tokenPopover.title")}</PopoverTitle>
        <dl className="space-y-2 text-sm tabular-nums">
          <div className="flex justify-between gap-4">
            <dt>{t("usage.statStrip.tokenPopover.inputTokens")}</dt>
            <dd>{t("format.number", { val: totals.inputTokens })}</dd>
          </div>
          <div className="flex justify-between gap-4 pl-3 text-xs text-muted-foreground">
            <dt>{t("usage.statStrip.tokenPopover.cacheReads")}</dt>
            <dd>{t("format.number", { val: totals.cacheReadTokens })}</dd>
          </div>
          <div className="flex justify-between gap-4 pl-3 text-xs text-muted-foreground">
            <dt>{t("usage.statStrip.tokenPopover.cacheWrites")}</dt>
            <dd>{t("format.number", { val: totals.cacheWriteTokens })}</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt>{t("usage.statStrip.tokenPopover.outputTokens")}</dt>
            <dd>{t("format.number", { val: totals.outputTokens })}</dd>
          </div>
          <div className="flex justify-between gap-4 border-t pt-2 font-medium">
            <dt>{t("usage.statStrip.tokenPopover.totalTokens")}</dt>
            <dd>{t("format.number", { val: totals.tokens })}</dd>
          </div>
        </dl>
        <p className="text-xs leading-5 text-muted-foreground">{t("usage.statStrip.tokenPopover.pricingNote")}</p>
        <p className="text-xs leading-5 text-muted-foreground">{t("usage.statStrip.tokenPopover.clientNote")}</p>
      </PopoverContent>
    </Popover>
  );
}

function TokensCell({ totals, loading, failed, daily, change }: CellProps) {
  const { t } = useTranslation();
  const label = t("usage.statStrip.totalTokens");
  const ready = !loading && !failed;
  const value = (() => {
    if (failed) return t("usage.statStrip.unavailable");
    if (loading) return null;
    return t("format.compactNumber", { val: totals.tokens });
  })();
  return (
    <StatCell label={label} trend={<Sparkline data={daily} dataKey="tokens" color={BRAND} className="h-14" />}>
      <Stat
        label={label}
        value={value}
        exact={
          failed ? undefined : t("usage.statStrip.exactTokens", { tokens: t("format.number", { val: totals.tokens }) })
        }
        pending={loading}
        hint={
          ready
            ? t("usage.statStrip.cacheReadHint", { tokens: t("format.compactNumber", { val: totals.cacheReadTokens }) })
            : undefined
        }
      />
      {change}
      {ready && <TokenBreakdownPopover totals={totals} />}
    </StatCell>
  );
}

function AveragesCell({ totals, loading, failed, change }: CellProps) {
  const { t } = useTranslation();
  const label = t("usage.statStrip.avgCostPerRequest");
  const successHint =
    totals.successRate === null
      ? undefined
      : t("usage.statStrip.successRateHint", { rate: t("format.number", { val: totals.successRate, ...ONE_DECIMAL }) });
  return (
    <StatCell label={label}>
      <div className="grid gap-3">
        <Stat
          label={label}
          value={
            failed
              ? t("usage.statStrip.unavailable")
              : t("format.usd", { val: totals.avgCostPerRequest, ...FOUR_DECIMALS })
          }
          pending={loading}
        />
        {change}
        <Stat
          label={t("usage.statStrip.avgLatency")}
          value={totals.avgLatencyMs === null || failed ? "—" : latency(totals.avgLatencyMs, t)}
          pending={loading}
          hint={successHint}
        />
      </div>
    </StatCell>
  );
}

/** Spend, requests, tokens and averages for a range, each with a daily sparkline. */
export default function UsageStatStrip({
  results,
  totals,
  loading,
  requestCountsPending,
  budget,
  showBudgetHint = true,
  comparison,
  failed = false,
  requestsFailed = failed,
}: UsageStatStripProps) {
  const { t } = useTranslation();
  const daily = useMemo(() => dailyTotals(results), [results]);
  const period = comparison ? formatDayRange(comparison.previous) : "";
  const changeFor = (change: Change | null | undefined) =>
    change && !loading && !failed ? <ChangeLine change={change} period={period} /> : null;
  const budgetHint = budget
    ? t("usage.statStrip.budgetHint", { budget: t("format.usd", { val: budget }) })
    : t("usage.statStrip.noBudget");
  const shared = { totals, loading, failed, daily };
  return (
    <div className="grid grid-cols-1 overflow-hidden rounded-xl border bg-card sm:grid-cols-2 lg:grid-cols-4 lg:divide-x">
      <SpendCell {...shared} change={changeFor(comparison?.spend)} hint={showBudgetHint ? budgetHint : undefined} />
      <RequestsCell
        {...shared}
        loading={requestCountsPending}
        failed={requestsFailed}
        change={changeFor(comparison?.requests)}
      />
      <TokensCell {...shared} change={changeFor(comparison?.tokens)} />
      <AveragesCell {...shared} change={changeFor(comparison?.avgCostPerRequest)} />
    </div>
  );
}

function latency(ms: number, t: TFunction): string {
  return ms < 1000
    ? t("format.number", { val: ms, ...MILLISECONDS })
    : t("format.number", { val: ms / 1000, ...SECONDS });
}

function StatCell({ label, children, trend }: { label: string; children: ReactNode; trend?: ReactNode }) {
  return (
    <div
      role="group"
      aria-label={label}
      className={cn(
        "flex min-w-0 flex-col justify-between gap-3 border-b py-4 last:border-b-0 lg:border-b-0",
        PANEL_INSET_X,
      )}
    >
      <div className="min-w-0">{children}</div>
      {trend}
    </div>
  );
}
