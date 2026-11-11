"use client";

import React, { useMemo } from "react";
import type { TFunction } from "i18next";
import { Trans, useTranslation } from "react-i18next";
import type { DailyData } from "@/components/UsagePage/types";
import { cn } from "@/lib/cva.config";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { dailyTotals, type OverviewTotals } from "./overviewData";
import { PANEL_INSET_X, Sparkline, Stat } from "./Primitives";

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
}

/** Spend, requests, tokens and averages for a range, each with a daily sparkline. */
export default function UsageStatStrip({
  results,
  totals,
  loading,
  requestCountsPending,
  budget,
}: UsageStatStripProps) {
  const { t } = useTranslation();
  const daily = useMemo(() => dailyTotals(results), [results]);
  return (
    <div className="grid grid-cols-1 overflow-hidden rounded-xl border bg-card sm:grid-cols-2 lg:grid-cols-4 lg:divide-x">
      <StatCell trend={<Sparkline data={daily} dataKey="spend" color={BRAND} className="h-14" />}>
        <Stat
          label={t("usage.statStrip.spend")}
          value={t("format.usd", { val: totals.spend })}
          pending={loading}
          hint={
            budget
              ? t("usage.statStrip.budgetHint", { budget: t("format.usd", { val: budget }) })
              : t("usage.statStrip.noBudget")
          }
        />
      </StatCell>
      <StatCell trend={<Sparkline data={daily} dataKey="requests" color={BRAND} className="h-14" />}>
        <Stat
          label={t("usage.statStrip.totalRequests")}
          value={t("format.number", { val: totals.requests })}
          pending={requestCountsPending}
        />
        {!requestCountsPending && (
          <div className="mt-0.5 flex gap-3 text-xs leading-4 text-muted-foreground">
            <span>
              <Trans
                i18nKey="usage.statStrip.successfulCount"
                components={{
                  count: (
                    <span className="text-success tabular-nums">{t("format.number", { val: totals.successful })}</span>
                  ),
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
        )}
      </StatCell>
      <StatCell trend={<Sparkline data={daily} dataKey="tokens" color={BRAND} className="h-14" />}>
        <Stat
          label={t("usage.statStrip.totalTokens")}
          value={loading ? null : t("format.compactNumber", { val: totals.tokens })}
          exact={t("usage.statStrip.exactTokens", { tokens: t("format.number", { val: totals.tokens }) })}
          pending={loading}
          hint={
            loading
              ? undefined
              : t("usage.statStrip.cacheReadHint", {
                  tokens: t("format.compactNumber", { val: totals.cacheReadTokens }),
                })
          }
        />
        {!loading && (
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
        )}
      </StatCell>
      <StatCell>
        <div className="grid gap-3">
          <Stat
            label={t("usage.statStrip.avgCostPerRequest")}
            value={t("format.usd", { val: totals.avgCostPerRequest, ...FOUR_DECIMALS })}
            pending={loading}
          />
          <Stat
            label={t("usage.statStrip.avgLatency")}
            value={totals.avgLatencyMs === null ? "—" : latency(totals.avgLatencyMs, t)}
            pending={loading}
            hint={
              totals.successRate === null
                ? undefined
                : t("usage.statStrip.successRateHint", {
                    rate: t("format.number", { val: totals.successRate, ...ONE_DECIMAL }),
                  })
            }
          />
        </div>
      </StatCell>
    </div>
  );
}

function latency(ms: number, t: TFunction): string {
  return ms < 1000
    ? t("format.number", { val: ms, ...MILLISECONDS })
    : t("format.number", { val: ms / 1000, ...SECONDS });
}

function StatCell({ children, trend }: { children: React.ReactNode; trend?: React.ReactNode }) {
  return (
    <div
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
