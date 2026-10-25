"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import React, { useMemo } from "react";

import SummaryCard from "@/components/shared/SummaryCard";
import {
  autorouterOf,
  cachingOf,
  compressionOf,
  gatewayAttributedCachingOf,
  SAVINGS_DRIVERS,
  savedTokensOf,
  sumOverDays,
  usd,
} from "@/app/(dashboard)/cost-optimization/_components/costOptimizationUtils";
import { DailyData } from "@/components/UsagePage/types";
import { formatNumberWithCommas } from "@/utils/dataUtils";

// The total sums SAVINGS_DRIVERS, so it is by construction the sum of what the
// charts plot; the donut and timelines derive from the same list in costOptimizationUtils.
const useSavingsTotals = (results: DailyData[]) =>
  useMemo(
    () => ({
      compression: sumOverDays(results, compressionOf),
      caching: sumOverDays(results, cachingOf),
      autorouter: sumOverDays(results, autorouterOf),
      gatewayAttributedCaching: sumOverDays(results, gatewayAttributedCachingOf),
      savedTokens: sumOverDays(results, savedTokensOf),
      total: SAVINGS_DRIVERS.reduce((sum, { of }) => sum + sumOverDays(results, of), 0),
    }),
    [results],
  );

const SavingsTiles = ({ results, isLoading }: { results: DailyData[]; isLoading: boolean }) => {
  const { t } = useLanguage();
  const totals = useSavingsTotals(results);

  // 固定提示在传入卡片前翻译，金额和统计值保持原样。
  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
      <SummaryCard
        label={t("Total recorded savings")}
        value={usd(totals.total)}
        hint={isLoading ? t("Loading...") : t("Compression + prompt caching + auto-router")}
        info={t(
          "The sum of recorded savings in the three tiles beside it. Auto-router requests without an estimate are excluded. Its caching term is the LiteLLM-injected share; caching supplied by clients or providers appears only in the caching tile's Total figure.",
        )}
      />
      <SummaryCard
        label={t("Compression savings")}
        value={usd(totals.compression)}
        hint={t("{count} tokens compressed", { count: formatNumberWithCommas(totals.savedTokens) })}
        info={t("Tokens Headroom removed before the call, priced at the model's input rate.")}
      />
      <SummaryCard
        label={t("Prompt caching savings")}
        value={usd(totals.gatewayAttributedCaching)}
        hint={t("LiteLLM injected")}
        secondary={{ label: t("Total"), value: usd(totals.caching) }}
        info={t(
          "What caching saved against paying the input rate for every token: the discount on tokens served from cache, less the premium providers charge to write a cache entry. The headline figure is the share LiteLLM earned by inserting the breakpoints itself, through configured injection points or auto prompt caching. The total beside it also counts requests that arrived with their own cache_control and providers that cache implicitly. Either can be negative on traffic that writes more cache than it reuses, which is why the headline is not always the smaller of the two.",
        )}
      />
      <SummaryCard
        label={t("Auto-router savings")}
        value={usd(totals.autorouter)}
        hint={t("Recorded estimates subtotal")}
        info={t(
          "Sum of available per-request savings estimates against each router's highest-tier baseline, net of classifier cost. Requests without an estimate contribute nothing to this subtotal; this does not mean they saved zero. Historical records retain the estimator used when they were written. The Auto-router usage tab shows coverage for current estimates.",
        )}
      />
    </div>
  );
};

export default SavingsTiles;
