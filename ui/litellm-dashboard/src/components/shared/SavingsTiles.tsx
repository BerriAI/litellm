"use client";

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
  const totals = useSavingsTotals(results);

  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
      <SummaryCard
        label="Total recorded savings"
        value={usd(totals.total)}
        hint={isLoading ? "Loading..." : "Compression + prompt caching + auto-router"}
        info="The sum of recorded savings in the three tiles beside it. Auto-router requests without an estimate are excluded. Its caching term is the LiteLLM-injected share; caching supplied by clients or providers appears only in the caching tile's Total figure."
      />
      <SummaryCard
        label="Compression savings"
        value={usd(totals.compression)}
        hint={`${formatNumberWithCommas(totals.savedTokens)} tokens compressed`}
        info="Tokens Headroom removed before the call, priced at the model's input rate."
      />
      <SummaryCard
        label="Prompt caching savings"
        value={usd(totals.gatewayAttributedCaching)}
        hint="LiteLLM injected"
        secondary={{ label: "Total", value: usd(totals.caching) }}
        info="What caching saved against paying the input rate for every token: the discount on tokens served from cache, less the premium providers charge to write a cache entry. The headline figure is the share LiteLLM earned by inserting the breakpoints itself, through configured injection points or auto prompt caching. The total beside it also counts requests that arrived with their own cache_control and providers that cache implicitly. Either can be negative on traffic that writes more cache than it reuses, which is why the headline is not always the smaller of the two."
      />
      <SummaryCard
        label="Auto-router savings"
        value={usd(totals.autorouter)}
        hint="Recorded estimates subtotal"
        info="Sum of available per-request savings estimates against each router's highest-tier baseline, net of classifier cost. Requests without an estimate contribute nothing to this subtotal; this does not mean they saved zero. Historical records retain the estimator used when they were written. The Auto-router usage tab shows coverage for current estimates."
      />
    </div>
  );
};

export default SavingsTiles;
