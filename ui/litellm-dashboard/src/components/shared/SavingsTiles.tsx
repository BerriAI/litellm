"use client";

import React, { useMemo } from "react";

import SummaryCard from "@/components/shared/SummaryCard";
import {
  autorouterOf,
  CachingSavingsScope,
  cachingOf,
  compressionOf,
  gatewayAttributedCachingOf,
  savedTokensOf,
  savingsDriversFor,
  sumOverDays,
  usd,
} from "@/app/(dashboard)/cost-optimization/_components/costOptimizationUtils";
import { DailyData } from "@/components/UsagePage/types";
import { formatNumberWithCommas } from "@/utils/dataUtils";

// The total sums the selected drivers, so it is by construction the sum of what the
// charts plot; the donut and timelines derive from the same list in costOptimizationUtils.
const useSavingsTotals = (results: DailyData[], cachingScope: CachingSavingsScope) =>
  useMemo(
    () => ({
      compression: sumOverDays(results, compressionOf),
      caching: sumOverDays(results, cachingOf),
      autorouter: sumOverDays(results, autorouterOf),
      gatewayAttributedCaching: sumOverDays(results, gatewayAttributedCachingOf),
      savedTokens: sumOverDays(results, savedTokensOf),
      total: savingsDriversFor(cachingScope).reduce((sum, { of }) => sum + sumOverDays(results, of), 0),
    }),
    [results, cachingScope],
  );

interface SavingsTilesProps {
  results: DailyData[];
  isLoading: boolean;
  cachingScope?: CachingSavingsScope;
}

const SavingsTiles = ({ results, isLoading, cachingScope = "litellm-injected" }: SavingsTilesProps) => {
  const totals = useSavingsTotals(results, cachingScope);
  const showAllCaching = cachingScope === "all";
  const totalSavedInfo = showAllCaching
    ? "The sum of compression, all prompt caching, and auto-router savings. The caching term includes client-supplied cache controls and providers that cache implicitly."
    : "The sum of compression, LiteLLM-injected prompt caching, and auto-router savings. Caching that clients or providers brought on their own appears only in the caching tile's Total figure.";
  const promptCachingInfo = showAllCaching
    ? "What all caching saved against paying the input rate for every token, including client-supplied cache controls and providers that cache implicitly. The secondary figure isolates the share LiteLLM earned by inserting cache breakpoints itself."
    : "What LiteLLM-injected caching saved against paying the input rate for every token. The secondary Total also counts requests that arrived with their own cache_control and providers that cache implicitly.";

  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
      <SummaryCard
        label="Total saved"
        value={usd(totals.total)}
        hint={isLoading ? "Loading..." : "Compression + prompt caching + auto-router"}
        info={totalSavedInfo}
      />
      <SummaryCard
        label="Compression savings"
        value={usd(totals.compression)}
        hint={`${formatNumberWithCommas(totals.savedTokens)} tokens compressed`}
        info="Tokens Headroom removed before the call, priced at the model's input rate."
      />
      <SummaryCard
        label="Prompt caching savings"
        value={usd(showAllCaching ? totals.caching : totals.gatewayAttributedCaching)}
        hint={showAllCaching ? "All caching" : "LiteLLM injected"}
        secondary={{
          label: showAllCaching ? "LiteLLM injected" : "Total",
          value: usd(showAllCaching ? totals.gatewayAttributedCaching : totals.caching),
        }}
        info={promptCachingInfo}
      />
      <SummaryCard
        label="Auto-router savings"
        value={usd(totals.autorouter)}
        hint="vs. the priciest model it could pick"
        info="What this traffic would have cost had every request gone to the most expensive model the auto-router can route to, minus what it actually cost. Switching leaves the new model with a cold cache, so it pays to write the prompt again while the baseline is priced as already warm; a route that thrashes the cache can total below zero, and a genuine first turn, where neither side had anything cached, is undercounted."
      />
    </div>
  );
};

export default SavingsTiles;
