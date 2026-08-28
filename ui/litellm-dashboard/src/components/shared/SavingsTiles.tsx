"use client";

import React, { useMemo } from "react";

import SummaryCard from "@/components/shared/SummaryCard";
import {
  autorouterOf,
  cachingOf,
  gatewayAttributedCachingOf,
  compressionOf,
  savedTokensOf,
  usd,
} from "@/app/(dashboard)/cost-optimization/_components/costOptimizationUtils";
import { DailyData } from "@/components/UsagePage/types";
import { formatNumberWithCommas } from "@/utils/dataUtils";

// Exported because the by-driver donut has to slice the same numbers the tiles print, and two
// totalling paths over the same rows is how a chart and the tile above it end up disagreeing.
export const useSavingsTotals = (results: DailyData[]) =>
  useMemo(() => {
    const sumOf = (of: (metrics: DailyData["metrics"]) => number) => results.reduce((sum, d) => sum + of(d.metrics), 0);
    const compression = sumOf(compressionOf);
    const caching = sumOf(cachingOf);
    const autorouter = sumOf(autorouterOf);
    const gatewayAttributedCaching = sumOf(gatewayAttributedCachingOf);
    return {
      compression,
      caching,
      autorouter,
      gatewayAttributedCaching,
      savedTokens: sumOf(savedTokensOf),
      total: compression + gatewayAttributedCaching + autorouter,
    };
  }, [results]);

const SavingsTiles = ({ results, isLoading }: { results: DailyData[]; isLoading: boolean }) => {
  const totals = useSavingsTotals(results);

  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
      <SummaryCard
        label="Total saved"
        value={usd(totals.total)}
        hint={isLoading ? "Loading..." : "Compression + prompt caching + auto-router"}
        info="The sum of the three tiles beside it. Its caching term is the LiteLLM-injected share, so this total is what the gateway itself delivered; caching that clients or providers brought on their own appears only in the caching tile's Total figure."
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
        hint="vs. the priciest model it could pick"
        info="What this traffic would have cost had every request gone to the most expensive model the auto-router can route to, minus what it actually cost. Switching leaves the new model with a cold cache, so it pays to write the prompt again while the baseline is priced as already warm; a route that thrashes the cache can total below zero, and a genuine first turn, where neither side had anything cached, is undercounted."
      />
    </div>
  );
};

export default SavingsTiles;
