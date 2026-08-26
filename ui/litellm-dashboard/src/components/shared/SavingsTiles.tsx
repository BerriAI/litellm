"use client";

import React, { useMemo } from "react";

import SummaryCard from "@/components/shared/SummaryCard";
import {
  autorouterOf,
  cachingOf,
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
    return {
      compression,
      caching,
      autorouter,
      savedTokens: sumOf(savedTokensOf),
      total: compression + caching + autorouter,
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
      />
      <SummaryCard
        label="Compression savings"
        value={usd(totals.compression)}
        hint={`${formatNumberWithCommas(totals.savedTokens)} tokens compressed`}
        info="Tokens Headroom removed before the call, priced at the model's input rate."
      />
      <SummaryCard
        label="Prompt caching savings"
        value={usd(totals.caching)}
        hint="Cache reads, net of write premium"
        info="What caching saved against paying the input rate for every token: the discount on tokens served from cache, less the premium providers charge to write a cache entry. Can be negative on traffic that writes more cache than it reuses."
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
