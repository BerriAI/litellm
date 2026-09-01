"use client";

import React, { useMemo } from "react";

import SummaryCard from "@/components/shared/SummaryCard";
import { savingsTotalsOf, usd } from "@/app/(dashboard)/cost-optimization/_components/costOptimizationUtils";
import { DailyData } from "@/components/UsagePage/types";
import { formatNumberWithCommas } from "@/utils/dataUtils";

const SavingsTiles = ({ results, isLoading }: { results: DailyData[]; isLoading: boolean }) => {
  const totals = useMemo(() => savingsTotalsOf(results), [results]);

  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
      <SummaryCard
        label="Compression savings"
        value={usd(totals.compression)}
        hint={`${formatNumberWithCommas(totals.savedTokens)} tokens compressed`}
        info="Tokens Headroom removed before the call, priced at the model's input rate. This is its own counterfactual, not a slice of a combined total."
      />
      <SummaryCard
        label="Prompt caching savings"
        value={usd(totals.gatewayAttributedCaching)}
        hint="LiteLLM injected"
        secondary={{ label: "Total", value: usd(totals.caching) }}
        info="What caching saved against paying the input rate for every token: the discount on tokens served from cache, less the premium providers charge to write a cache entry. The headline figure is the share LiteLLM earned by inserting the breakpoints itself, through configured injection points or auto prompt caching. The total beside it also counts requests that arrived with their own cache_control and providers that cache implicitly. Either can be negative on traffic that writes more cache than it reuses, which is why the headline is not always the smaller of the two. This view is not added to auto-router savings."
      />
      <SummaryCard
        label="Auto-router savings"
        value={usd(totals.autorouter)}
        hint={isLoading ? "Loading..." : "vs. the priciest model it could pick"}
        info="What this traffic would have cost had every request gone to the most expensive model the auto-router can route to, minus what it actually cost. That baseline is already cache-aware, so adding prompt-caching savings on top would count the same cache effect twice. Switching leaves the new model with a cold cache, so it pays to write the prompt again while the baseline is priced as already warm; a route that thrashes the cache can total below zero, and a genuine first turn, where neither side had anything cached, is undercounted."
      />
    </div>
  );
};

export default SavingsTiles;
