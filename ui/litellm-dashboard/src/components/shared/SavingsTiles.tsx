"use client";

import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";

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
  const { t } = useTranslation();
  const totals = useSavingsTotals(results);

  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
      <SummaryCard
        label={t("shared.savingsTiles.total.label")}
        value={usd(totals.total)}
        hint={isLoading ? t("shared.savingsTiles.total.loading") : t("shared.savingsTiles.total.hint")}
        info={t("shared.savingsTiles.total.info")}
      />
      <SummaryCard
        label={t("shared.savingsTiles.compression.label")}
        value={usd(totals.compression)}
        hint={t("shared.savingsTiles.compression.hint", { tokens: formatNumberWithCommas(totals.savedTokens) })}
        info={t("shared.savingsTiles.compression.info")}
      />
      <SummaryCard
        label={t("shared.savingsTiles.caching.label")}
        value={usd(totals.gatewayAttributedCaching)}
        hint={t("shared.savingsTiles.caching.hint")}
        secondary={{ label: t("shared.savingsTiles.caching.total"), value: usd(totals.caching) }}
        info={t("shared.savingsTiles.caching.info")}
      />
      <SummaryCard
        label={t("shared.savingsTiles.autorouter.label")}
        value={usd(totals.autorouter)}
        hint={t("shared.savingsTiles.autorouter.hint")}
        info={t("shared.savingsTiles.autorouter.info")}
      />
    </div>
  );
};

export default SavingsTiles;
