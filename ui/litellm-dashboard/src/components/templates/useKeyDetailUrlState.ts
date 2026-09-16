import { parseAsString, useQueryStates } from "nuqs";
import { useCallback } from "react";

import type { SavingsAccumulation } from "@/app/(dashboard)/cost-optimization/_components/costOptimizationUtils";
import { useUrlTab } from "@/hooks/useUrlTab";

const ALL_KEY_TABS = ["overview", "savings", "auto-router-usage", "settings"] as const;
export type KeyDetailTab = (typeof ALL_KEY_TABS)[number];
const TABS_WITHOUT_AUTO_ROUTER: readonly KeyDetailTab[] = ALL_KEY_TABS.filter((tab) => tab !== "auto-router-usage");

const SAVINGS_VIEWS: readonly SavingsAccumulation[] = ["cumulative", "per-interval"];

const KEY_DETAIL_URL_KEYS = {
  key_tab: parseAsString,
  key_savings_view: parseAsString,
};

export interface KeyDetailUrlState {
  tab: KeyDetailTab;
  setTab: (tab: KeyDetailTab) => void;
  clearKeyDetailUrlState: () => void;
}

export function useKeyDetailUrlState(showAutoRouterUsage: boolean): KeyDetailUrlState {
  const [tab, setTab] = useUrlTab<KeyDetailTab>(
    showAutoRouterUsage ? ALL_KEY_TABS : TABS_WITHOUT_AUTO_ROUTER,
    "overview",
    "key_tab",
  );
  const [, setDetailParams] = useQueryStates(KEY_DETAIL_URL_KEYS);
  const clearKeyDetailUrlState = useCallback(() => void setDetailParams(null), [setDetailParams]);
  return { tab, setTab, clearKeyDetailUrlState };
}

export function useKeySavingsView(): [SavingsAccumulation, (view: SavingsAccumulation) => void] {
  return useUrlTab(SAVINGS_VIEWS, "cumulative", "key_savings_view");
}
