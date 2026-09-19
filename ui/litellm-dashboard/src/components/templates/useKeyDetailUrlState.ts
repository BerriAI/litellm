import { parseAsString, useQueryStates } from "nuqs";
import { useCallback, useEffect, useRef } from "react";

import type { SavingsAccumulation } from "@/app/(dashboard)/cost-optimization/_components/costOptimizationUtils";
import { useUrlTab } from "@/hooks/useUrlTab";

const KEY_TAB = "key_tab";
const KEY_SAVINGS_VIEW = "key_savings_view";

const ALL_KEY_TABS = ["overview", "savings", "auto-router-usage", "settings"] as const;
export type KeyDetailTab = (typeof ALL_KEY_TABS)[number];
const TABS_WITHOUT_AUTO_ROUTER: readonly KeyDetailTab[] = ALL_KEY_TABS.filter((tab) => tab !== "auto-router-usage");

const SAVINGS_VIEWS: readonly SavingsAccumulation[] = ["cumulative", "per-interval"];

const KEY_DETAIL_URL_PARSERS = {
  [KEY_TAB]: parseAsString,
  [KEY_SAVINGS_VIEW]: parseAsString,
};

export interface KeyDetailUrlState {
  tab: KeyDetailTab;
  setTab: (tab: KeyDetailTab) => void;
  clearKeyDetailUrlState: () => void;
}

function useClearAfterUnmount(hasState: boolean, clear: () => void): void {
  const latest = useRef({ hasState, clear });
  const mounted = useRef(false);

  useEffect(() => {
    latest.current = { hasState, clear };
  }, [hasState, clear]);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      setTimeout(() => {
        if (!mounted.current && latest.current.hasState) {
          latest.current.clear();
        }
      }, 0);
    };
  }, []);
}

export function useKeyDetailUrlState(showAutoRouterUsage: boolean): KeyDetailUrlState {
  const [tab, setTab] = useUrlTab<KeyDetailTab>(
    showAutoRouterUsage ? ALL_KEY_TABS : TABS_WITHOUT_AUTO_ROUTER,
    "overview",
    KEY_TAB,
  );
  const [detailParams, setDetailParams] = useQueryStates(KEY_DETAIL_URL_PARSERS);
  const clearKeyDetailUrlState = useCallback(() => void setDetailParams(null), [setDetailParams]);
  const hasDetailState = detailParams[KEY_TAB] !== null || detailParams[KEY_SAVINGS_VIEW] !== null;
  useClearAfterUnmount(hasDetailState, clearKeyDetailUrlState);
  return { tab, setTab, clearKeyDetailUrlState };
}

export function useKeySavingsView(): [SavingsAccumulation, (view: SavingsAccumulation) => void] {
  return useUrlTab(SAVINGS_VIEWS, "cumulative", KEY_SAVINGS_VIEW);
}
