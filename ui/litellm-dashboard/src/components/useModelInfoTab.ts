import { useCallback } from "react";

import { useUrlTab } from "@/hooks/useUrlTab";

const MODEL_INFO_TABS = ["overview", "raw"] as const;
type ModelInfoTab = (typeof MODEL_INFO_TABS)[number];
const DEFAULT_MODEL_INFO_TAB: ModelInfoTab = "overview";

export interface ModelInfoTabState {
  tab: ModelInfoTab;
  setTab: (tab: ModelInfoTab) => void;
  close: () => void;
}

export function useModelInfoTab(onClose: () => void): ModelInfoTabState {
  const [tab, setTab] = useUrlTab(MODEL_INFO_TABS, DEFAULT_MODEL_INFO_TAB, "model_tab");

  const close = useCallback(() => {
    setTab(DEFAULT_MODEL_INFO_TAB);
    onClose();
  }, [setTab, onClose]);

  return { tab, setTab, close };
}
