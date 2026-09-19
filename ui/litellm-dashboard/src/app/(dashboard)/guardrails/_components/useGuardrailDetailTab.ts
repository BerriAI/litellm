import { useUrlTab } from "@/hooks/useUrlTab";

export const GUARDRAIL_DETAIL_TAB_KEY = "detail_tab";

const ADMIN_DETAIL_TABS = ["overview", "settings"] as const;
type GuardrailDetailTab = (typeof ADMIN_DETAIL_TABS)[number];
const VIEWER_DETAIL_TABS: readonly GuardrailDetailTab[] = ["overview"];

export const useGuardrailDetailTab = (isAdmin: boolean) =>
  useUrlTab(isAdmin ? ADMIN_DETAIL_TABS : VIEWER_DETAIL_TABS, "overview", GUARDRAIL_DETAIL_TAB_KEY);
