import { useUrlTab } from "@/hooks/useUrlTab";

export const GARDEN_TAB_KEY = "garden_tab";

const GARDEN_TABS_WITH_EVAL = ["overview", "eval"] as const;
type GardenTab = (typeof GARDEN_TABS_WITH_EVAL)[number];
const GARDEN_TABS_WITHOUT_EVAL: readonly GardenTab[] = ["overview"];

export const useGardenDetailTab = (hasEval: boolean) =>
  useUrlTab(hasEval ? GARDEN_TABS_WITH_EVAL : GARDEN_TABS_WITHOUT_EVAL, "overview", GARDEN_TAB_KEY);
