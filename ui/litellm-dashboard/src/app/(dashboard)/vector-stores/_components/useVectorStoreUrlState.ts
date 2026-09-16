import { parseAsBoolean, parseAsString, useQueryStates } from "nuqs";

import { useUrlTab } from "@/hooks/useUrlTab";

const DETAIL_TAB_KEY = "detail_tab";
const DETAIL_TABS = ["details", "test"] as const;

const VECTOR_STORE_DETAIL_PARSERS = {
  vector_store: parseAsString.withOptions({ history: "push" }),
  edit: parseAsBoolean.withDefault(false),
  [DETAIL_TAB_KEY]: parseAsString,
};

export const useVectorStoreDetailUrlState = () => useQueryStates(VECTOR_STORE_DETAIL_PARSERS);

export const useVectorStoreDetailTab = () => useUrlTab(DETAIL_TABS, "details", DETAIL_TAB_KEY);
