import { usePathname } from "next/navigation";
import { parseAsString, useQueryState, useQueryStates } from "nuqs";
import { useCallback, useEffect } from "react";
import { recordUiEvent } from "@/lib/telemetry/uiEvents";
import { routeSegmentForPathname } from "@/utils/uiHref";

export function useUrlTab<T extends string>(values: readonly T[], fallback: T, key = "tab"): [T, (tab: T) => void] {
  const [urlTab, setUrlTab] = useQueryState(key, parseAsString.withDefault(fallback));
  const page = routeSegmentForPathname(usePathname() ?? "") || "home";
  const tab = values.find((value) => value === urlTab) ?? fallback;
  useEffect(() => {
    if (urlTab !== tab) void setUrlTab(null);
  }, [urlTab, tab, setUrlTab]);
  const setTab = useCallback(
    (next: T) => {
      void recordUiEvent({ page, action: "click", target: `${key}=${next}` });
      void setUrlTab(next);
    },
    [page, key, setUrlTab],
  );
  return [tab, setTab];
}

export function useResetDetailState(tabParam = "detail_tab"): () => void {
  const parsers = {
    [tabParam]: parseAsString,
    key_tab: parseAsString,
    activity_from: parseAsString,
    activity_to: parseAsString,
  };
  const [, setParams] = useQueryStates(parsers);
  return useCallback(() => {
    void setParams(null);
  }, [setParams]);
}
