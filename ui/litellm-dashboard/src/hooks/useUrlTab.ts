import { parseAsString, useQueryState } from "nuqs";
import { useCallback, useEffect } from "react";

export function useUrlTab<T extends string>(values: readonly T[], fallback: T, key = "tab"): [T, (tab: T) => void] {
  const [urlTab, setUrlTab] = useQueryState(key, parseAsString.withDefault(fallback));
  const tab = values.find((value) => value === urlTab) ?? fallback;
  useEffect(() => {
    if (urlTab !== tab) void setUrlTab(null);
  }, [urlTab, tab, setUrlTab]);
  const setTab = useCallback((next: T) => void setUrlTab(next), [setUrlTab]);
  return [tab, setTab];
}
