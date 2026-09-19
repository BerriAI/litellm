import { parseAsString, useQueryStates } from "nuqs";
import { useCallback } from "react";

import { useUrlTab } from "@/hooks/useUrlTab";

const ENDPOINT_TAB_KEY = "endpoint_tab";
const DETAIL_PARSERS = { endpoint: parseAsString, detailTab: parseAsString };
const DETAIL_OPTIONS = { history: "push", urlKeys: { detailTab: ENDPOINT_TAB_KEY } } as const;

const DETAIL_TABS = ["overview", "settings"] as const;
type PassThroughDetailTab = (typeof DETAIL_TABS)[number];
const OVERVIEW_ONLY: readonly PassThroughDetailTab[] = ["overview"];

export interface PassThroughDetailRouting {
  endpointKey: string | null;
  openEndpoint: (endpointKey: string) => void;
  closeEndpoint: () => void;
}

export function usePassThroughDetailRouting(): PassThroughDetailRouting {
  const [{ endpoint }, setParams] = useQueryStates(DETAIL_PARSERS, DETAIL_OPTIONS);

  const openEndpoint = useCallback(
    (endpointKey: string) => {
      void setParams({ endpoint: endpointKey, detailTab: null });
    },
    [setParams],
  );

  const closeEndpoint = useCallback(() => {
    void setParams({ endpoint: null, detailTab: null });
  }, [setParams]);

  return { endpointKey: endpoint, openEndpoint, closeEndpoint };
}

export function usePassThroughDetailTab(
  canViewSettings: boolean,
): [PassThroughDetailTab, (tab: PassThroughDetailTab) => void] {
  return useUrlTab(canViewSettings ? DETAIL_TABS : OVERVIEW_ONLY, "overview", ENDPOINT_TAB_KEY);
}

export function findPassThroughEndpoint<T extends { id?: string; path: string }>(
  endpoints: readonly T[],
  endpointKey: string,
): T | undefined {
  const byId = endpoints.find((endpoint) => endpoint.id === endpointKey);
  return byId ?? endpoints.find((endpoint) => endpoint.path === endpointKey);
}
