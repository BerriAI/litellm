"use client";

import { useQueries } from "@tanstack/react-query";

import { apiClient } from "@/components/networking";
import type { components } from "@/lib/http/schema";

import { PUBLIC_MODEL_HUB_PATH } from "./usePublicModelHubList";

type FacetResponse = components["schemas"]["FacetListResponse"];

export const MODEL_HUB_FACETS = ["providers", "modes", "features"] as const;

export type ModelHubFacet = (typeof MODEL_HUB_FACETS)[number];

/** The route caps a page at 100, which is far above the distinct providers, modes or features any proxy publishes. */
const FACET_PAGE_SIZE = 100;

export interface PublicModelHubFacets {
  providers: string[];
  modes: string[];
  features: string[];
}

const fetchFacet = (facet: ModelHubFacet, signal: AbortSignal): Promise<FacetResponse> =>
  apiClient.get<FacetResponse>(`${PUBLIC_MODEL_HUB_PATH}/${facet}`, {
    query: { page_size: FACET_PAGE_SIZE },
    signal,
  });

/**
 * The values each filter dropdown offers, read from the route rather than derived from a
 * page of rows, which can only ever show the values that page happens to contain.
 */
export const usePublicModelHubFacets = (enabled: boolean): PublicModelHubFacets => {
  const results = useQueries({
    queries: MODEL_HUB_FACETS.map((facet) => ({
      queryKey: ["publicModelHub", "facet", facet],
      queryFn: ({ signal }: { signal: AbortSignal }) => fetchFacet(facet, signal),
      enabled,
      staleTime: Infinity,
    })),
  });

  const [providers, modes, features] = results.map((result) => result.data?.data ?? []);
  return { providers, modes, features };
};
