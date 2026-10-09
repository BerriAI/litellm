"use client";

import { useQuery } from "@tanstack/react-query";
import { uiHref } from "@/utils/uiHref";
import type { BuilderInsightsData } from "./builderInsightsData";

const fetchBuilderInsights = async (): Promise<BuilderInsightsData> => {
  // eslint-disable-next-line no-restricted-syntax -- static public demo asset is not served by the API client
  const response = await fetch(uiHref("builder-insights-sample.json"));
  if (!response.ok) throw new Error("Unable to load Builder Insights");
  return (await response.json()) as BuilderInsightsData;
};

export const useBuilderInsights = () =>
  useQuery({
    queryKey: ["builder-insights-sample"],
    queryFn: fetchBuilderInsights,
    staleTime: 5 * 60 * 1000,
  });
