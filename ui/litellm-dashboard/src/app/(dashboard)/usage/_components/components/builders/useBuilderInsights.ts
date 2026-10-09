"use client";

import { useQuery } from "@tanstack/react-query";
import type { BuilderInsightsData } from "./builderInsightsData";

const fetchBuilderInsights = async (): Promise<BuilderInsightsData> => {
  // eslint-disable-next-line no-restricted-syntax -- static public demo asset is not served by the API client
  const response = await fetch("/builder-insights-demo.json");
  if (!response.ok) throw new Error("Unable to load Builder Insights");
  return (await response.json()) as BuilderInsightsData;
};

export const useBuilderInsights = () =>
  useQuery({
    queryKey: ["builder-insights-demo"],
    queryFn: fetchBuilderInsights,
    staleTime: 5 * 60 * 1000,
  });
