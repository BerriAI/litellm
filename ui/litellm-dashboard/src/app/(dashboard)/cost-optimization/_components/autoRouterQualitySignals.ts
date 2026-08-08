import type { components } from "@/lib/http/schema";

import { ALL_ROUTERS } from "./autoRouterBenchmarks";

export type AutoRouterQualitySignalsResponse = components["schemas"]["AutoRouterQualitySignalsResponse"];
export type AutoRouterQualitySignals = components["schemas"]["AutoRouterQualitySignals"];
export type AutoRouterQualityCohort = components["schemas"]["AutoRouterQualityCohort"];

export const BASELINE_UNAVAILABLE_COPY: Record<string, string> = {
  no_session_ids: "Non-router traffic isn't sending session IDs, so it can't be grouped into sessions to compare",
  insufficient_sessions: "Not enough comparable non-router traffic in this window to compare",
};

export const signalsFor = (
  data: AutoRouterQualitySignalsResponse,
  selectedRouterName: string | null,
): AutoRouterQualitySignals => {
  if (selectedRouterName === null || selectedRouterName === ALL_ROUTERS) return data.totals;
  return data.groups.find((group) => group.router_name === selectedRouterName) ?? data.totals;
};

export const ratePctLabel = (value: number | null | undefined): string =>
  value === null || value === undefined ? "—" : `${value.toFixed(1)}%`;

/** Positive when the router escalates more often than the operator's own direct traffic. */
export const deltaVsBaseline = (
  routed: number | null | undefined,
  baseline: number | null | undefined,
): number | null => {
  if (routed === null || routed === undefined) return null;
  if (baseline === null || baseline === undefined) return null;
  return routed - baseline;
};
