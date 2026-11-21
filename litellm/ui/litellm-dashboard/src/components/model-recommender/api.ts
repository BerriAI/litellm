import { createParser, parseAsArrayOf, parseAsBoolean, parseAsStringLiteral } from "nuqs";
import { fetchClient } from "@/lib/http/api";
import { getAuthHeaderName } from "@/lib/http/runtime";
import type { components } from "@/lib/http/schema";

export type RecommendationRequest = components["schemas"]["RecommendationRequest"];
export type RecommendationResponse = components["schemas"]["RecommendationResponse"];
export type Recommendation = components["schemas"]["Recommendation"];
export const FOCUS = ["general", "math", "coding", "agentic"] as const;
export const PRIORITIES = ["intelligence", "speed", "cost"] as const;
export const IMPORTANCE = ["Not considered", "Low importance", "Important", "Very important"] as const;
const priority = createParser({
  parse: (value) => (/^[0-3]$/.test(value) ? Number(value) : null),
  serialize: String,
}).withDefault(2);
const context = createParser({
  parse: (value) => (/^\d+$/.test(value) && Number.isSafeInteger(Number(value)) ? Number(value) : null),
  serialize: String,
});

export const recommenderParsers = {
  intelligence: priority,
  speed: priority,
  cost: priority,
  focus: parseAsArrayOf(parseAsStringLiteral(FOCUS)).withDefault(["general"]),
  reasoning: parseAsBoolean.withDefault(false),
  vision: parseAsBoolean.withDefault(false),
  function_calling: parseAsBoolean.withDefault(false),
  context,
};

export type RecommenderState = {
  intelligence: number;
  speed: number;
  cost: number;
  focus: (typeof FOCUS)[number][];
  reasoning: boolean;
  vision: boolean;
  function_calling: boolean;
  context: number | null;
  include_external?: boolean;
  providers?: string[];
};

export function buildRequest(state: RecommenderState): RecommendationRequest {
  return {
    priorities: { intelligence: state.intelligence, speed: state.speed, cost: state.cost },
    focus: state.focus,
    requirements: {
      reasoning: state.reasoning,
      vision: state.vision,
      function_calling: state.function_calling,
      min_context_tokens: state.context,
    },
    limit: 10,
    include_external: state.include_external ?? false,
    providers: state.providers ?? [],
  };
}

export async function requestRecommendations(
  body: RecommendationRequest,
  accessToken: string,
  signal?: AbortSignal,
): Promise<RecommendationResponse> {
  const result = await fetchClient.POST("/model/recommend", {
    body,
    headers: { [getAuthHeaderName()]: `Bearer ${accessToken}` },
    signal,
  });
  if (!result.data) throw new Error("The gateway returned no recommendations. Please try again.");
  return result.data;
}

export function metricValue(kind: (typeof PRIORITIES)[number], raw: number | null): string {
  if (raw === null) return "Unknown";
  if (kind === "cost") return `$${raw.toFixed(2)} / 1M`;
  if (kind === "speed") return `${raw.toFixed(1)}s`;
  return `${Math.round(raw * 100)}% benchmark score`;
}

export function compactNumber(value: number): string {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}
