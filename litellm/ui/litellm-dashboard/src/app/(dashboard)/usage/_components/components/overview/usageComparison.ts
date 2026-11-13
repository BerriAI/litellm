import { EMPTY_DAILY_ACTIVITY_METADATA, type DailyActivityMetadata } from "@/components/UsagePage/dailyActivityApi";
import type { DayRange } from "./usageTrendData";

export type Change =
  | { readonly kind: "pending" }
  | { readonly kind: "unavailable" }
  | { readonly kind: "new" }
  | { readonly kind: "same" }
  | { readonly kind: "change"; readonly percent: number };

export type ComparisonStatus = "loading" | "failed" | "ready";

export interface StatComparison {
  readonly previous: DayRange;
  readonly spend: Change;
  readonly tokens: Change;
  readonly requests: Change | null;
  readonly avgCostPerRequest: Change;
}

const PENDING: Change = { kind: "pending" };
const UNAVAILABLE: Change = { kind: "unavailable" };

/** Percent change from `previous` to `current`. A zero baseline has no ratio, so it reads New or unchanged. */
export const changeBetween = (current: number, previous: number): Change => {
  if (!Number.isFinite(current) || !Number.isFinite(previous)) return UNAVAILABLE;
  if (previous === 0) return current === 0 ? { kind: "same" } : { kind: "new" };
  return { kind: "change", percent: ((current - previous) / previous) * 100 };
};

const perRequest = (spend: number, requests: number): number | null => (requests > 0 ? spend / requests : null);

const avgCostChange = (current: Required<DailyActivityMetadata>, previous: Required<DailyActivityMetadata>): Change => {
  const now = perRequest(current.total_spend, current.total_api_requests);
  const before = perRequest(previous.total_spend, previous.total_api_requests);
  if (now === null || before === null) return UNAVAILABLE;
  return changeBetween(now, before);
};

/**
 * Headline changes against the preceding period, every figure from the same aggregate on both sides.
 * Request counts are only compared when the tile shows that aggregate's count, not gateway tallies.
 */
export const statComparison = ({
  current,
  previous,
  previousRange,
  status,
  requestsComparable,
}: {
  current: DailyActivityMetadata;
  previous: DailyActivityMetadata | null;
  previousRange: DayRange;
  status: ComparisonStatus;
  requestsComparable: boolean;
}): StatComparison => {
  if (status !== "ready" || previous === null) {
    const placeholder = status === "loading" ? PENDING : UNAVAILABLE;
    return {
      previous: previousRange,
      spend: placeholder,
      tokens: placeholder,
      requests: requestsComparable ? placeholder : null,
      avgCostPerRequest: placeholder,
    };
  }
  const now = { ...EMPTY_DAILY_ACTIVITY_METADATA, ...current } as Required<DailyActivityMetadata>;
  const before = { ...EMPTY_DAILY_ACTIVITY_METADATA, ...previous } as Required<DailyActivityMetadata>;
  return {
    previous: previousRange,
    spend: changeBetween(now.total_spend, before.total_spend),
    tokens: changeBetween(now.total_tokens, before.total_tokens),
    requests: requestsComparable ? changeBetween(now.total_api_requests, before.total_api_requests) : null,
    avgCostPerRequest: avgCostChange(now, before),
  };
};
