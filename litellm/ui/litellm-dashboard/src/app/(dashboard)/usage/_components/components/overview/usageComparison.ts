import type { DailyActivityMetadata } from "@/components/UsagePage/dailyActivityApi";
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

type TotalField = "total_spend" | "total_tokens" | "total_api_requests";

const PENDING: Change = { kind: "pending" };
const UNAVAILABLE: Change = { kind: "unavailable" };

const reported = (metadata: DailyActivityMetadata | null, field: TotalField): number | null => {
  const value: unknown = metadata?.[field];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
};

export const changeBetween = (current: number | null, previous: number | null): Change => {
  if (current === null || previous === null || !Number.isFinite(current) || !Number.isFinite(previous)) {
    return UNAVAILABLE;
  }
  if (previous === 0) return current === 0 ? { kind: "same" } : { kind: "new" };
  return { kind: "change", percent: ((current - previous) / previous) * 100 };
};

const perRequest = (metadata: DailyActivityMetadata | null): number | null => {
  const spend = reported(metadata, "total_spend");
  const requests = reported(metadata, "total_api_requests");
  return spend !== null && requests !== null && requests > 0 ? spend / requests : null;
};

const fieldChange = (
  current: DailyActivityMetadata | null,
  previous: DailyActivityMetadata | null,
  field: TotalField,
): Change => changeBetween(reported(current, field), reported(previous, field));

/** Request counts are only compared when the tile shows the aggregate's own count, not gateway tallies. */
export const statComparison = ({
  current,
  previous,
  previousRange,
  status,
  requestsComparable,
}: {
  current: DailyActivityMetadata | null;
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
  return {
    previous: previousRange,
    spend: fieldChange(current, previous, "total_spend"),
    tokens: fieldChange(current, previous, "total_tokens"),
    requests: requestsComparable ? fieldChange(current, previous, "total_api_requests") : null,
    avgCostPerRequest: changeBetween(perRequest(current), perRequest(previous)),
  };
};
