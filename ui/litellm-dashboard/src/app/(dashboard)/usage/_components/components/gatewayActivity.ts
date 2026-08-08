/**
 * Gateway request counts (SGR) from `/gateway/daily/activity`.
 *
 * Recorded by the proxy's request-metrics middleware rather than derived from
 * spend logs, so it counts what the gateway actually answered. Deployment-wide
 * with no per-key or per-user dimension, which is why it is admin-only and why
 * the per-key and per-model breakdowns on the usage page still come from the
 * spend tables.
 */

export const GATEWAY_TOP_ROUTES = 15;

export type SGRLimitState = "under" | "soft_exceeded" | "hard_exceeded";

/**
 * Standing against the configured SGR allowance, for the limit's own window.
 *
 * `successful_requests` here is not the total above: that one covers the
 * selected date range, this one covers the window the limit is counted over.
 */
export interface SGRLimit {
  limit: number;
  soft_limit: number;
  window: "month" | "year";
  window_start: string;
  successful_requests: number;
  state: SGRLimitState;
}

export interface GatewayActivity {
  total_successful_requests: number;
  total_failed_requests: number;
  by_date: { date: string; successful_requests: number; failed_requests: number }[];
  by_route: { category: string; route: string; successful_requests: number; failed_requests: number }[];
  /** Absent on a deployment with no allowance configured, and on older proxies. */
  sgr_limit?: SGRLimit | null;
}

export interface SGRLimitBanner {
  severity: "warning" | "error";
  headline: string;
  detail: string;
}

/**
 * The banner to show for the SGR allowance, or null when there is nothing to say.
 *
 * Nothing to say covers three cases: no allowance configured, an allowance the
 * deployment is still under, and a proxy too old to report one.
 */
export const sgrLimitBanner = (activity: GatewayActivity | null): SGRLimitBanner | null => {
  const sgrLimit = activity?.sgr_limit;
  if (sgrLimit == null || sgrLimit.state === "under") return null;
  const used = `${sgrLimit.successful_requests.toLocaleString()} of ${sgrLimit.limit.toLocaleString()}`;
  const percent = Math.floor((100 * sgrLimit.soft_limit) / sgrLimit.limit);
  return sgrLimit.state === "hard_exceeded"
    ? {
        severity: "error",
        headline: `Gateway request limit reached: ${used} successful requests this ${sgrLimit.window}`,
        detail: `Counted since ${sgrLimit.window_start} (UTC). Requests are still being served, this is an alert only.`,
      }
    : {
        severity: "warning",
        headline: `Approaching the gateway request limit: ${used} successful requests this ${sgrLimit.window}`,
        detail: `Past ${percent}% of the limit, counted since ${sgrLimit.window_start} (UTC).`,
      };
};

/** A fetched result carrying the range key it was fetched for. */
export interface FetchedForRange<T> {
  rangeKey: string;
  value: T;
}

export type FetchedGatewayActivity = FetchedForRange<GatewayActivity>;

/** Extends Record so it satisfies the chart component's row constraint. */
export interface GatewayRouteBar extends Record<string, unknown> {
  route: string;
  successful_requests: number;
  failed_requests: number;
}

/**
 * Identifies what a result was fetched for: the date range, plus any other
 * input that changes the answer. The usage aggregate is scoped to a user, so
 * two results covering the same dates still describe different numbers.
 */
export const fetchedRangeKey = (
  startTime: Date | null | undefined,
  endTime: Date | null | undefined,
  scope: string | null | undefined = null,
): string => `${startTime?.toISOString() ?? ""}|${endTime?.toISOString() ?? ""}|${scope ?? ""}`;

/**
 * The value safe to render right now, or null to fall back.
 *
 * Clearing the state inside the fetch effect is one render too late: the render
 * that follows a date change still holds the previous range's value and can
 * paint before effects run. Comparing the stamp during render is what makes a
 * superseded range unrepresentable rather than merely brief.
 */
export const selectForRange = <T>(fetched: FetchedForRange<T> | null, currentRangeKey: string): T | null =>
  fetched != null && fetched.rangeKey === currentRangeKey ? fetched.value : null;

/**
 * As `selectForRange`, and additionally withholds the counts from a non-admin:
 * they are deployment-wide, so they are not a non-admin's to read.
 */
export const selectGatewayActivity = (
  isAdmin: boolean,
  fetched: FetchedGatewayActivity | null,
  currentRangeKey: string,
): GatewayActivity | null => (isAdmin ? selectForRange(fetched, currentRangeKey) : null);

/**
 * Bars for the endpoint breakdown chart, capped so a deployment exercising many
 * endpoints does not render an unreadable axis. `by_route` arrives sorted by
 * successful_requests descending, so the cap keeps the busiest endpoints.
 */
export const topGatewayRoutes = (
  activity: GatewayActivity | null,
  limit: number = GATEWAY_TOP_ROUTES,
): GatewayRouteBar[] =>
  (activity?.by_route ?? []).slice(0, limit).map((entry) => ({
    // The llm routes are already fully qualified; mcp and a2a routes are not, so
    // their category prefix is what keeps "/mcp" apart from "/a2a".
    route: entry.category === "llm" ? entry.route : `${entry.category}${entry.route}`,
    successful_requests: entry.successful_requests,
    failed_requests: entry.failed_requests,
  }));
