import { describe, expect, it } from "vitest";
import {
  GATEWAY_TOP_ROUTES,
  fetchedRangeKey,
  selectForRange,
  selectGatewayActivity,
  sgrLimitBanner,
  topGatewayRoutes,
  type GatewayActivity,
  type SGRLimit,
} from "./gatewayActivity";

const activity = (total: number): GatewayActivity => ({
  total_successful_requests: total,
  total_failed_requests: 0,
  by_date: [{ date: "2025-01-01", successful_requests: total, failed_requests: 0 }],
  by_route: [{ category: "llm", route: "/chat/completions", successful_requests: total, failed_requests: 0 }],
});

const JANUARY = fetchedRangeKey(new Date("2025-01-01T00:00:00Z"), new Date("2025-01-31T00:00:00Z"));
const FEBRUARY = fetchedRangeKey(new Date("2025-02-01T00:00:00Z"), new Date("2025-02-28T00:00:00Z"));

describe("fetchedRangeKey", () => {
  it("distinguishes ranges that differ only in their end", () => {
    const start = new Date("2025-01-01T00:00:00Z");
    expect(fetchedRangeKey(start, new Date("2025-01-31T00:00:00Z"))).not.toEqual(
      fetchedRangeKey(start, new Date("2025-02-28T00:00:00Z")),
    );
  });

  it("distinguishes the same range fetched for two different users", () => {
    const start = new Date("2025-01-01T00:00:00Z");
    const end = new Date("2025-01-31T00:00:00Z");
    expect(fetchedRangeKey(start, end, "user-a")).not.toEqual(fetchedRangeKey(start, end, "user-b"));
  });

  it("is stable for equal instants held in different Date objects", () => {
    expect(fetchedRangeKey(new Date("2025-01-01T00:00:00Z"), new Date("2025-01-31T00:00:00Z"))).toEqual(JANUARY);
  });

  it("tolerates a range that has not been picked yet", () => {
    expect(fetchedRangeKey(null, null)).toEqual("||");
  });
});

describe("selectForRange", () => {
  it("returns the value when it was fetched for the selected range", () => {
    expect(selectForRange({ rangeKey: JANUARY, value: 7 }, JANUARY)).toEqual(7);
  });

  it("withholds the previous range's value while a new range is in flight", () => {
    expect(selectForRange({ rangeKey: JANUARY, value: 7 }, FEBRUARY)).toBeNull();
  });

  it("returns null before anything has been fetched", () => {
    expect(selectForRange(null, JANUARY)).toBeNull();
  });
});

describe("selectGatewayActivity", () => {
  it("returns the counts when an admin's result matches the selected range", () => {
    expect(selectGatewayActivity(true, { rangeKey: JANUARY, value: activity(7) }, JANUARY)).toEqual(activity(7));
  });

  it("withholds the previous range's counts while a new range is in flight", () => {
    expect(selectGatewayActivity(true, { rangeKey: JANUARY, value: activity(7) }, FEBRUARY)).toBeNull();
  });

  it("withholds deployment-wide counts from a non-admin", () => {
    expect(selectGatewayActivity(false, { rangeKey: JANUARY, value: activity(7) }, JANUARY)).toBeNull();
  });

  it("returns null before anything has been fetched", () => {
    expect(selectGatewayActivity(true, null, JANUARY)).toBeNull();
  });
});

describe("topGatewayRoutes", () => {
  it("leaves an llm route unprefixed and prefixes the others so they stay distinguishable", () => {
    const bars = topGatewayRoutes({
      ...activity(0),
      by_route: [
        { category: "llm", route: "/chat/completions", successful_requests: 3, failed_requests: 1 },
        { category: "mcp", route: "/tools/call", successful_requests: 2, failed_requests: 0 },
        { category: "a2a", route: "/tools/call", successful_requests: 1, failed_requests: 0 },
      ],
    });
    expect(bars.map((bar) => bar.route)).toEqual(["/chat/completions", "mcp/tools/call", "a2a/tools/call"]);
    expect(bars[0]).toEqual({ route: "/chat/completions", successful_requests: 3, failed_requests: 1 });
  });

  it("caps the bars at the top N so a wide deployment stays readable", () => {
    const many = Array.from({ length: GATEWAY_TOP_ROUTES + 5 }, (_, i) => ({
      category: "llm",
      route: `/route-${i}`,
      successful_requests: 100 - i,
      failed_requests: 0,
    }));
    const bars = topGatewayRoutes({ ...activity(0), by_route: many });
    expect(bars).toHaveLength(GATEWAY_TOP_ROUTES);
    // The cap keeps the busiest endpoints, which is only true because it slices
    // the server's descending order rather than re-sorting.
    expect(bars[0].route).toEqual("/route-0");
    expect(bars[GATEWAY_TOP_ROUTES - 1].route).toEqual(`/route-${GATEWAY_TOP_ROUTES - 1}`);
  });

  it("renders no bars when there is nothing to show", () => {
    expect(topGatewayRoutes(null)).toEqual([]);
  });
});

describe("sgrLimitBanner", () => {
  const withLimit = (sgrLimit: SGRLimit | null): GatewayActivity => ({ ...activity(0), sgr_limit: sgrLimit });

  const limit: SGRLimit = {
    limit: 1_000_000,
    soft_limit: 800_000,
    window: "month",
    window_start: "2026-08-01",
    successful_requests: 900_000,
    state: "soft_exceeded",
  };

  it("says nothing when the deployment has no allowance, or is under it", () => {
    expect(sgrLimitBanner(null)).toBeNull();
    expect(sgrLimitBanner(activity(0))).toBeNull();
    expect(sgrLimitBanner(withLimit(null))).toBeNull();
    expect(sgrLimitBanner(withLimit({ ...limit, successful_requests: 10, state: "under" }))).toBeNull();
  });

  it("warns once past the soft threshold, naming the count, the limit and the window", () => {
    const banner = sgrLimitBanner(withLimit(limit));
    expect(banner?.severity).toEqual("warning");
    expect(banner?.headline).toContain("900,000 of 1,000,000");
    expect(banner?.headline).toContain("this month");
    expect(banner?.detail).toContain("80%");
    expect(banner?.detail).toContain("2026-08-01");
  });

  it("escalates to an error once the limit itself is reached, and says traffic is unaffected", () => {
    const banner = sgrLimitBanner(withLimit({ ...limit, successful_requests: 1_100_000, state: "hard_exceeded" }));
    expect(banner?.severity).toEqual("error");
    expect(banner?.headline).toContain("1,100,000 of 1,000,000");
    expect(banner?.detail).toContain("still being served");
  });
});
