import { describe, expect, it } from "vitest";
import {
  GATEWAY_TOP_ROUTES,
  failedRequestBreakdown,
  fetchedRangeKey,
  selectForRange,
  selectGatewayActivity,
  statusCodeLabel,
  topGatewayRoutes,
  type GatewayActivity,
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

describe("failedRequestBreakdown", () => {
  it("splits client and server errors, labels rows in received order, and finds unrecorded failures", () => {
    const expectedBreakdown = {
      clientErrors: 3,
      serverErrors: 3,
      rows: [
        { status_code: 500, label: "Internal Server Error", failed_requests: 2 },
        { status_code: 429, label: "Too Many Requests", failed_requests: 3 },
        { status_code: 503, label: "Service Unavailable", failed_requests: 1 },
      ],
      notRecorded: 4,
    };

    expect(
      failedRequestBreakdown({
        ...activity(0),
        total_failed_requests: 10,
        by_status_code: [
          { status_code: 500, failed_requests: 2 },
          { status_code: 429, failed_requests: 3 },
          { status_code: 503, failed_requests: 1 },
        ],
      }),
    ).toEqual(expectedBreakdown);
  });

  it("treats a missing status-code breakdown as no recorded failures", () => {
    const result = failedRequestBreakdown({ ...activity(0), total_failed_requests: 3, by_status_code: undefined });
    const expectedBreakdown = { clientErrors: 0, serverErrors: 0, rows: [], notRecorded: 3 };
    expect(result).toEqual(expectedBreakdown);
  });

  it("does not let recorded rows exceed the failed-request total", () => {
    const result = failedRequestBreakdown({
      ...activity(0),
      total_failed_requests: 2,
      by_status_code: [{ status_code: 0, failed_requests: 3 }],
    });
    expect(result.notRecorded).toBe(0);
  });
});

describe("statusCodeLabel", () => {
  it.each([
    [0, "No response"],
    [400, "Bad Request"],
    [401, "Unauthorized"],
    [403, "Forbidden"],
    [404, "Not Found"],
    [408, "Request Timeout"],
    [413, "Payload Too Large"],
    [422, "Unprocessable Entity"],
    [429, "Too Many Requests"],
    [500, "Internal Server Error"],
    [502, "Bad Gateway"],
    [503, "Service Unavailable"],
    [504, "Gateway Timeout"],
    [418, "HTTP 418"],
  ] as const)("labels status %s", (code, label) => {
    expect(statusCodeLabel(code)).toBe(label);
  });
});
