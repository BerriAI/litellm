import { describe, expect, it } from "vitest";
import { SEQUENTIAL_COLOR_RAMP } from "@/components/shared/charts";
import type { ActiveRequest } from "./activeRequestsApi";
import { chartHeight, countBy, countByAge, magnitudeFills, TOP_N, UNATTRIBUTED } from "./activeRequestGrouping";

const NOW_MS = 1_700_000_000_000;
const NOW_SECONDS = NOW_MS / 1000;

const request = (overrides: Partial<ActiveRequest>): ActiveRequest =>
  ({
    request_id: "req",
    started_at: NOW_SECONDS,
    streaming: false,
    ...overrides,
  }) as ActiveRequest;

describe("countBy", () => {
  it("counts requests per value and orders them by count", () => {
    const items = [
      request({ model: "a" }),
      request({ model: "b" }),
      request({ model: "b" }),
      request({ model: "c" }),
      request({ model: "c" }),
      request({ model: "c" }),
    ];

    expect(countBy(items, "model")).toEqual([
      { label: "c", requests: 3 },
      { label: "b", requests: 2 },
      { label: "a", requests: 1 },
    ]);
  });

  it("breaks count ties alphabetically so the order is stable across polls", () => {
    const items = [request({ model: "zeta" }), request({ model: "alpha" })];

    expect(countBy(items, "model").map((entry) => entry.label)).toEqual(["alpha", "zeta"]);
  });

  it("buckets missing and blank values as unattributed instead of dropping them", () => {
    const items = [request({ end_user_id: null }), request({ end_user_id: "   " }), request({ end_user_id: "u1" })];

    expect(countBy(items, "end_user_id")).toEqual([
      { label: UNATTRIBUTED, requests: 2 },
      { label: "u1", requests: 1 },
    ]);
  });

  it("folds everything past the top N into a single Other entry that keeps the total intact", () => {
    const items = Array.from({ length: TOP_N + 3 }, (_, index) =>
      Array.from({ length: TOP_N + 3 - index }, () => request({ model: `m${index}` })),
    ).flat();

    const grouped = countBy(items, "model");
    const total = grouped.reduce((sum, entry) => sum + entry.requests, 0);

    expect(grouped).toHaveLength(TOP_N + 1);
    expect(grouped[TOP_N].label).toBe("Other (3)");
    expect(total).toBe(items.length);
  });

  it("returns nothing for an empty page", () => {
    expect(countBy([], "model")).toEqual([]);
  });
});

describe("countByAge", () => {
  it("places each request in exactly one bin, upper bound exclusive", () => {
    const items = [
      request({ started_at: NOW_SECONDS - 1 }),
      request({ started_at: NOW_SECONDS - 10 }),
      request({ started_at: NOW_SECONDS - 30 }),
      request({ started_at: NOW_SECONDS - 60 }),
      request({ started_at: NOW_SECONDS - 3600 }),
    ];

    expect(countByAge(items, NOW_MS)).toEqual([
      { label: "< 10s", requests: 1 },
      { label: "10-30s", requests: 1 },
      { label: "30-60s", requests: 1 },
      { label: "1-5m", requests: 1 },
      { label: "> 5m", requests: 1 },
    ]);
  });

  it("keeps every bin present so the chart axis does not shift between polls", () => {
    expect(countByAge([], NOW_MS).map((entry) => entry.label)).toEqual(["< 10s", "10-30s", "30-60s", "1-5m", "> 5m"]);
  });

  it("counts a clock skewed start time as the youngest bin rather than losing it", () => {
    expect(countByAge([request({ started_at: NOW_SECONDS + 5 })], NOW_MS)[0].requests).toBe(0);
  });
});

describe("magnitudeFills", () => {
  it("gives the largest value the darkest step on a light surface", () => {
    expect(magnitudeFills(3, { dark: false, ascending: true })[0]).toBe(SEQUENTIAL_COLOR_RAMP[0]);
  });

  it("flips the ramp in dark mode so the largest value stays visible", () => {
    const fills = magnitudeFills(3, { dark: true, ascending: true });

    expect(fills[0]).toBe(SEQUENTIAL_COLOR_RAMP[SEQUENTIAL_COLOR_RAMP.length - 1]);
  });

  it("clamps to the last ramp step when there are more rows than steps", () => {
    const fills = magnitudeFills(SEQUENTIAL_COLOR_RAMP.length + 4, { dark: false, ascending: true });

    expect(fills).toHaveLength(SEQUENTIAL_COLOR_RAMP.length + 4);
    expect(fills[fills.length - 1]).toBe(SEQUENTIAL_COLOR_RAMP[SEQUENTIAL_COLOR_RAMP.length - 1]);
  });
});

describe("chartHeight", () => {
  it("grows with the row count between a floor and a ceiling", () => {
    expect(chartHeight(0)).toBe(200);
    expect(chartHeight(5)).toBe(258);
    expect(chartHeight(100)).toBe(440);
  });
});
