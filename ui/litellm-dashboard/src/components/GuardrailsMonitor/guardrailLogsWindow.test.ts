import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { toUtcInstantRange } from "./guardrailLogsWindow";

// The bug is invisible under UTC: there, the local day and the UTC day coincide and a
// naive implementation passes. Every case below pins a non-UTC zone so a regression
// fails instead of silently agreeing.
const withTimezone = (zone: string) => {
  const original = process.env.TZ;
  beforeAll(() => {
    process.env.TZ = zone;
  });
  afterAll(() => {
    process.env.TZ = original;
  });
};

describe("toUtcInstantRange, east of UTC (Asia/Kolkata, +05:30)", () => {
  withTimezone("Asia/Kolkata");

  it("resolves the local day to instants rather than the UTC day", () => {
    expect(toUtcInstantRange("2026-08-10", "2026-08-10")).toEqual({
      start: "2026-08-09T18:30:00.000Z",
      end: "2026-08-10T18:29:59.999Z",
    });
  });

  it("keeps a row just after local midnight inside the range", () => {
    const { start, end } = toUtcInstantRange("2026-08-10", "2026-08-10");
    // 2026-08-09T20:00:00Z is 01:30 on 2026-08-10 in IST, so it belongs to that local day.
    // Padding the bare date to 2026-08-10T00:00:00Z, as the endpoint does, excludes it.
    const row = Date.parse("2026-08-09T20:00:00Z");
    expect(row).toBeGreaterThanOrEqual(Date.parse(start));
    expect(row).toBeLessThanOrEqual(Date.parse(end));
    expect(row).toBeLessThan(Date.parse("2026-08-10T00:00:00Z"));
    // Guards against snapping to a UTC midnight in either direction.
    expect(Date.parse(start)).toBeGreaterThan(Date.parse("2026-08-09T00:00:00Z"));
    expect(Date.parse(end)).toBeGreaterThan(Date.parse("2026-08-10T00:00:00Z"));
  });

  it("does not reach past the end of the local day", () => {
    const { end } = toUtcInstantRange("2026-08-10", "2026-08-10");
    // 23:59:59Z is 05:29 on 2026-08-11 in IST, a day the viewer did not ask for.
    expect(Date.parse(end)).toBeLessThan(Date.parse("2026-08-10T23:59:59Z"));
    // ...but it must still cover the whole local day, which a UTC-day ceiling truncates.
    expect(Date.parse(end)).toBeGreaterThan(Date.parse("2026-08-10T12:00:00Z"));
  });

  it("keeps a row in the final millisecond of the local day", () => {
    const { end } = toUtcInstantRange("2026-08-10", "2026-08-10");
    // 23:59:59.500 local. Truncating the inclusive bound to whole seconds drops it.
    expect(Date.parse(end)).toBeGreaterThanOrEqual(Date.parse("2026-08-10T18:29:59.500Z"));
  });

  it("spans a multi-day range end to end", () => {
    expect(toUtcInstantRange("2026-08-03", "2026-08-10")).toEqual({
      start: "2026-08-02T18:30:00.000Z",
      end: "2026-08-10T18:29:59.999Z",
    });
  });
});

describe("toUtcInstantRange, west of UTC (America/Los_Angeles, -07:00 in August)", () => {
  withTimezone("America/Los_Angeles");

  it("resolves the local day to instants rather than the UTC day", () => {
    expect(toUtcInstantRange("2026-08-10", "2026-08-10")).toEqual({
      start: "2026-08-10T07:00:00.000Z",
      end: "2026-08-11T06:59:59.999Z",
    });
  });

  it("keeps a row late in the local day inside the range", () => {
    const { start, end } = toUtcInstantRange("2026-08-10", "2026-08-10");
    // 2026-08-11T02:00:00Z is 19:00 on 2026-08-10 in PT, still the viewer's Aug 10.
    // The endpoint's 2026-08-10T23:59:59Z ceiling would drop it.
    const row = Date.parse("2026-08-11T02:00:00Z");
    expect(row).toBeGreaterThanOrEqual(Date.parse(start));
    expect(row).toBeLessThanOrEqual(Date.parse(end));
    expect(row).toBeGreaterThan(Date.parse("2026-08-10T23:59:59Z"));
  });
});

describe("toUtcInstantRange, on UTC", () => {
  withTimezone("UTC");

  it("matches the day the endpoint would have assumed", () => {
    expect(toUtcInstantRange("2026-08-10", "2026-08-10")).toEqual({
      start: "2026-08-10T00:00:00.000Z",
      end: "2026-08-10T23:59:59.999Z",
    });
  });
});
