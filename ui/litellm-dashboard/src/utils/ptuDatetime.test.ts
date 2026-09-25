import dayjs from "dayjs";
import { describe, expect, it } from "vitest";
import { formatPtuUtcDisplay, ptuPickerToUtcIso, utcIsoToPickerValue } from "./ptuDatetime";

describe("ptuDatetime", () => {
  it("stores the picked wall-clock time as UTC instead of shifting across zones", () => {
    const picked = dayjs("2024-03-10T23:00:00");
    expect(ptuPickerToUtcIso(picked)).toBe("2024-03-10T23:00:00.000Z");
  });

  it("returns null for empty picker values", () => {
    expect(ptuPickerToUtcIso(null)).toBeNull();
    expect(ptuPickerToUtcIso(undefined)).toBeNull();
  });

  it("round-trips a UTC ISO string back to the same wall-clock in the picker", () => {
    const value = utcIsoToPickerValue("2024-03-10T23:00:00.000Z");
    expect(value).not.toBeNull();
    expect(value!.format("YYYY-MM-DDTHH:mm:ss")).toBe("2024-03-10T23:00:00");
    expect(ptuPickerToUtcIso(value)).toBe("2024-03-10T23:00:00.000Z");
  });

  it("returns null for empty ISO strings", () => {
    expect(utcIsoToPickerValue(null)).toBeNull();
    expect(utcIsoToPickerValue(undefined)).toBeNull();
    expect(utcIsoToPickerValue("")).toBeNull();
  });
});

describe("formatPtuUtcDisplay", () => {
  it("renders the two stored serialisations identically", () => {
    // the backend writes +00:00, a just-saved form holds the picker's .000Z
    expect(formatPtuUtcDisplay("2026-08-01T23:00:00+00:00")).toBe("2026-08-01 23:00:00 UTC");
    expect(formatPtuUtcDisplay("2026-08-01T23:00:00.000Z")).toBe("2026-08-01 23:00:00 UTC");
  });

  it("shows the UTC instant regardless of the offset it was written with", () => {
    expect(formatPtuUtcDisplay("2026-08-01T16:00:00-07:00")).toBe("2026-08-01 23:00:00 UTC");
  });

  it("returns null for empty values so the caller can fall back to Not Set", () => {
    expect(formatPtuUtcDisplay(null)).toBeNull();
    expect(formatPtuUtcDisplay(undefined)).toBeNull();
    expect(formatPtuUtcDisplay("")).toBeNull();
  });

  it("passes an unparseable value through rather than hiding it", () => {
    expect(formatPtuUtcDisplay("not-a-date")).toBe("not-a-date");
  });
});

describe("DST spring-forward gap", () => {
  // 2027-03-14 02:30 does not exist in America/Los_Angeles: the clock jumps 02:00 -> 03:00.
  const GAP_ISO = "2027-03-14T02:30:00+00:00";

  it("keeps the stored wall clock when it falls in the local DST gap", () => {
    const picked = utcIsoToPickerValue(GAP_ISO);
    expect(picked).not.toBeNull();
    expect(picked!.format("YYYY-MM-DDTHH:mm:ss")).toBe("2027-03-14T02:30:00");
  });

  it("round-trips the gap instant back out unchanged, so a save cannot shift it", () => {
    expect(ptuPickerToUtcIso(utcIsoToPickerValue(GAP_ISO))).toBe("2027-03-14T02:30:00.000Z");
  });

  it("round-trips a fall-back ambiguous instant unchanged too", () => {
    // 2027-11-07 01:30 occurs twice in America/Los_Angeles
    const AMBIGUOUS = "2027-11-07T01:30:00+00:00";
    expect(ptuPickerToUtcIso(utcIsoToPickerValue(AMBIGUOUS))).toBe("2027-11-07T01:30:00.000Z");
  });

  it("returns null for an unparseable stored value instead of an Invalid Date picker", () => {
    expect(utcIsoToPickerValue("not-a-date")).toBeNull();
  });
});

describe("sub-second precision", () => {
  // The backend persists value.isoformat() verbatim, so a window set out of band (curl,
  // which is this repo's documented setup path) can carry microseconds. Every save re-sends
  // both window fields, so a lossy round-trip would rewrite an untouched billing window.
  it("preserves a sub-second stored instant through a save", () => {
    const stored = "2026-08-01T23:00:00.500000+00:00";
    expect(ptuPickerToUtcIso(utcIsoToPickerValue(stored))).toBe("2026-08-01T23:00:00.500Z");
  });

  it("keeps the sub-second component visible on the picker value", () => {
    expect(utcIsoToPickerValue("2026-08-01T23:00:00.500000+00:00")!.millisecond()).toBe(500);
  });

  it("still reinterprets a freshly picked local-mode value as UTC", () => {
    // a value the operator picks has no sub-second part and must not be zone-converted
    const localPick = dayjs("2026-08-01T23:00:00");
    expect(localPick.isUTC()).toBe(false);
    expect(ptuPickerToUtcIso(localPick)).toBe("2026-08-01T23:00:00.000Z");
  });
});
