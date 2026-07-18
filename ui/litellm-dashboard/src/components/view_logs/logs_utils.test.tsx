import moment from "moment";
import { describe, expect, it } from "vitest";
import { getTimeRangeDisplay } from "./logs_utils";

// startTime built relative to "now"; getTimeRangeDisplay computes now() internally.
const ago = (amount: number, unit: moment.unitOfTime.DurationConstructor) =>
  moment().subtract(amount, unit).toISOString();

describe("getTimeRangeDisplay", () => {
  it("labels a ~1-minute window as 'Last 1 Minute'", () => {
    expect(getTimeRangeDisplay(false, ago(1, "minutes"), "")).toBe("Last 1 Minute");
  });

  it("labels a ~10-minute window as 'Last 15 Minutes'", () => {
    expect(getTimeRangeDisplay(false, ago(10, "minutes"), "")).toBe("Last 15 Minutes");
  });

  it("labels a ~30-minute window as 'Last Hour'", () => {
    expect(getTimeRangeDisplay(false, ago(30, "minutes"), "")).toBe("Last Hour");
  });

  it("labels a ~2-hour window as 'Last 5 Hours'", () => {
    expect(getTimeRangeDisplay(false, ago(2, "hours"), "")).toBe("Last 5 Hours");
  });

  it("labels a ~10-hour window as 'Last 24 Hours'", () => {
    expect(getTimeRangeDisplay(false, ago(10, "hours"), "")).toBe("Last 24 Hours");
  });

  it("labels a ~3-day window as 'Last 7 Days'", () => {
    expect(getTimeRangeDisplay(false, ago(3, "days"), "")).toBe("Last 7 Days");
  });

  it("falls back to a 'MMM D - MMM D' range beyond 7 days", () => {
    const label = getTimeRangeDisplay(false, ago(30, "days"), "");
    expect(label).toMatch(/^[A-Z][a-z]{2} \d{1,2} - [A-Z][a-z]{2} \d{1,2}$/);
  });

  it("renders an explicit start - end range when isCustomDate is true", () => {
    const start = "2025-01-02T03:04:00Z";
    const end = "2025-01-05T06:07:00Z";
    const expected = `${moment(start).format("MMM D, h:mm A")} - ${moment(end).format("MMM D, h:mm A")}`;
    expect(getTimeRangeDisplay(true, start, end)).toBe(expected);
  });
});
