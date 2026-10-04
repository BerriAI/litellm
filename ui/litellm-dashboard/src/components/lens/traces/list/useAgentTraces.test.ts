import { describe, expect, it } from "vitest";

import { traceWindowStartMs } from "./useAgentTraces";
import { spanLogWindow } from "../detail/useSpanRequestLog";

const HOUR = 3600 * 1000;

describe("traceWindowStartMs", () => {
  it("rolls a preset range forward with now so live tail keeps a fixed-length window", () => {
    const start = "2026-09-29T10:00";
    const end = "2026-09-30T10:00";
    const mountedAt = Date.parse("2026-09-30T10:00:00");
    const tenHoursLater = mountedAt + 10 * HOUR;
    expect(tenHoursLater - traceWindowStartMs(start, end, false, tenHoursLater)).toBe(24 * HOUR);
    expect(traceWindowStartMs(start, end, false, tenHoursLater)).toBeGreaterThan(
      traceWindowStartMs(start, end, false, mountedAt),
    );
  });

  it("keeps a custom range pinned to what the user picked", () => {
    const start = "2026-09-01T00:00";
    expect(traceWindowStartMs(start, "2026-09-02T00:00", true, Date.parse("2026-09-30T00:00:00"))).toBe(
      Date.parse(start),
    );
  });
});

describe("spanLogWindow", () => {
  it("looks up the request log around the span's own time, not the logs tab window", () => {
    const spanStart = Date.parse("2026-08-01T12:00:00Z");
    expect(spanLogWindow(spanStart)).toEqual({ start_date: "2026-08-01 11:30:00", end_date: "2026-08-01 12:30:00" });
  });
});
