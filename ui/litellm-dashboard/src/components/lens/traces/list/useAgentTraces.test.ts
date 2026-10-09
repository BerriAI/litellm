import { describe, expect, it } from "vitest";

import { spanLogWindow } from "../detail/useSpanRequestLog";

describe("spanLogWindow", () => {
  it("looks up the request log around the span's own time, not the logs tab window", () => {
    const spanStart = Date.parse("2026-08-01T12:00:00Z");
    expect(spanLogWindow(spanStart)).toEqual({ start_date: "2026-08-01 11:30:00", end_date: "2026-08-01 12:30:00" });
  });
});
