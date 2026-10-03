import { describe, expect, it } from "vitest";

import { runRequest } from "./RunNowDialog";

const base = { agent: "support", saved: "support", start: "", end: "" };

describe("runRequest", () => {
  it("sends nothing extra for the default since-last-run choice", () => {
    expect(runRequest({ ...base, preset: null })).toEqual({});
  });

  it("only overrides the agent when it differs from the saved one", () => {
    expect(runRequest({ ...base, preset: null, agent: "billing" })).toEqual({ agent_name: "billing" });
    expect(runRequest({ ...base, preset: null, agent: "  support " })).toEqual({});
  });

  it("turns a preset into a lookback window", () => {
    expect(runRequest({ ...base, preset: 168 })).toEqual({ lookback_hours: 168 });
  });

  it("sends an exact window as ISO timestamps for a custom range", () => {
    const request = runRequest({ ...base, preset: -1, start: "2026-10-02T10:00", end: "2026-10-02T12:00" });
    expect(request).toEqual({
      start: new Date("2026-10-02T10:00").toISOString(),
      end: new Date("2026-10-02T12:00").toISOString(),
    });
  });

  it("rejects a missing or backwards custom range with a message instead of a request", () => {
    expect(runRequest({ ...base, preset: -1, start: "", end: "2026-10-02T12:00" })).toBe("Choose a start and end time");
    expect(runRequest({ ...base, preset: -1, start: "2026-10-02T12:00", end: "2026-10-02T10:00" })).toBe(
      "Start time must be before end time",
    );
  });
});
