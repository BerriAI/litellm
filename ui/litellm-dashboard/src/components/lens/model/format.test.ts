import { describe, expect, it } from "vitest";
import { durationLabel, durationText, money, scopeLabel, when } from "./format";

describe("Lens labels", () => {
  it("formats an empty scope and joins recorded scope constraints in their original order", () => {
    expect(scopeLabel({})).toBe("All activity");
    const settings = { agent_name: "research", service: "shared", filters: [{ key: "team", value: "quality" }] };
    expect(scopeLabel(settings)).toBe("research · shared · team: quality");
  });

  it("formats elapsed time and treats a non-finite duration as zero", () => {
    expect(durationText(NaN)).toBe("0s");
    expect(durationText(59)).toBe("59s");
    expect(durationText(61)).toBe("1m 1s");
    expect(durationText(3661)).toBe("1h 1m");
  });

  it("keeps the units and singular forms of sampling and monitoring windows", () => {
    expect(durationLabel(1)).toBe("1 minute");
    expect(durationLabel(60)).toBe("1 hour");
    expect(durationLabel(90)).toBe("90 minutes");
    expect(durationLabel(24, "hours")).toBe("1 day");
    expect(durationLabel(36, "hours")).toBe("1.5 days");
  });

  it("keeps USD precision and the placeholder for a scan that has not run", () => {
    expect(money(12.12345)).toBe("$12.123");
    expect(money(0)).toBe("$0.00");
    expect(when()).toBe("Not yet");
    expect(when(null)).toBe("Not yet");
  });
});
