import { describe, expect, it } from "vitest";
import { agoLabel, durationLabel, durationText, money, scopeLabel, when } from "./format";

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

describe("agoLabel", () => {
  const now = Date.UTC(2026, 9, 3, 12, 0, 0);

  it("rounds down to the largest whole unit", () => {
    expect(agoLabel(now - 2_000, now)).toBe("just now");
    expect(agoLabel(now - 45_000, now)).toBe("45s ago");
    expect(agoLabel(now - 12 * 60_000 - 59_000, now)).toBe("12m ago");
    expect(agoLabel(now - 3 * 3_600_000, now)).toBe("3h ago");
    expect(agoLabel(now - 2 * 86_400_000, now)).toBe("2d ago");
  });

  it("treats a timestamp slightly in the future as just now instead of negative", () => {
    expect(agoLabel(now + 10_000, now)).toBe("just now");
  });
});
